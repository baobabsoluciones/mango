import logging
import os
from datetime import datetime
from typing import Union, Any, Tuple, Literal

from mango.clients.rest_client import RESTClient
from mango.shared import ApiKeyError, haversine
from mango.validators.aemet import *
from tqdm import tqdm


class AEMETClient(RESTClient):
    """
    This class will handle the connection to the AEMET API. It will allow to get the meteorological data from the
    meteorological stations in Spain. To initialize the class, an API key is needed. This key can be obtained in the
    AEMET website: https://opendata.aemet.es/centrodedescargas/altaUsuario

    :param api_key: API key to connect to the AEMET API
    :param wait_time: Wait time between requests in seconds. Default is 1.25 seconds
    :doc-author: baobab soluciones

    Usage
    -----

    >>> import os
    >>> from mango.clients.aemet import AEMETClient
    >>> client = AEMETClient(api_key=os.environ["AEMET_API_KEY"])
    """

    def __init__(self, *, api_key: str, wait_time: Union[float, int] = None):
        super().__init__()
        # Default values
        self._DEFAULT_WAIT_TIME = float(os.environ.get("AEMET_DEFAULT_WAIT_TIME", 1.25))
        self._FETCH_STATIONS_URL = os.environ.get(
            "AEMET_FETCH_STATIONS_URL",
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones",
        )
        self._FETCH_MUNICIPIOS_URL = os.environ.get(
            "AEMET_FETCH_MUNICIPIOS_URL",
            "https://opendata.aemet.es/opendata/api/maestro/municipios",
        )
        self._FETCH_HISTORIC_URL = os.environ.get(
            "AEMET_FETCH_HISTORIC_URL",
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini"
            "/{start_date_str}/fechafin/{end_date_str}"
            "/estacion/{station}",
        )
        self._FETCH_DAILY_URL = os.environ.get(
            "AEMET_FETCH_DAILY_URL",
            "https://opendata.aemet.es/opendata/api/observacion/convencional/datos/estacion/{station}",
        )
        self._FETCH_FORECAST_URL = os.environ.get(
            "AEMET_FETCH_FORECAST_URL",
            "https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{postal_code}",
        )
        self._api_key = api_key
        self._wait_time = wait_time or self._DEFAULT_WAIT_TIME
        logging.getLogger("root")
        if self._wait_time < self._DEFAULT_WAIT_TIME:
            logging.warning(
                "Wait time is too low. API limits may be exceeded. Minimum wait time is {} seconds.".format(
                    self._DEFAULT_WAIT_TIME
                )
            )
        logging.info(
            "Aemet API has a limit of 50 requests per minute. Please be patient. Wait time is set to {} seconds.".format(
                self._wait_time
            )
        )

    def connect(self) -> None:
        """
        This method will connect to the AEMET API and cache the stations and municipios.
        Checks if the API key is valid.
        :doc-author: baobab soluciones
        """
        # Allows to check if the api_key is valid and cache the stations
        try:
            self._all_stations = self._get_all_stations()
        except ApiKeyError as e:
            raise e
        # Setup municipios
        self.municipios = self._get_municipios()

    @property
    def all_stations(self) -> List[dict]:
        """
        This property will return all the meteorological stations in Spain.

        :return: List of dictionaries with the meteorological stations
        :doc-author: baobab soluciones
        """
        return self._all_stations

    @property
    def municipios(self) -> List[dict]:
        """
        This property will return all the municipios in Spain.

        :return: List of dictionaries with the municipios
        :doc-author: baobab soluciones
        """
        return self._municipios

    @municipios.setter
    def municipios(self, value) -> None:
        """
        Keep only the relevant information for the municipios
        """
        tmp = []
        for m in value:
            m["codigo postal"] = m["id"].lstrip("id")
            tmp.append(
                {
                    k: v
                    for k, v in m.items()
                    if k in ["codigo postal", "nombre", "longitud_dec", "latitud_dec"]
                }
            )
        self._municipios = tmp

    @staticmethod
    def _parse_lat_long(lat_long: Tuple[str, str]) -> Tuple[float, float]:
        """
        This function will parse the latitude and longitude strings returned by the AEMET API to a tuple of floats
        """
        """
        The parse_lat_long function parses the latitude and longitude of a tuple.
        :param lat_long_tuple: Tuple with latitude and longitude as strings in the format DDMMSSX
        :return: Tuple with latitude and longitude as floats in the format DD.XXXXXX * -1 if S or W
        :doc-author: baobab soluciones
        """
        lat, long = lat_long
        sign_dict = {"N": 1, "S": -1, "E": 1, "W": -1}
        # Is written in degrees, minutes and seconds concatenated
        lat = sign_dict[lat[-1]] * (
            float(lat[:2]) + float(lat[2:4]) / 60 + float(lat[4:6]) / 3600
        )
        long = (
            sign_dict[long[-1]] * float(long[:2])
            + float(long[2:4]) / 60
            + float(long[4:6]) / 3600
        )
        return lat, long

    def _get_municipios(self) -> List[dict]:
        """
        This function will search for all the municipios in Spain.
        :return: List with all the municipios in Spain
        :doc-author: baobab soluciones
        """
        municipios = self.request_handler(
            self._FETCH_MUNICIPIOS_URL,
            params={"api_key": self._api_key},
            wait_time=self._wait_time,
            expected_schema=FetchMunicipiosResponse,
        )
        return municipios

    def _get_stations_filtered(
        self,
        station_code: str = None,
        lat: Union[float, int] = None,
        long: Union[float, int] = None,
        province: str = None,
    ) -> List[str]:
        """
        This function will filter the stations given the parameters with higher priority.
        First it will check if station_code is provided. If it is, it will return only that station.
        If station_code is not provided, it will check if lat and long are provided. If they are, it will return the
        closest station to that point.
        If lat, long and province are provided, it will return the closest station to that point in that province.
        If lat and long are not provided, it will check if province is provided. If it is, it will return all the
        stations in that province.
        If province is not provided, it will return all the stations in Spain.

        :param station_code: meteorological station code
        :param lat: latitude
        :param long: longitude
        :param province: province
        :return: List of station codes
        """
        if station_code:
            logging.info(
                "Selecting only station: " + station_code + " ignoring other parameters"
            )
            if station_code not in [s["indicativo"] for s in self.all_stations]:
                raise ValueError(
                    f"{station_code} not found in Spain. Possible values: {[s['indicativo'] for s in self.all_stations]}"
                )
            # Use list for consistency in code
            station_codes = [station_code]
        elif lat and long:
            # Search for closest station
            logging.info(
                "Searching for closest station to lat: "
                + str(lat)
                + " long: "
                + str(long)
            )
            station_codes = self._search_closest_station(lat, long, province)
        elif province:
            # Search for all stations in province
            logging.info("Searching for all stations in province: " + province)
            possible_provinces = list(
                set([stat.get("provincia").lower() for stat in self._all_stations])
            )
            if province.lower() not in possible_provinces:
                logging.warning(
                    f"Province not found in Spain. Possible values: {possible_provinces}"
                )
                logging.warning("Searching in all Spain")
                station_codes = [est["indicativo"] for est in self._all_stations]
            else:
                station_codes = [
                    est["indicativo"]
                    for est in self._search_stations_province(province)
                ]
        else:
            # Search for all stations in Spain
            logging.info("Searching for all stations in Spain")
            station_codes = [est["indicativo"] for est in self.all_stations]
        return station_codes

    def _search_closest_station(
        self, lat: Union[float, int], long: Union[float, int], province: str = None
    ) -> List[dict]:
        """
        This function will search for the closest meteorological station to the given point.
        If province is provided, it will search for the closest station in that province.
        If province is not provided, it will search for the closest station in all Spain.
        :param lat: Latitude of the point in degrees
        :param long: Longitude of the point in degrees
        :param province: Province to search for the closest station
        :return: List with the closest station. Only one but to mantain consistency embedded in list
        :doc-author: baobab soluciones
        """
        # Default to all stations
        stations_to_search = self._all_stations
        if province:
            possible_provinces = list(
                set([stat.get("provincia").lower() for stat in self._all_stations])
            )
            if province.lower() not in possible_provinces:
                logging.warning(
                    f"Province not found in Spain. Possible values: {possible_provinces}"
                )
                logging.warning("Searching in all Spain")
            else:
                stations_to_search = self._search_stations_province(province)
        lat_long_list = [
            (est["indicativo"], self._parse_lat_long((est["latitud"], est["longitud"])))
            for est in stations_to_search
        ]
        objective_lat_long = (lat, long)
        distances = [
            (
                station_indicative,
                haversine(point1=objective_lat_long, point2=station_coords),
            )
            for station_indicative, station_coords in lat_long_list
        ]
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        station = distances[0][0]
        # Only one element however to maintain consistency embed in list
        return [station]

    def _search_stations_province(self, province: str) -> List[dict]:
        """
        This function will search for all the meteorological stations in the given province.
        :param province: Province to search for the closest station
        :return: List with all the stations in the given province
        :doc-author: baobab soluciones
        """
        all_stations = self._all_stations
        stations = [
            stat
            for stat in all_stations
            if stat.get("provincia").lower() == province.lower()
        ]
        return stations

    def _get_all_stations(self) -> List[dict]:
        """
        This function will search for all the meteorological stations in Spain.
        :return: List with all the stations in Spain
        :doc-author: baobab soluciones
        """
        all_stations_call = self.request_handler(
            self._FETCH_STATIONS_URL,
            params={"api_key": self._api_key},
            wait_time=self._wait_time,
            expected_schema=UrlCallResponse,
        )
        # AEMET API returns a URL to get the data
        all_station_res_url = all_stations_call.get("datos")
        # Fetch the data
        stations = self.request_handler(
            all_station_res_url,
            params={},
            wait_time=self._wait_time,
            expected_schema=FetchStationsResponse,
        )
        return stations

    def _get_historical_data(
        self, station_codes: List[str], start_date: datetime, end_date: datetime
    ) -> List[dict]:
        """
        This function will get the historical data from the given stations between the given dates.
        :param station_codes: List of station codes to get the data from
        :param start_date: Start date of the data to get
        :param end_date: End date of the data to get
        :return: List with the historical data
        :doc-author: baobab soluciones
        """
        data = []
        for station in tqdm(station_codes):
            # Make it in a loop to make it easier to debug errors in HTTP requests
            hist_data_url_call = self.request_handler(
                self._FETCH_HISTORIC_URL.format(
                    start_date_str=start_date.strftime("%Y-%m-%dT%H:%M:%SUTC"),
                    end_date_str=end_date.strftime("%Y-%m-%dT%H:%M:%SUTC"),
                    station=station,
                ),
                params={"api_key": self._api_key},
                wait_time=self._wait_time,
                if_error="warn",
                expected_schema=UrlCallResponse,
            )
            hist_data_url = hist_data_url_call.get("datos")
            if not hist_data_url:
                logging.warning(
                    f"No data found for station: {station} between {start_date} and {end_date}"
                )
                continue
            hist_data = self.request_handler(
                hist_data_url,
                params={},
                wait_time=self._wait_time,
                if_error="warn",
                expected_schema=FetchHistoricResponse,
            )
            data.extend(hist_data)
        if not data:
            raise Exception("Failed for all stations")
        return data

    def _get_live_data(self, station_codes: List[str]) -> List[dict]:
        """
        This function will get the live data from the given stations.
        :param station_codes: List of station codes to get the data from
        :return: List with the live data
        :doc-author: baobab soluciones
        """
        data = []
        for station in tqdm(station_codes):
            live_data_url_call = self.request_handler(
                self._FETCH_DAILY_URL.format(station=station),
                params={"api_key": self._api_key},
                wait_time=self._wait_time,
                if_error="warn",
                expected_schema=UrlCallResponse,
            )
            live_data_url = live_data_url_call.get("datos")
            if not live_data_url:
                logging.warning(f"No data found for station: {station}")
                continue
            live_data = self.request_handler(
                live_data_url,
                params={},
                wait_time=self._wait_time,
                if_error="warn",
            )
            data.extend(live_data)
        if not data:
            raise Exception("Failed for all stations")
        return data

    def _format_data(self, data: List[dict], output_format) -> Any:
        """
        This function will format the data to the desired output format
        """
        if output_format == "df":
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("Pandas is required to return a dataframe")
            # Flatten list of lists of dicts to list of dicts
            data = pd.DataFrame(data)
        return data

    def get_meteo_data(
        self,
        station_code: str = None,
        lat: float = None,
        long: float = None,
        province: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        output_format: Literal["df", "raw"] = "raw",
    ):
        """
        Main method of the class. This method will return the meteorological data from the meteorological stations in
        Spain.

        :param str station_code: meteorological station code
        :param float lat: latitude
        :param float long: longitude
        :param str province: province
        :param datetime start_date: start date
        :param datetime end_date: end date
        :return: a list of dictionaries with the meteorological data
        :doc-author: baobab soluciones

        Usage
        -----

        >>> import pandas as pd
        >>> from datetime import datetime
        >>> from mango.clients.aemet import AEMETClient
        >>> import os
        >>>
        >>> client = AEMETClient(api_key=os.environ["AEMET_API_KEY"])
        >>> client.connect()

        If lat and long are provided this method will search for the closest meteorological station to that point.
        Province is not necessary however it will speed up the search.

        >>> data = client.get_meteo_data(lat=40.4165, long=-3.70256, province="Madrid", start_date=datetime(2021, 1, 1), end_date=datetime(2021, 1, 31), output_format="df")

        If lat and long are not provided but province is provided, this method will search for all the meteorological
        stations in that province.

        >>> data = client.get_meteo_data(province="Madrid", start_date=datetime(2021, 1, 1), end_date=datetime(2021, 1, 31), output_format="df")

        If lat and long are not provided and province is not provided, this method will search for all the meteorological
        stations in Spain.

        >>> data = client.get_meteo_data(start_date=datetime(2021, 1, 1), end_date=datetime(2021, 1, 31), output_format="df")

        If neither start_date nor end_date are provided, this method will search for live data following the rules
        described above. end_date needs start_date to be provided.

        >>> data = client.get_meteo_data(lat=40.4165, long=-3.70256, province="Madrid", output_format="df")


        """
        # Check datatype
        if station_code and not isinstance(station_code, str):
            raise TypeError("indicativo must be a string")
        if lat and not isinstance(lat, (float, int)):
            raise TypeError("lat must be a float or an int")
        if long and not isinstance(long, (float, int)):
            raise TypeError("long must be a float or an int")
        if province and not isinstance(province, str):
            raise TypeError("province must be a string")
        if start_date and not isinstance(start_date, datetime):
            raise TypeError("start_date must be a datetime")
        if end_date and not isinstance(end_date, datetime):
            raise TypeError("end_date must be a datetime")
        if output_format not in ["df", "raw"]:
            raise NotImplementedError(
                f"output_format {output_format} not implemented. Only dataframe and raw"
            )
        # Ensure params consistency
        if not station_code and not lat and not long:
            raise ValueError("station_code or lat and long must be provided")
        if start_date and not end_date:
            end_date = datetime.now()
            logging.warning("end_date not provided. Using current date as end_date")
        if end_date and not start_date:
            raise ValueError("end_date provided but not start_date")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be before end_date")
        if lat and not long:
            raise ValueError("lat and long must be provided together")
        if long and not lat:
            raise ValueError("lat and long must be provided together")

        # Decision logic for search given parameters
        station_codes_filtered = self._get_stations_filtered(
            station_code,
            lat,
            long,
            province,
        )
        if start_date:
            # Search for data between start_date and end_date
            data = self._get_historical_data(
                station_codes_filtered, start_date, end_date
            )
        else:
            logging.info(
                "As start_date is not provided, last hours data will be returned"
            )
            data = self._get_live_data(station_codes_filtered)

        # Return data
        return self._format_data(data, output_format)

    def get_forecast_data(self, postal_code: str = None):
        """
        This method will return the forecast data for the given postal code for the next days
        provided by the AEMET API. If postal_code is not provided, it will return the forecast data for all the
        municipios in Spain, doing so takes a long time due to API limitations.

        In this version it returns the raw data from the API. In future versions it will return a dataframe with the
        forecast data properly formatted.

        :param postal_code: Postal code of the municipio to get the forecast data from
        :return: List of dictionaries with the forecast data
        """
        if postal_code and not isinstance(postal_code, str):
            raise TypeError("postal_code must be a string")
        if postal_code:
            assert postal_code in [
                m["codigo postal"] for m in self.municipios
            ], "Municipio not found"
            postal_codes = [postal_code]
        else:
            postal_codes = [m["codigo postal"] for m in self.municipios]
        data = []
        for postal_code in tqdm(postal_codes):
            forecast_url_call = self.request_handler(
                f"https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{postal_code}",
                params={"api_key": self._api_key},
                wait_time=self._wait_time,
                if_error="warn",
                expected_schema=UrlCallResponse,
            )
            forecast_url = forecast_url_call.get("datos", None)
            if not forecast_url:
                logging.warning(f"No data found for postal_code: {postal_code}")
                continue
            forecast_data = self.request_handler(
                forecast_url, params={}, wait_time=self._wait_time, if_error="warn"
            )
            data.append(
                {
                    k: v
                    for k, v in forecast_data[0].items()
                    if k in ["nombre", "provincia", "elaborado", "prediccion"]
                }
            )
        if not data:
            raise Exception("Failed for all municipios")
        return data

    def custom_endpoint(self, endpoint: str):
        # Check datatype
        if not isinstance(endpoint, str):
            raise TypeError("endpoint must be a string")
        # Check it starts with /api
        if not endpoint.startswith("/api"):
            raise ValueError("endpoint must start with /api")
        custom_endpoint_url_call = self.request_handler(
            "https://opendata.aemet.es/opendata" + endpoint,
            params={"api_key": self._api_key},
            wait_time=self._wait_time,
            if_error="warn",
            expected_schema=UrlCallResponse,
        )
        custom_enpoint_url = custom_endpoint_url_call.get("datos")
        if not custom_enpoint_url:
            logging.info(f"No data found for endpoint: {endpoint}")
            raise ValueError(f"No data found for endpoint: {endpoint}")
        custom_enpoint_data = self.request_handler(
            custom_enpoint_url, params={}, wait_time=self._wait_time, if_error="warn"
        )
        return custom_enpoint_data
