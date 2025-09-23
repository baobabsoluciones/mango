import os
from datetime import datetime
from typing import Union, Any, Tuple, Literal, List

from mango.clients.rest_client import RESTClient
from mango.logging import get_configured_logger
from mango.shared import ApiKeyError, haversine
from mango.validators.aemet import *
from tqdm import tqdm

log = get_configured_logger(__name__)


class AEMETClient(RESTClient):
    """
    Client for accessing AEMET (Spanish Meteorological Agency) API.

    This class provides access to meteorological data from weather stations across Spain.
    It supports retrieving historical data, live observations, and weather forecasts.
    An API key is required and can be obtained from the AEMET website.

    :param api_key: API key to connect to the AEMET API
    :type api_key: str
    :param wait_time: Wait time between requests in seconds. Default is 1.25 seconds
    :type wait_time: Union[float, int], optional
    :raises ValueError: If wait_time is below the minimum required value

    Example:
        >>> import os
        >>> from mango.clients.aemet import AEMETClient
        >>> client = AEMETClient(api_key=os.environ["AEMET_API_KEY"])
        >>> client.connect()
        >>> data = client.get_meteo_data(lat=40.4165, long=-3.70256)
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
        if self._wait_time < self._DEFAULT_WAIT_TIME:
            log.warning(
                "Wait time is too low. API limits may be exceeded. Minimum wait time is {} seconds.".format(
                    self._DEFAULT_WAIT_TIME
                )
            )
        log.info(
            "Aemet API has a limit of 50 requests per minute. Please be patient. Wait time is set to {} seconds.".format(
                self._wait_time
            )
        )

    def connect(self) -> None:
        """
        Connect to the AEMET API and cache station and municipality data.

        This method establishes connection to the AEMET API, validates the API key,
        and caches all available weather stations and municipalities in Spain.
        This is a prerequisite for using other methods in the class.

        :raises ApiKeyError: If the provided API key is invalid
        :raises Exception: If connection to the API fails

        Example:
            >>> client = AEMETClient(api_key="your_api_key")
            >>> client.connect()
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
        Get all meteorological stations in Spain.

        Returns a list of dictionaries containing information about all
        available weather stations in Spain, including their codes, names,
        locations, and other metadata.

        :return: List of dictionaries with meteorological station information
        :rtype: List[dict]

        Example:
            >>> stations = client.all_stations
            >>> print(f"Total stations: {len(stations)}")
        """
        return self._all_stations

    @property
    def municipios(self) -> List[dict]:
        """
        Get all municipalities in Spain.

        Returns a list of dictionaries containing information about all
        municipalities in Spain, including postal codes, names, and coordinates.

        :return: List of dictionaries with municipality information
        :rtype: List[dict]

        Example:
            >>> municipalities = client.municipios
            >>> print(f"Total municipalities: {len(municipalities)}")
        """
        return self._municipios

    @municipios.setter
    def municipios(self, value) -> None:
        """
        Set municipalities data with filtered information.

        Filters the municipality data to keep only relevant fields:
        postal code, name, longitude, and latitude.

        :param value: List of municipality dictionaries from AEMET API
        :type value: List[dict]
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
        Parse latitude and longitude strings from AEMET API format.

        Converts latitude and longitude strings from the AEMET API format (DDMMSSX)
        to decimal degrees. The format uses degrees, minutes, and seconds concatenated
        with a direction indicator (N/S for latitude, E/W for longitude).

        :param lat_long: Tuple containing latitude and longitude strings
        :type lat_long: Tuple[str, str]
        :return: Tuple with latitude and longitude as decimal degrees
        :rtype: Tuple[float, float]

        Example:
            >>> coords = AEMETClient._parse_lat_long(("402500N", "0034200W"))
            >>> print(coords)  # (40.4167, -3.7)
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
        Retrieve all municipalities in Spain from AEMET API.

        Fetches the complete list of municipalities in Spain with their
        associated metadata including postal codes and coordinates.

        :return: List of dictionaries containing municipality information
        :rtype: List[dict]
        :raises Exception: If API request fails
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
        Filter weather stations based on provided criteria with priority order.

        Station selection follows this priority:
        1. If station_code is provided, return only that station
        2. If lat/long are provided, find the closest station (optionally in province)
        3. If only province is provided, return all stations in that province
        4. Otherwise, return all stations in Spain

        :param station_code: Meteorological station code (e.g., "3195")
        :type station_code: str, optional
        :param lat: Latitude coordinate in decimal degrees
        :type lat: Union[float, int], optional
        :param long: Longitude coordinate in decimal degrees
        :type long: Union[float, int], optional
        :param province: Province name to limit search scope
        :type province: str, optional
        :return: List of station codes matching the criteria
        :rtype: List[str]
        :raises ValueError: If station_code is not found in available stations
        """
        if station_code:
            log.info(
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
            log.info(
                "Searching for closest station to lat: "
                + str(lat)
                + " long: "
                + str(long)
            )
            station_codes = self._search_closest_station(lat, long, province)
        elif province:
            # Search for all stations in province
            log.info("Searching for all stations in province: " + province)
            possible_provinces = list(
                set([stat.get("provincia").lower() for stat in self._all_stations])
            )
            if province.lower() not in possible_provinces:
                log.warning(
                    f"Province not found in Spain. Possible values: {possible_provinces}"
                )
                log.warning("Searching in all Spain")
                station_codes = [est["indicativo"] for est in self._all_stations]
            else:
                station_codes = [
                    est["indicativo"]
                    for est in self._search_stations_province(province)
                ]
        else:
            # Search for all stations in Spain
            log.info("Searching for all stations in Spain")
            station_codes = [est["indicativo"] for est in self.all_stations]
        return station_codes

    def _search_closest_station(
        self, lat: Union[float, int], long: Union[float, int], province: str = None
    ) -> List[str]:
        """
        Find the closest meteorological station to the given coordinates.

        Searches for the nearest weather station using the Haversine formula to
        calculate distances. If a province is specified, the search is limited
        to stations within that province.

        :param lat: Latitude of the target point in decimal degrees
        :type lat: Union[float, int]
        :param long: Longitude of the target point in decimal degrees
        :type long: Union[float, int]
        :param province: Province name to limit search scope
        :type province: str, optional
        :return: List containing the closest station code (single element for consistency)
        :rtype: List[str]
        """
        # Default to all stations
        stations_to_search = self._all_stations
        if province:
            possible_provinces = list(
                set([stat.get("provincia").lower() for stat in self._all_stations])
            )
            if province.lower() not in possible_provinces:
                log.warning(
                    f"Province not found in Spain. Possible values: {possible_provinces}"
                )
                log.warning("Searching in all Spain")
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
        Find all meteorological stations in the specified province.

        Searches through all available stations and returns those located
        in the specified province, performing case-insensitive matching.

        :param province: Province name to search for stations
        :type province: str
        :return: List of station dictionaries in the specified province
        :rtype: List[dict]
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
        Retrieve all meteorological stations in Spain from AEMET API.

        Fetches the complete list of weather stations in Spain with their
        metadata including station codes, names, locations, and operational status.

        :return: List of dictionaries containing station information
        :rtype: List[dict]
        :raises ApiKeyError: If API key is invalid
        :raises Exception: If API request fails
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
        Retrieve historical meteorological data from specified stations.

        Fetches historical weather data for the given stations within the
        specified date range. Data is retrieved from the AEMET API with
        progress tracking and error handling.

        :param station_codes: List of station codes to retrieve data from
        :type station_codes: List[str]
        :param start_date: Start date for data retrieval
        :type start_date: datetime
        :param end_date: End date for data retrieval
        :type end_date: datetime
        :return: List of dictionaries containing historical weather data
        :rtype: List[dict]
        :raises Exception: If no data is found for any station
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
                log.warning(
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
        Retrieve live meteorological data from specified stations.

        Fetches current weather observations from the given stations.
        This provides the most recent data available from each station
        with progress tracking and error handling.

        :param station_codes: List of station codes to retrieve data from
        :type station_codes: List[str]
        :return: List of dictionaries containing live weather data
        :rtype: List[dict]
        :raises Exception: If no data is found for any station
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
                log.warning(f"No data found for station: {station}")
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
        Format meteorological data to the specified output format.

        Converts the raw data list to the requested format. Currently supports
        "raw" (list of dictionaries) and "df" (pandas DataFrame) formats.

        :param data: List of dictionaries containing meteorological data
        :type data: List[dict]
        :param output_format: Desired output format ("raw" or "df")
        :type output_format: str
        :return: Data in the specified format
        :rtype: Any
        :raises ImportError: If pandas is required but not available
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
        Retrieve meteorological data from Spanish weather stations.

        This is the main method for obtaining weather data. It supports multiple
        search criteria and can return historical or live data in different formats.

        Station selection priority:
        1. If station_code is provided, use only that station
        2. If lat/long are provided, find the closest station (optionally in province)
        3. If only province is provided, use all stations in that province
        4. Otherwise, use all stations in Spain

        :param station_code: Meteorological station code (e.g., "3195")
        :type station_code: str, optional
        :param lat: Latitude coordinate in decimal degrees
        :type lat: float, optional
        :param long: Longitude coordinate in decimal degrees
        :type long: float, optional
        :param province: Province name to limit search scope
        :type province: str, optional
        :param start_date: Start date for historical data retrieval
        :type start_date: datetime, optional
        :param end_date: End date for historical data retrieval
        :type end_date: datetime, optional
        :param output_format: Output format - "df" for DataFrame, "raw" for list of dicts
        :type output_format: Literal["df", "raw"]
        :return: Meteorological data in the specified format
        :rtype: Union[pandas.DataFrame, List[dict]]
        :raises TypeError: If parameter types are incorrect
        :raises ValueError: If parameter combinations are invalid
        :raises Exception: If no data is found for any station

        Example:
            >>> from datetime import datetime
            >>> # Get data from closest station to coordinates
            >>> data = client.get_meteo_data(
            ...     lat=40.4165, long=-3.70256,
            ...     start_date=datetime(2021, 1, 1),
            ...     end_date=datetime(2021, 1, 31),
            ...     output_format="df"
            ... )
            >>> # Get live data from specific station
            >>> data = client.get_meteo_data(
            ...     station_code="3195",
            ...     output_format="raw"
            ... )
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
            log.warning("end_date not provided. Using current date as end_date")
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
            log.info("As start_date is not provided, last hours data will be returned")
            data = self._get_live_data(station_codes_filtered)

        # Return data
        return self._format_data(data, output_format)

    def get_forecast_data(self, postal_code: str = None):
        """
        Retrieve weather forecast data for Spanish municipalities.

        This method fetches weather forecast data from the AEMET API for the next
        few days. If no postal code is specified, it retrieves forecasts for all
        municipalities in Spain (this can take a long time due to API rate limits).

        :param postal_code: Postal code of the municipality (e.g., "28001")
        :type postal_code: str, optional
        :return: List of dictionaries containing forecast data
        :rtype: List[dict]
        :raises TypeError: If postal_code is not a string
        :raises AssertionError: If postal_code is not found in available municipalities
        :raises Exception: If no forecast data is found for any municipality

        Example:
            >>> # Get forecast for specific municipality
            >>> forecast = client.get_forecast_data(postal_code="28001")
            >>> # Get forecasts for all municipalities (slow)
            >>> all_forecasts = client.get_forecast_data()
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
                log.warning(f"No data found for postal_code: {postal_code}")
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
        """
        Access custom AEMET API endpoints directly.

        Allows direct access to any AEMET API endpoint by providing the
        endpoint path. This method provides flexibility for accessing
        endpoints not covered by the standard methods.

        :param endpoint: API endpoint path (must start with "/api")
        :type endpoint: str
        :return: Raw data from the specified endpoint
        :rtype: Any
        :raises TypeError: If endpoint is not a string
        :raises ValueError: If endpoint doesn't start with "/api" or no data is found

        Example:
            >>> data = client.custom_endpoint("/api/valores/climatologicos/normales/estacion/3195")
        """
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
            log.info(f"No data found for endpoint: {endpoint}")
            raise ValueError(f"No data found for endpoint: {endpoint}")
        custom_enpoint_data = self.request_handler(
            custom_enpoint_url, params={}, wait_time=self._wait_time, if_error="warn"
        )
        return custom_enpoint_data
