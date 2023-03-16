import time
from datetime import datetime

import requests
from geopy.distance import geodesic as geo_distance
from tqdm import tqdm

from mango.shared import ApiKeyError


class Aemet:
    """
    This class will implement the AEMET API to make it easier to use meteorological data as input for
    machine learning models.
    """
    __DEFAULT_WAIT_TIME = 1.25
    def __init__(self, *, api_key: str, **kwargs):
        self._api_key: str = api_key
        self._wait_time: int = kwargs.get("wait_time", self.__DEFAULT_WAIT_TIME)
        print(
            "Aemet API has a limit of 50 requests per minute. Please be patient. Wait time is set to {} seconds.".format(
                self._wait_time
            )
        )
        if self._wait_time < self.__DEFAULT_WAIT_TIME:
            print("Wait time is too low. API limits can be exceeded.")
        # Allows to check if the api_key is valid and cache the stations
        try:
            self._all_stations = self.get_all_stations()
        except ApiKeyError as e:
            raise e

    @property
    def all_stations(self):
        return self._all_stations

    def get_meteo_data(
        self,
        indicativo: str = None,
        lat: float = None,
        long: float = None,
        province: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ):
        """
        This function will get the meteorological data from the AEMET API. Possible parameters are:

        - indicativo: meteorological station code
        - lat: latitude
        - long: longitude
        - province: province
        - start_date: start date
        - end_date: end date

        If lat and long are provided this method will search for the closest meteorological station to that point.
        Province is not necessary however it will speed up the search.

        If lat and long are not provided but province is provided, this method will search for all the meteorological
        stations in that province.

        If lat and long are not provided and province is not provided, this method will search for all the meteorological
        stations in Spain.

        If neither start_date nor end_date are provided, this method will search for live data following the rules
        described above. end_date needs start_date to be provided.

        :param str indicativo: meteorological station code
        :param float lat: latitude
        :param float long: longitude
        :param str province: province
        :param datetime start_date: start date
        :param datetime end_date: end date
        :return: a list of dictionaries with the meteorological data
        :doc-author: baobab soluciones
        """
        # Check datatype
        if indicativo and not isinstance(indicativo, str):
            raise TypeError("indicativo must be a string")
        if lat and not isinstance(lat, float):
            raise TypeError("lat must be a float")
        if long and not isinstance(long, float):
            raise TypeError("long must be a float")
        if province and not isinstance(province, str):
            raise TypeError("province must be a string")
        if start_date and not isinstance(start_date, datetime):
            raise TypeError("start_date must be a datetime")
        if end_date and not isinstance(end_date, datetime):
            raise TypeError("end_date must be a datetime")

        # Decision logic for search given parameters
        if indicativo:
            station_codes = [indicativo]
        elif lat and long:
            # Search for closest station
            # province None is handled in search_closest_station
            # Only one but to mantain consistency embed in list
            station_codes = [
                est["indicativo"]
                for est in self.search_closest_station(lat, long, province)
            ]
        elif province:
            # Search for all stations in province
            station_codes = [
                est["indicativo"] for est in self.search_stations_province(province)
            ]
        else:
            # Search for all stations in Spain
            station_codes = [est["indicativo"] for est in self.all_stations]

        if start_date and end_date:
            # Search for data between start_date and end_date
            data = self.historical_data(station_codes, start_date, end_date)
        elif start_date:
            # Search for data from start_date to now
            end_date = datetime.now()
            data = self.historical_data(station_codes, start_date, end_date)
        else:
            data = self.live_data(station_codes)

        # Return data
        return data

    @staticmethod
    def parse_lat_long(lat_long: tuple[str, str]) -> tuple[float, float]:
        """
        This function will parse the latitude and longitude returned by the AEMET API to a tuple of floats
        """
        """
        The parse_lat_long function parses the latitude and longitude of a tuple.
        :param lat_long_tuple: Tuple with latitude and longitude as strings in the format DDMMSSX
        :return: Tuple with latitude and longitude as floats in the format DD.XXXXXX * -1 if S or W
        :doc-author: baobab soluciones
        """
        lat = lat_long[0]
        long = lat_long[1]
        # Is written in degrees, minutes and seconds concatenated
        lat_deg = lat[:2]
        long_deg = long[:2]
        lat_min = lat[2:4]
        long_min = long[2:4]
        lat_sec = lat[4:6]
        long_sec = long[4:6]
        lat_sign = 1
        long_sign = 1
        if lat.__contains__("S"):
            lat_sign = -1
        if long.__contains__("W"):
            long_sign = -1
        lat = lat_sign * (float(lat_deg) + float(lat_min) / 60 + float(lat_sec) / 3600)
        long = long_sign * (
            float(long_deg) + float(long_min) / 60 + float(long_sec) / 3600
        )
        return lat, long

    def search_closest_station(self, lat, long, province=None) -> list[dict]:
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
        try:
            # If province is None will raise exception also if province is not valid
            all_stations = self.search_stations_province(province)
        except Exception as e:
            print(e)
            print("Searching for closest station in all Spain")
            all_stations = self._all_stations
        lat_long = [
            (est["indicativo"], self.parse_lat_long((est["latitud"], est["longitud"])))
            for est in all_stations
        ]
        distancias = [
            (est[0], geo_distance((lat, long), est[1]).km) for est in lat_long
        ]
        # Sort by distance
        distancias.sort(key=lambda x: x[1])
        station = distancias[0][0]
        # Only one element however to mantain consistency embed in list
        return [stat for stat in all_stations if stat["indicativo"] == station]

    def search_stations_province(self, province):
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
        if not stations:
            possible_values = sorted(
                set([stat.get("provincia") for stat in all_stations])
            )
            raise Exception(
                f"No stations found in province {province}. Possible values: {possible_values}"
            )
        return stations

    def get_all_stations(self):
        """
        This function will search for all the meteorological stations in Spain.
        :return: List with all the stations in Spain
        :doc-author: baobab soluciones
        """
        api_key = self._api_key
        estaciones_url_call = requests.get(
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"
            "/?api_key=" + api_key
        ).json()
        time.sleep(self._wait_time)
        if estaciones_url_call["estado"] == 401:
            raise ApiKeyError("Invalid API key")
        if estaciones_url_call["estado"] != 200:
            raise Exception(
                "Error getting stations from AEMET API: "
                + str(estaciones_url_call["estado"])
            )
        estaciones_url = estaciones_url_call.get("datos")
        if not estaciones_url.startswith("https://opendata.aemet.es"):
            raise Exception("Error in the url returned by AEMET " + estaciones_url)
        estaciones = requests.get(estaciones_url).json()
        time.sleep(self._wait_time)
        if not estaciones:  # Empty json
            raise Exception("No stations found")
        return estaciones

    def historical_data(self, station_codes, start_date, end_date):
        """
        This function will get the historical data from the given stations between the given dates.
        :param station_codes: List of station codes to get the data from
        :param start_date: Start date of the data to get
        :param end_date: End date of the data to get
        :return: List with the historical data
        :doc-author: baobab soluciones
        """
        api_key = self._api_key
        data = []
        for station in tqdm(station_codes):
            # Make it in a loop to make it easier to debug errors in HTTP requests
            url = (
                "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/"
                + start_date.strftime("%Y-%m-%dT%H:%M:%SUTC")
                + "/fechafin/"
                + end_date.strftime("%Y-%m-%dT%H:%M:%SUTC")
                + "/estacion/"
                + station
                + "?api_key="
                + api_key
            )
            hist_data_url_call = requests.get(url).json()
            time.sleep(self._wait_time)
            if hist_data_url_call["estado"] != 200:
                raise Exception(
                    "Error getting historical data from AEMET API: "
                    + str(hist_data_url_call["estado"])
                )
            hist_data_url = hist_data_url_call.get("datos")
            if not hist_data_url.startswith("https://opendata.aemet.es"):
                raise Exception(
                    "Error in the url returned by AEMET "
                    + str(hist_data_url.get("datos"))
                )
            hist_data = requests.get(hist_data_url).json()
            time.sleep(self._wait_time)
            if not hist_data:  # Empty json
                print("No data found for station: " + station)
            data.append(hist_data)
            time.sleep(0.1)
        if not data:
            raise Exception("Failed for all stations")
        return data

    def live_data(self, station_codes):
        """
        This function will get the live data from the given stations.
        :param station_codes: List of station codes to get the data from
        :return: List with the live data
        :doc-author: baobab soluciones
        """
        api_key = self._api_key
        data = []
        for station in tqdm(station_codes):
            url = (
                "https://opendata.aemet.es/opendata/api/observacion/convencional/datos/estacion/"
                + station
                + "?api_key="
                + api_key
            )
            live_data_url_call = requests.get(url).json()
            time.sleep(self._wait_time)
            if live_data_url_call["estado"] == 404:  # Empty json
                print("No data found for station: " + station)
                continue
            if live_data_url_call["estado"] != 200:
                raise Exception(
                    "Error getting historical data from AEMET API: "
                    + str(live_data_url_call["descripcion"])
                )
            live_data_url = live_data_url_call.get("datos")
            if not live_data_url.startswith("https://opendata.aemet.es"):
                raise Exception(
                    "Error in the url returned by AEMET "
                    + str(live_data_url.get("datos"))
                )
            live_data = requests.get(live_data_url).json()
            time.sleep(self._wait_time)
            data.append(live_data)
        if not data:
            raise Exception("Failed for all stations")
        return data


if __name__ == "__main__":
    c = Aemet(
        api_key="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbnRvbmlvLmdvbnphbGV6QGJhb2JhYnNvbHVjaW9uZXMuZXMiLCJqdGkiOiIzMWRk"
        "NzJiNS1hYmFmLTQ2OWQtOTViZi1lNTkxN2U2OTcyNWIiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTY3ODcyMDY3MCwidXNlcklkI"
        "joiMzFkZDcyYjUtYWJhZi00NjlkLTk1YmYtZTU5MTdlNjk3MjViIiwicm9sZSI6IiJ9.0rMlgaNipG5T4rlfvyzmYMg6jGfWN"
        "uLKCb0pnGfYzSw"
    )
    c.get_meteo_data()
