from datetime import datetime

import requests
from geopy.distance import geodesic as geo_distance


class Aemet:
    """
    This class will implement the AEMET API to make it easier to use meteorological data as input for
    machine learning models.
    """

    def __init__(self, *args, **kwargs):
        # TODO: Add checks for api_key. How to check if it is valid?
        self._api_key: str = kwargs.get("api_key")

    def get_meteo_data(self, *args, **kwargs):
        """
        This function will get the meteorological data from the AEMET API. Possible parameters are:

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

        :param args:
        :param kwargs:
        :doc-author: baobab soluciones
        """
        # TODO: Add checks for parameters
        # TODO: Improve error handling to avoid duplicate code
        lat: float = kwargs.get("lat")
        long: float = kwargs.get("long")
        province: str = kwargs.get("province")
        start_date: datetime = kwargs.get("start_date")
        end_date: datetime = kwargs.get("end_date")

        # Check datatype
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
        if lat and long:
            # Search for closest station
            # province None is handled in search_closest_station
            # Only one but to mantain consistency embed in list
            station_codes = self.search_closest_station(lat, long, province)
        elif province:
            # Search for all stations in province
            station_codes = self.search_stations_province(province)
        else:
            # Search for all stations in Spain
            station_codes = self.get_all_stations()

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
        # TODO: Cache stations to avoid calling API every time
        try:
            # If province is None will raise exception also if province is not valid
            all_stations = self.search_stations_province(province)
        except Exception as e:
            print(e)
            print("Searching for closest station in all Spain")
            all_stations = self.get_all_stations()
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
        all_stations = self.get_all_stations()
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
        if estaciones_url_call["estado"] != 200:
            raise Exception(
                "Error getting stations from AEMET API: "
                + str(estaciones_url_call["estado"])
            )
        estaciones_url = estaciones_url_call.get("datos")
        if not estaciones_url.startswith("https://opendata.aemet.es"):
            raise Exception("Error in the url returned by AEMET " + estaciones_url)
        estaciones = requests.get(estaciones_url).json()
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
        for station in station_codes:
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
            if not hist_data:  # Empty json
                print("No data found for station: " + station)
            data.append(hist_data)
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
        for station in station_codes:
            url = (
                "https://opendata.aemet.es/opendata/api/observacion/convencional/datos/estacion/"
                + station
                + "?api_key="
                + api_key
            )
            live_data_url_call = requests.get(url).json()
            if live_data_url_call["estado"] != 200:
                raise Exception(
                    "Error getting historical data from AEMET API: "
                    + str(live_data_url_call["estado"])
                )
            live_data_url = live_data_url_call.get("datos")
            if not live_data_url.startswith("https://opendata.aemet.es"):
                raise Exception(
                    "Error in the url returned by AEMET "
                    + str(live_data_url.get("datos"))
                )
            live_data = requests.get(live_data_url).json()
            if not live_data:  # Empty json
                print("No data found for station: " + station)
            data.append(live_data)
            if not data:
                raise Exception("Failed for all stations")
            return data
