import json
import time

import requests
from artic.shared import InvalidCredentials, ARCGIS_TOKEN_URL, validate_args
from artic.shared.const import (
    ARCGIS_GEOCODE_URL,
    ARCIS_ODMATRIX_JOB_URL,
)


class ArcGisClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None

    def connect(self):
        """
        The connect function is used to authenticate the user and get a token.
        It takes in two parameters, client_id and client_secret, which are both strings.
        The function then makes a POST request to the ArcGIS API for Python login URL with these parameters as form data.
        If successful, it stores an access token that can be used for subsequent requests.

        :return: nothing
        :doc-author: baobab soluciones
        """
        body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        try:
            response = requests.post(url=ARCGIS_TOKEN_URL, data=body)
        except Exception as e:
            # TODO: narrow down the possible exception errors
            raise InvalidCredentials(f"There was an error on login into ArcGis: {e}")

        try:
            self.token = response.json()["access_token"]
        except KeyError as e:
            raise InvalidCredentials(
                f"There was an error on login into ArcGis: {response.json()}"
            )

    def get_geo_location(self, address: str) -> tuple:
        """
        Get the geo location of an address.

        :param str address: the address to get the geo location
        :return: a tuple with the longitude and the latitude for the geo location
        :rtype: tuple
        :doc-author: baobab soluciones
        """
        url = f"{ARCGIS_GEOCODE_URL}?f=json&token={self.token}&singleLine={address}&sourceCountry=ESP&forStorage=false"
        response = requests.get(url=url)

        try:
            location = response.json()["candidates"][0]["location"]
        except KeyError as e:
            raise InvalidCredentials(
                f"There was an error on login into ArcGis: {response.json()}"
            )
        return location["x"], location["y"]

    @validate_args(
        origins="./schemas/location.json",
        destinations="./schemas/location.json",
    )
    def get_origin_destination_matrix(
        self, *, origins: list, destinations: list, travel_mode: dict
    ) -> list:
        """
        Get the origin destination matrix for a list of origins and destinations.

        :param list origins: a list of dictionaries with the origin information. Keys needed: "Name", "x", "y".
        :param list destinations: a list of dictionaries with the destination information. Keys needed: "Name", "x", "y".
        :param dict travel_mode:
        :return: list of dictionaries with the origin destination matrix with distance in meters and time in seconds
        :rtype: list
        :doc-author: baobab soluciones
        """
        # TODO: based on the size of the matrix attack the direct API or the job API.
        if len(origins) > 1000 or len(destinations) > 1000:
            raise NotImplementedError(
                "The size of the matrix is too big. Maximum number of origins or destinations is 1000."
            )

        origins = {
            "features": [
                {
                    "geometry": {"x": el["x"], "y": el["y"]},
                    "attributes": {"Name": el["name"]},
                }
                for el in origins
            ]
        }
        destinations = {
            "features": [
                {
                    "geometry": {"x": el["x"], "y": el["y"]},
                    "attributes": {"Name": el["name"]},
                }
                for el in destinations
            ]
        }

        url = (
            f"{ARCIS_ODMATRIX_JOB_URL}/submitJob?f=json&token={self.token}&distance_units=Meters&time_units=Seconds"
            f"&origins={json.dumps(origins, separators=(',', ':'))}"
            f"&destinations={json.dumps(destinations,separators=(',', ':'))}"
            f"&travel_mode={json.dumps(travel_mode, separators=(',', ':'))}"
        ).replace("%3A", ":")
        url = requests.Request("GET", url).prepare().url

        response = requests.get(url=url)
        job_id = response.json()["jobId"]
        while True:
            response = requests.get(
                url=f"{ARCIS_ODMATRIX_JOB_URL}/jobs/{job_id}?f=json&returnMessages=True&token={self.token}"
            )
            if response.json()["jobStatus"] == "esriJobSucceeded":
                break
            elif response.json()["jobStatus"] == "esriJobFailed":
                raise Exception(f"Error on ArcGis job: {response.json()}")
            else:
                time.sleep(2)

        matrix = requests.get(
            url=f"{ARCIS_ODMATRIX_JOB_URL}/jobs/{job_id}/results/output_origin_destination_lines?"
            f"token={self.token}&f=json"
        )

        results = [
            {
                "origin": el["attributes"]["OriginName"],
                "destination": el["attributes"]["DestinationName"],
                "distance": el["attributes"]["Total_Distance"],
                "time": el["attributes"]["Total_Time"],
            }
            for el in matrix.json()["value"]["features"]
        ]

        return results
