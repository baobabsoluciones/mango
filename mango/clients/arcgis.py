import logging
import json
import os
import time
from datetime import datetime

import requests
from mango.processing import load_json
from mango.shared import InvalidCredentials, ARCGIS_TOKEN_URL, validate_args, JobError
from mango.shared.const import (
    ARCGIS_GEOCODE_URL,
    ARCIS_ODMATRIX_JOB_URL,
    ARCGIS_CAR_TRAVEL_MODE,
)

this_dir, file = os.path.split(__file__)
schema = load_json(f"{this_dir}/../schemas/location.json")


class ArcGisClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        logging.getLogger("root")

    def connect(self):
        """
        The connect function is used to authenticate the user and get a token.
        It takes in two parameters, client_id and client_secret, which are both strings.
        The function then makes a POST request to the ArcGIS API for Python login URL
        with these parameters as form data.
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
                f"There was an error on login into ArcGis: {response.json()}. Exception: {e}"
            )

    def get_geolocation(self, address: str) -> tuple:
        """
        Get the geolocation of an address.

        :param str address: the address to get the geolocation
        :return: a tuple with the longitude and the latitude for the geolocation
        :rtype: tuple
        :doc-author: baobab soluciones
        """
        url = f"{ARCGIS_GEOCODE_URL}?f=json&token={self.token}&singleLine={address}&sourceCountry=ESP&forStorage=false"
        response = requests.get(url=url)

        try:
            location = response.json()["candidates"][0]["location"]
        except KeyError as e:
            raise JobError(
                f"There was an error on login into ArcGis: {response.json()}. Exception: {e}"
            )
        except IndexError as e:
            logging.warning(f"There was no candidates for address {address}")
            return None, None
        return location["x"], location["y"]

    @validate_args(
        origins=schema,
        destinations=schema,
    )
    def get_origin_destination_matrix(
        self, *, origins: list, destinations: list, travel_mode: dict = None
    ) -> list:
        """
        Get the origin destination matrix for a list of origins and destinations.

        :param list origins: a list of dictionaries with the origin information. Keys needed: "name", "x", "y".
        :param list destinations: a list of dictionaries with the destination information.
        Keys needed: "Name", "x", "y".
        :param dict travel_mode:
        :return: list of dictionaries with the origin destination matrix with distance in meters and time in seconds
        :rtype: list
        :doc-author: baobab soluciones
        """
        if travel_mode is None:
            travel_mode = ARCGIS_CAR_TRAVEL_MODE

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
        try:
            job_id = response.json()["jobId"]
        except KeyError as e:
            raise JobError(
                f"The job was not submitted correctly and "
                f"it did not give back an id: {response.json()}. Exception: {e}"
            )
        timeout = 600
        start = datetime.utcnow()
        while (datetime.utcnow() - start).seconds < timeout:
            response = requests.get(
                url=f"{ARCIS_ODMATRIX_JOB_URL}/jobs/{job_id}?f=json&returnMessages=True&token={self.token}"
            )
            status = response.json()["jobStatus"]
            if status == "esriJobSucceeded":
                break
            elif status == "esriJobFailed":
                raise JobError(f"ArcGis job has a failed status: {response.json()}")
            elif status == "esriJobCancelled":
                raise JobError(f"ArcGis job was cancelled: {response.json()}")
            elif status == "esriJobTimedOut":
                raise JobError(f"ArcGis job timed out: {response.json()}")
            elif status == "esriJobCancelling":
                raise JobError(
                    f"ArcGis job is in the process of getting cancelled: {response.json()}"
                )
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
