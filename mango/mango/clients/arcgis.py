import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone

import requests
from mango.processing import load_json
from mango.shared import InvalidCredentials, ARCGIS_TOKEN_URL, validate_args, JobError
from mango.shared.const import (
    ARCGIS_GEOCODE_URL,
    ARCIS_ODMATRIX_JOB_URL,
    ARCGIS_CAR_TRAVEL_MODE,
    ARCGIS_ODMATRIX_DIRECT_URL,
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

    def get_geolocation(self, address: str, country: str = "ESP") -> tuple:
        """
        Get the geolocation of an address.

        :param str address: the address to get the geolocation
        :param str country: the country of the address to get the geolocation (default Spain)
        :return: a tuple with the longitude and the latitude for the geolocation
        :rtype: tuple
        :doc-author: baobab soluciones
        """
        url = f"{ARCGIS_GEOCODE_URL}?f=json&token={self.token}&singleLine={address}&sourceCountry={country}&forStorage=false"
        response = requests.get(url=url)

        if response.status_code != 200:
            raise Exception(f"Wrong status: {response.status_code}")

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
        self,
        *,
        mode: str = "sync",
        origins: list,
        destinations: list,
        travel_mode: dict = None,
        sleep_time: int = 2,
    ) -> list:
        """
        Get the origin destination matrix for a list of origins and destinations.

        :param str mode: the mode to get the matrix. It can be "sync" or "async".
        :param list origins: a list of dictionaries with the origin information. Keys needed: "name", "x", "y".
        :param list destinations: a list of dictionaries with the destination information. Keys needed: "Name", "x", "y".
        :param dict travel_mode: a dictionary with the configuration for the travel mode
        :param int sleep_time: the time to wait for the response in asynchronous mode
        :return: list of dictionaries with the origin destination matrix with distance in meters and time in seconds
        :rtype: list
        :doc-author: baobab soluciones
        """

        if mode is None:
            mode = "sync"
        elif mode != "sync" and mode != "async":
            warnings.warn(f"The selected mode {mode} is not valid. Using sync mode.")
            mode = "sync"

        if travel_mode is None:
            travel_mode = ARCGIS_CAR_TRAVEL_MODE

        if len(origins) > 1000 or len(destinations) > 1000:
            raise NotImplementedError(
                "The size of the matrix is too big. Maximum number of origins or destinations is 1000."
            )

        if (len(origins) > 10 or len(destinations) > 10) and mode == "sync":
            warnings.warn(
                "The size of the matrix is too big for sync mode. Changing to async mode"
            )
            mode = "async"

        origins = {
            "features": [
                {
                    "geometry": {"x": el["x"], "y": el["y"]},
                    "attributes": {
                        "Name": el["name"],
                        "ObjectID": pos + 1001,
                    },
                }
                for pos, el in enumerate(origins)
            ]
        }
        destinations = {
            "features": [
                {
                    "geometry": {"x": el["x"], "y": el["y"]},
                    "attributes": {
                        "Name": el["name"],
                        "ObjectID": pos + 10001,
                    },
                }
                for pos, el in enumerate(destinations)
            ]
        }

        if mode == "sync":
            return self._get_origin_destination_matrix_sync(
                origins=origins, destinations=destinations, travel_mode=travel_mode
            )
        elif mode == "async":
            return self._get_origin_destination_matrix_async(
                origins=origins,
                destinations=destinations,
                travel_mode=travel_mode,
                sleep_time=sleep_time,
            )
        else:
            raise ValueError(f"The mode {mode} is not valid.")

    def _get_origin_destination_matrix_async(
        self,
        *,
        origins: dict,
        destinations: dict,
        travel_mode: dict = None,
        sleep_time: int = 2,
    ):
        """
        Get the origin destination matrix for a list of origins and destinations in async mode
        For more information about the request and the parameters, see:
        https://developers.arcgis.com/rest/network/api-reference/origin-destination-cost-matrix-service.htm

        :param dict origins: a dict with the features and attributes of the origins
        :param dict destinations: a dict with the features and attributes of the destinations
        :param dict travel_mode:
        :param int sleep_time: the time to wait for the response in asynchronous mode
        :return: list of dictionaries with the origin destination matrix with distance in meters and time in seconds
        :rtype: list
        :doc-author: baobab soluciones
        """

        url = (
            f"{ARCIS_ODMATRIX_JOB_URL}/submitJob?f=json&token={self.token}&distance_units=Meters&time_units=Seconds"
            f"&origins={json.dumps(origins, separators=(',', ':'))}"
            f"&destinations={json.dumps(destinations,separators=(',', ':'))}"
            f"&travel_mode={json.dumps(travel_mode, separators=(',', ':'))}"
        ).replace("%3A", ":")
        url = requests.Request("GET", url).prepare().url

        response = requests.get(url=url)

        if response.status_code != 200:
            raise Exception(f"Wrong status: {response.status_code}")

        try:
            job_id = response.json()["jobId"]
        except KeyError as e:
            raise JobError(
                f"The job was not submitted correctly and "
                f"it did not give back an id: {response.json()}. Exception: {e}"
            )
        timeout = 600
        start = datetime.now(timezone.utc)
        while (datetime.now(timezone.utc) - start).seconds < timeout:
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
                time.sleep(sleep_time)

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

    def _get_origin_destination_matrix_sync(
        self,
        *,
        origins: dict,
        destinations: dict,
        travel_mode: dict = None,
        sleep: int = 2,
    ):
        """
        Get the origin destination matrix for a list of origins and destinations in sync mode.
        For more information about the request and the parameters, see:
        https://developers.arcgis.com/rest/network/api-reference/origin-destination-cost-matrix-synchronous-service.htm

        :param dict origins: a dict with the features and attributes of the origins
        :param dict destinations: a dict with the features and attributes of the destinations
        :param dict travel_mode: the travel mode configuration
        :return: list of dictionaries with the origin destination matrix with distance in meters and time in seconds
        :rtype: list
        :doc-author: baobab soluciones
        """
        url = (
            f"{ARCGIS_ODMATRIX_DIRECT_URL}?f=json&token={self.token}"
            f"&origins={json.dumps(origins, separators=(',', ':'))}"
            f"&destinations={json.dumps(destinations, separators=(',', ':'))}"
            f"&travelMode={json.dumps(travel_mode, separators=(',', ':'))}"
            f"&impedanceAttributeName=Minutes"
        ).replace("%3A", ":")

        url = requests.Request("GET", url).prepare().url

        response = requests.get(url=url)
        if response.status_code != 200:
            raise Exception(f"Wrong status: {response.status_code}")

        if "error" in response.json().keys():
            raise Exception(f"Error on calculation: {response.json()['error']}")

        data = response.json()["odCostMatrix"]
        results = []

        kilometers = data["costAttributeNames"].index("Kilometers")
        time = data["costAttributeNames"].index("TravelTime")

        for origin in origins["features"]:
            for destination in destinations["features"]:
                try:
                    origin_id = str(origin["attributes"]["ObjectID"])
                    destination_id = str(destination["attributes"]["ObjectID"])
                    results.append(
                        {
                            "origin": origin["attributes"]["Name"],
                            "destination": destination["attributes"]["Name"],
                            "distance": data[origin_id][destination_id][kilometers]
                            * 1000,
                            "time": data[origin_id][destination_id][time] * 60,
                        }
                    )
                except KeyError:
                    warnings.warn(
                        f"The calculation for {origin['attributes']['Name']} and {destination['attributes']['Name']} "
                        f"was not available"
                    )

        return results
