import json
import os
import time
import warnings
from datetime import datetime, timezone

import requests
from mango.logging import get_configured_logger
from mango.processing import load_json
from mango.shared import InvalidCredentials, ARCGIS_TOKEN_URL, validate_args, JobError
from mango.shared.const import (
    ARCGIS_GEOCODE_URL,
    ARCIS_ODMATRIX_JOB_URL,
    ARCGIS_CAR_TRAVEL_MODE,
    ARCGIS_ODMATRIX_DIRECT_URL,
)

log = get_configured_logger(__name__)

this_dir, file = os.path.split(__file__)
schema = load_json(f"{this_dir}/../schemas/location.json")


class ArcGisClient:
    """
    Client for accessing ArcGIS services including geocoding and routing.

    This class provides access to ArcGIS REST API services for geocoding addresses
    and calculating origin-destination matrices. Authentication is required using
    client credentials.

    :param client_id: ArcGIS client ID for authentication
    :type client_id: str
    :param client_secret: ArcGIS client secret for authentication
    :type client_secret: str

    Example:
        >>> client = ArcGisClient(client_id="your_client_id", client_secret="your_secret")
        >>> client.connect()
        >>> coords = client.get_geolocation("Madrid, Spain")
    """

    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None

    def connect(self):
        """
        Authenticate with ArcGIS services and obtain access token.

        Establishes connection to ArcGIS REST API using client credentials.
        The obtained access token is stored for use in subsequent API calls.
        This method must be called before using other methods in the class.

        :raises InvalidCredentials: If authentication fails or credentials are invalid

        Example:
            >>> client = ArcGisClient(client_id="your_id", client_secret="your_secret")
            >>> client.connect()
        """
        body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        try:
            response = requests.post(url=ARCGIS_TOKEN_URL, data=body)
        except Exception as e:
            raise InvalidCredentials(f"There was an error on login into ArcGis: {e}")

        try:
            self.token = response.json()["access_token"]
        except KeyError as e:
            raise InvalidCredentials(
                f"There was an error on login into ArcGis: {response.json()}. Exception: {e}"
            )

    def get_geolocation(self, address: str, country: str = "ESP") -> tuple:
        """
        Get the geolocation coordinates for a given address.

        Uses ArcGIS geocoding service to convert an address into longitude
        and latitude coordinates. The service returns the best match for
        the provided address.

        :param address: The address to geocode
        :type address: str
        :param country: Country code for the address (default: "ESP" for Spain)
        :type country: str
        :return: Tuple containing (longitude, latitude) coordinates
        :rtype: tuple
        :raises Exception: If API request fails or returns invalid status
        :raises JobError: If geocoding fails or no candidates are found

        Example:
            >>> coords = client.get_geolocation("Madrid, Spain")
            >>> print(f"Longitude: {coords[0]}, Latitude: {coords[1]}")
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
            log.warning(f"There was no candidates for address {address}")
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
        Calculate origin-destination matrix with travel times and distances.

        Computes travel times and distances between multiple origins and destinations
        using ArcGIS routing services. Supports both synchronous and asynchronous modes.
        The method automatically switches to async mode for large matrices.

        :param mode: Processing mode - "sync" for immediate results, "async" for large datasets
        :type mode: str
        :param origins: List of origin points with keys: "name", "x", "y"
        :type origins: list
        :param destinations: List of destination points with keys: "name", "x", "y"
        :type destinations: list
        :param travel_mode: Travel mode configuration (default: car travel mode)
        :type travel_mode: dict, optional
        :param sleep_time: Wait time between status checks in async mode (seconds)
        :type sleep_time: int
        :return: List of dictionaries with origin, destination, distance (meters), and time (seconds)
        :rtype: list
        :raises NotImplementedError: If matrix size exceeds 1000x1000 limit
        :raises ValueError: If mode is invalid

        Example:
            >>> origins = [{"name": "Madrid", "x": -3.7038, "y": 40.4168}]
            >>> destinations = [{"name": "Barcelona", "x": 2.1734, "y": 41.3851}]
            >>> matrix = client.get_origin_destination_matrix(
            ...     origins=origins, destinations=destinations, mode="sync"
            ... )
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
        Calculate origin-destination matrix using asynchronous processing.

        Processes large matrices asynchronously by submitting a job and polling
        for completion. This method is used for matrices that are too large
        for synchronous processing (>10x10 origins/destinations).

        :param origins: Dictionary with features and attributes of origin points
        :type origins: dict
        :param destinations: Dictionary with features and attributes of destination points
        :type destinations: dict
        :param travel_mode: Travel mode configuration for routing calculations
        :type travel_mode: dict, optional
        :param sleep_time: Wait time between status checks in seconds
        :type sleep_time: int
        :return: List of dictionaries with origin, destination, distance (meters), and time (seconds)
        :rtype: list
        :raises Exception: If API request fails or returns invalid status
        :raises JobError: If job submission, execution, or retrieval fails

        Note:
            For more information about the API, see:
            https://developers.arcgis.com/rest/network/api-reference/origin-destination-cost-matrix-service.htm
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
        self, *, origins: dict, destinations: dict, travel_mode: dict = None
    ):
        """
        Calculate origin-destination matrix using synchronous processing.

        Processes small to medium matrices synchronously for immediate results.
        This method is suitable for matrices up to 10x10 origins/destinations.
        Results are returned immediately without job polling.

        :param origins: Dictionary with features and attributes of origin points
        :type origins: dict
        :param destinations: Dictionary with features and attributes of destination points
        :type destinations: dict
        :param travel_mode: Travel mode configuration for routing calculations
        :type travel_mode: dict, optional
        :return: List of dictionaries with origin, destination, distance (meters), and time (seconds)
        :rtype: list
        :raises Exception: If API request fails or calculation errors occur

        Note:
            For more information about the API, see:
            https://developers.arcgis.com/rest/network/api-reference/origin-destination-cost-matrix-synchronous-service.htm
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
