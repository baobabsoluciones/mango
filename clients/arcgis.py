import requests
from shared import InvalidCredentials, ARCGIS_TOKEN_URL


class ArcGisClient:
    def __init__(self, base_api_url, client_id, client_secret):
        self.base_api_url = base_api_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None

    def connect(self):
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

    def get_origin_destination_matrix(self, origins, destinations):
        # TODO: based on the size of the matrix attack the direct API or the job API.
        pass
