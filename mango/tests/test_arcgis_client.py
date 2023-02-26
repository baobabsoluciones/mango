from unittest import TestCase, mock

from mango.clients.arcgis import ArcGisClient
from mango.shared.exceptions import InvalidCredentials


class TestArcGisClient(TestCase):
    def setUp(self):
        self.client_id = ""
        self.client_secret = ""

    def tearDown(self):
        pass

    def test_init_class(self):
        client = ArcGisClient(self.client_id, self.client_secret)
        self.assertIsInstance(client, ArcGisClient)

    @mock.patch("mango.clients.arcgis.requests")
    def test_connect(self, mock_requests):
        client = ArcGisClient(self.client_id, self.client_secret)
        client.connect()
        mock_requests.post.assert_called_once()
        client.token = "SOMETOKEN"
        return client

    @mock.patch("mango.clients.arcgis.requests")
    def test_bad_response_on_connect(self, mock_requests):
        mock_requests.post.return_value.json.return_value = {"error": "invalid_grant"}
        client = ArcGisClient(self.client_id, self.client_secret)
        self.assertRaises(InvalidCredentials, client.connect)
        mock_requests.post.assert_called_once()

    @mock.patch("mango.clients.arcgis.requests")
    def test_error_on_connect(self, mock_requests):
        mock_requests.post.side_effect = Exception("Error")
        client = ArcGisClient(self.client_id, self.client_secret)
        self.assertRaises(Exception, client.connect)
        mock_requests.post.assert_called_once()

    @mock.patch("mango.clients.arcgis.requests")
    def test_get_geo_location(self, mock_requests):
        mock_requests.get.return_value.json.return_value = {
            "candidates": [{"location": {"x": 1, "y": 2}}]
        }
        client = self.test_connect()
        result = client.get_geolocation("Some address somewhere")
        self.assertEqual(result, (1, 2))
        mock_requests.get.asser_called_once()

    @mock.patch("mango.clients.arcgis.requests")
    def test_geo_location_no_candidates(self, mock_requests):
        mock_requests.get.return_value.json.return_value = {"candidates": []}
        client = self.test_connect()
        result = client.get_geolocation("Some address somewhere")
        self.assertEqual(result, (None, None))
        mock_requests.get.asser_called_once()

    @mock.patch("mango.clients.arcgis.requests")
    def test_geo_location_no_candidates_key(self, mock_requests):
        mock_requests.get.return_value.json.return_value = {}
        client = self.test_connect()
        self.assertRaises(
            InvalidCredentials, client.get_geolocation, "Some address somewhere"
        )
        mock_requests.get.asser_called_once()

    @mock.patch("mango.clients.arcgis.requests")
    def test_geo_location_no_location_key(self, mock_requests):
        mock_requests.get.return_value.json.return_value = {
            "candidates": [{"other": 0}]
        }
        client = self.test_connect()
        self.assertRaises(
            InvalidCredentials, client.get_geolocation, "Some address somewhere"
        )
        mock_requests.get.asser_called_once()

    @mock.patch("mango.clients.arcgis.requests")
    def test_get_distances(self, mock_requests):
        origins = [{"x": 1, "y": 1, "name": "First location"}]
        destinations = [{"x": 2, "y": 2, "name": "Second location"}]
        client = self.test_connect()

        mock_requests.get.return_value.json.side_effect = [
            {"jobId": 100},
            {"jobStatus": "esriJobSucceeded"},
            {},
        ]

        response = client.get_origin_destination_matrix(
            origins=origins, destinations=destinations
        )
