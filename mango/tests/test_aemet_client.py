from unittest import TestCase, mock

from mango.clients.aemet import AemetClient
from mango.shared import ApiKeyError


class TestAemet(TestCase):
    @mock.patch("mango.clients.aemet.requests")
    def test_init_class(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            {
                "datos": "https://opendata.aemet.es/data",
                "estado": 200,
            },
            [{"latitud": "412342N", "longitud": "123123W", "indicativo": "3194U"}],
        ]

        client = AemetClient(api_key="1234567890")
        self.assertIsInstance(client, AemetClient)
        self.assertEqual(
            client.all_stations,
            [{"latitud": "412342N", "longitud": "123123W", "indicativo": "3194U"}],
        )

    @mock.patch("mango.clients.aemet.requests")
    def test_init_class_invalid_api_key(self, mock_requests):
        mock_requests.get.return_value.raise_for_status.side_effect = ApiKeyError(
            "Error"
        )
        mock_requests.get.return_value.status_code = 401
        mock_requests.get.return_value.text = "Api key is invalid"
        with self.assertRaises(ApiKeyError):
            AemetClient(api_key="")

    @mock.patch("mango.clients.aemet.requests")
    def test_get_filtered_stations(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            {
                "datos": "https://opendata.aemet.es/data",
                "estado": 200,
            },
            [
                {
                    "latitud": "412342N",
                    "longitud": "123123W",
                    "indicativo": "3194U",
                    "provincia": "NAVARRA",
                },
                {
                    "latitud": "672342N",
                    "longitud": "123123W",
                    "indicativo": "3195",
                    "provincia": "MADRID",
                },
            ],
        ]
        client = AemetClient(api_key="1234567890")
        result = client._get_stations_filtered()
        self.assertEqual(result, [st["indicativo"] for st in client.all_stations])

    @mock.patch("mango.clients.aemet.requests")
    def test_get_filtered_stations_on_province(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            {
                "datos": "https://opendata.aemet.es/data",
                "estado": 200,
            },
            [
                {
                    "latitud": "412342N",
                    "longitud": "123123W",
                    "indicativo": "3194U",
                    "provincia": "NAVARRA",
                },
                {
                    "latitud": "672342N",
                    "longitud": "123123W",
                    "indicativo": "3195",
                    "provincia": "MADRID",
                },
            ],
        ]
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.text = "OK"
        client = AemetClient(api_key="1234567890")
        result = client._get_stations_filtered(province="NAVARRA")
        self.assertEqual(result, ["3194U"])
