# Set wait time to 0 to avoid waiting time between requests as it is not needed for testing
from unittest import TestCase, mock

from mango.clients.rest_client import RestClient


class TestRest(TestCase):
    """
    Test Rest Client
    """

    def setUp(self) -> None:
        pass

    def test_child_creation(self):
        class TestRestClient(RestClient):
            def __init__(self):
                super().__init__()

            # This is needed to avoid Abstract class error
            def connect(self, *args, **kwargs):
                super().connect(*args, **kwargs)

        test = TestRestClient()
        self.assertIsInstance(test, TestRestClient)
        self.assertIsInstance(test, RestClient)
        self.assertRaises(NotImplementedError, test.connect)

        class TestRestClient2(RestClient):
            def __init__(self):
                super().__init__()

            def connect(self):
                pass

        test2 = TestRestClient2()
        self.assertIsInstance(test2, TestRestClient2)
        self.assertIsInstance(test2, RestClient)
        self.assertIsNone(test2.connect())

    @mock.patch("mango.clients.rest_client.requests")
    def test_request_handler(self, mock_requests):
        """
        Test request handler
        """
        mock_requests.get.return_value = mock.MagicMock()
        mock_requests.get.return_value.json.return_value = {"test": "test"}
        mock_requests.get.return_value.raise_for_status.side_effect = Exception(
            "Test error"
        )
        # Setup
        rest_client = (
            RestClient  # Is an abstract class however, _request_handler is static
        )
        wait_time = 0
        # Test with error raise
        with self.assertRaises(Exception):
            rest_client._request_handler(
                url="url", params={}, if_error="raise", wait_time=wait_time
            )
        # Test with error warn
        with self.assertWarns(Warning):
            rest_client._request_handler(
                url="url", params={}, if_error="warn", wait_time=wait_time
            )
        # Test with error ignore
        rest_client._request_handler(
            url="url", params={}, if_error="ignore", wait_time=wait_time
        )
        # Test with invalid if_error
        with self.assertWarns(Warning) and self.assertRaises(Exception):
            rest_client._request_handler(
                url="url", params={}, if_error="invalid", wait_time=wait_time
            )
        # Now with invalid JSON for parsing
        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = Exception("Test error")
        # Test with error raise
        with self.assertRaises(Exception):
            rest_client._request_handler(
                url="url", params={}, if_error="raise", wait_time=wait_time
            )
        # Test with error warn
        with self.assertWarns(Warning):
            rest_client._request_handler(
                url="url", params={}, if_error="warn", wait_time=wait_time
            )
        # Test with error ignore
        rest_client._request_handler(
            url="url", params={}, if_error="ignore", wait_time=wait_time
        )
        # Test using ExpectedSchema but empty json
        from mango.validators.aemet import FetchHistoricElement

        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = None
        mock_requests.get.return_value.json.return_value = {"invalid": "invalid"}
        with self.assertRaises(Exception):
            result = rest_client._request_handler(
                url="url",
                params={},
                if_error="raise",
                expected_schema=FetchHistoricElement,
                wait_time=wait_time,
            )
        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = None
        mock_requests.get.return_value.json.return_value = {"invalid": "invalid"}
        with self.assertWarns(Warning):
            result = rest_client._request_handler(
                url="url",
                params={},
                if_error="warn",
                expected_schema=FetchHistoricElement,
                wait_time=wait_time,
            )
            self.assertEqual(
                result, {key: None for key in FetchHistoricElement.model_fields}
            )
        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = None
        mock_requests.get.return_value.json.return_value = {"invalid": "invalid"}
        result = rest_client._request_handler(
            url="url",
            params={},
            if_error="ignore",
            expected_schema=FetchHistoricElement,
            wait_time=wait_time,
        )
        self.assertEqual(
            result, {key: None for key in FetchHistoricElement.model_fields}
        )
