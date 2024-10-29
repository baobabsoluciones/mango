# Set wait time to 0 to avoid waiting time between requests as it is not needed for testing
from unittest import TestCase, mock

from mango.clients.rest_client import RESTClient
from requests import HTTPError


class TestRest(TestCase):
    """
    Test Rest Client
    """

    def setUp(self) -> None:
        pass

    def test_child_creation(self):
        class TestRESTClient(RESTClient):
            def __init__(self):
                super().__init__()

            # This is needed to avoid Abstract class error
            def connect(self, *args, **kwargs):
                super().connect(*args, **kwargs)

        test = TestRESTClient()
        self.assertIsInstance(test, TestRESTClient)
        self.assertIsInstance(test, RESTClient)
        self.assertRaises(NotImplementedError, test.connect)

        class TestRESTClient2(RESTClient):
            def __init__(self):
                super().__init__()

            def connect(self):
                pass

        test2 = TestRESTClient2()
        self.assertIsInstance(test2, TestRESTClient2)
        self.assertIsInstance(test2, RESTClient)
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
            RESTClient  # Is an abstract class however, _request_handler is static
        )
        wait_time = 0
        # Test with error raise
        with self.assertRaises(Exception):
            rest_client.request_handler(
                url="url", params={}, if_error="raise", wait_time=wait_time
            )
        # Test with error warn
        # Commented due to conflict with pyomo
        # with self.assertWarns(Warning):
        #     rest_client.request_handler(
        #         url="url", params={}, if_error="warn", wait_time=wait_time
        #     )
        # Test with error ignore
        rest_client.request_handler(
            url="url", params={}, if_error="ignore", wait_time=wait_time
        )
        # Test with invalid if_error
        with self.assertRaises(Exception):
            rest_client.request_handler(
                url="url", params={}, if_error="invalid", wait_time=wait_time
            )
        # Now with invalid JSON for parsing
        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = Exception("Test error")
        # Test with error raise
        with self.assertRaises(Exception):
            rest_client.request_handler(
                url="url", params={}, if_error="raise", wait_time=wait_time
            )
        # Test with error warn
        # Commented due to conflict with pyomo
        # with self.assertWarns(Warning):
        #     rest_client.request_handler(
        #         url="url", params={}, if_error="warn", wait_time=wait_time
        #     )
        # Test with error ignore
        rest_client.request_handler(
            url="url", params={}, if_error="ignore", wait_time=wait_time
        )
        # Test with invalid if_error
        with self.assertRaises(Exception):
            rest_client.request_handler(
                url="url", params={}, if_error="invalid", wait_time=wait_time
            )
        # Test using ExpectedSchema but empty json
        from mango.validators.aemet import FetchHistoricElement

        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = None
        mock_requests.get.return_value.json.return_value = {"invalid": "invalid"}
        with self.assertRaises(Exception):
            result = rest_client.request_handler(
                url="url",
                params={},
                if_error="raise",
                expected_schema=FetchHistoricElement,
                wait_time=wait_time,
            )
        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = None
        mock_requests.get.return_value.json.return_value = {"invalid": "invalid"}
        # Commented due to conflict with pyomo
        # with self.assertWarns(Warning):
        #     result = rest_client.request_handler(
        #         url="url",
        #         params={},
        #         if_error="warn",
        #         expected_schema=FetchHistoricElement,
        #         wait_time=wait_time,
        #     )
        #     self.assertEqual(
        #         result, {key: None for key in FetchHistoricElement.model_fields}
        #     )
        mock_requests.get.return_value.raise_for_status.side_effect = None
        mock_requests.get.return_value.json.side_effect = None
        mock_requests.get.return_value.json.return_value = {"invalid": "invalid"}
        result = rest_client.request_handler(
            url="url",
            params={},
            if_error="ignore",
            expected_schema=FetchHistoricElement,
            wait_time=wait_time,
        )
        self.assertEqual(
            result, {key: None for key in FetchHistoricElement.model_fields}
        )

    @mock.patch("requests.get")
    def test_decorator(self, mock_requests):
        import requests

        class Child(RESTClient):
            def __init__(self):
                super().__init__()
                self.raw_petition = self.expect_status(
                    self.petition, 200, response_type="raw"
                )
                self.parsed_petition = self.expect_status(
                    self.petition, 200, response_type="json"
                )
                self.invalid_petition = self.expect_status(
                    self.petition, 200, response_type="invalid"
                )

            def connect(self):
                pass

            def petition(self, url, params):
                return requests.get(url, params)

        # Set up 4 requests, with 200, 400, 200, 400 status codes
        mock_requests.side_effect = [
            mock.MagicMock(status_code=200),
            mock.MagicMock(status_code=400),
            mock.MagicMock(status_code=200, json=lambda: {"test": "test"}),
            mock.MagicMock(status_code=400),
            mock.MagicMock(status_code=200),
        ]
        child = Child()
        # Check raw petition works
        r = child.raw_petition("url", "params")
        self.assertEqual(r.status_code, 200)
        with self.assertRaises(HTTPError):
            child.raw_petition("url", "params")
        # Check parsed petition works
        r = child.parsed_petition("url", "params")
        self.assertEqual(r, {"test": "test"})
        with self.assertRaises(HTTPError):
            child.parsed_petition("url", "params")
        with self.assertRaises(ValueError):
            child.invalid_petition("url", "params")
