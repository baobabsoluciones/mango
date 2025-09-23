import time
import warnings
from abc import ABC, abstractmethod
from typing import Union, Literal, Type, List

import requests
from pydantic import BaseModel
from requests import HTTPError


class RESTClient(ABC):
    """
    Abstract base class for REST API clients.

    This class provides a foundation for implementing REST API clients with
    common functionality for request handling, error management, and response
    validation. Concrete implementations must provide their own connect method.

    Example:
        >>> class MyAPIClient(RESTClient):
        ...     def connect(self):
        ...         # Implementation for API connection
        ...         pass
    """

    def __init__(self):
        pass

    @abstractmethod
    def connect(self, *args, **kwargs):
        """
        Establish connection to the REST API.

        This abstract method must be implemented by subclasses to establish
        a connection to their specific API. It can be used to verify API
        availability and validate authentication credentials.

        :param args: Variable length argument list
        :param kwargs: Arbitrary keyword arguments
        :raises NotImplementedError: If not implemented in subclass

        Example:
            >>> client = MyAPIClient()
            >>> client.connect(api_key="your_key")
        """
        raise NotImplementedError("This method must be implemented in the child class")

    @staticmethod
    def request_handler(
        url: str,
        params: dict,
        wait_time: Union[float, int] = 0.5,
        if_error: Literal["raise", "warn", "ignore"] = "raise",
        expected_schema: Type[BaseModel] = None,
    ) -> Union[dict, List[dict]]:
        """
        Handle HTTP GET requests with error management and response validation.

        Makes a GET request to the specified URL with the given parameters,
        implements rate limiting through wait time, and provides flexible
        error handling options. Optionally validates response against a
        Pydantic schema.

        :param url: URL to make the request to
        :type url: str
        :param params: Parameters to pass to the request
        :type params: dict
        :param wait_time: Wait time in seconds after the request
        :type wait_time: Union[float, int]
        :param if_error: Error handling strategy - "raise", "warn", or "ignore"
        :type if_error: Literal["raise", "warn", "ignore"]
        :param expected_schema: Pydantic schema to validate response structure
        :type expected_schema: Type[BaseModel], optional
        :return: Parsed JSON response as dictionary or list of dictionaries
        :rtype: Union[dict, List[dict]]
        :raises HTTPError: If request fails and if_error is "raise"
        :raises ValueError: If if_error parameter is invalid

        Example:
            >>> response = RESTClient.request_handler(
            ...     url="https://api.example.com/data",
            ...     params={"key": "value"},
            ...     wait_time=1.0,
            ...     if_error="warn"
            ... )
        """
        response = requests.get(url, params=params)
        try:
            response.raise_for_status()
        except Exception as e:
            if if_error == "raise":
                raise e
            elif if_error == "warn":
                warnings.warn(str(e))
            elif if_error == "ignore":
                pass
            else:
                raise ValueError(f"Invalid if_error: {if_error}")
        time.sleep(wait_time)
        try:
            parsed = response.json()
        except Exception as e:
            if if_error == "raise":
                raise e
            elif if_error == "warn":
                warnings.warn(str(e))
                parsed = {}
            elif if_error == "ignore":
                parsed = {}
            else:
                raise ValueError(f"Invalid if_error: {if_error}")
        if expected_schema:
            if parsed:
                try:
                    if isinstance(parsed, dict):
                        parsed = expected_schema(**parsed).model_dump()
                    elif isinstance(parsed, list):
                        parsed = expected_schema(parsed).model_dump()
                except Exception as e:
                    if if_error == "raise":
                        raise e
                    elif if_error == "warn":
                        warnings.warn(str(e))
                        parsed = {key: None for key in expected_schema.model_fields}
                    elif if_error == "ignore":
                        parsed = {key: None for key in expected_schema.model_fields}
            else:
                # Return Dict with keys and None values
                parsed = {key: None for key in expected_schema.model_fields}
        return parsed

    @staticmethod
    def expect_status(
        func, expected_status=None, response_type: Literal["json", "raw"] = "json"
    ):
        """
        Decorator to validate HTTP response status codes and format responses.

        This decorator wraps functions that return HTTP responses and validates
        the status code against the expected value. It also handles response
        formatting to return either JSON data or raw response objects.

        :param func: Function that returns a requests.Response object
        :type func: callable
        :param expected_status: Expected HTTP status code (e.g., 200, 201)
        :type expected_status: int, optional
        :param response_type: Format of returned data - "json" or "raw"
        :type response_type: Literal["json", "raw"]
        :return: Decorated function that validates status and formats response
        :rtype: callable
        :raises HTTPError: If response status doesn't match expected_status
        :raises ValueError: If response_type is invalid

        Example:
            >>> @RESTClient.expect_status(expected_status=200, response_type="json")
            ... def get_data():
            ...     return requests.get("https://api.example.com/data")
        """

        def decorator(*args, **kwargs):
            response = func(*args, **kwargs)
            if expected_status is not None and response.status_code != expected_status:
                raise HTTPError(
                    f"Expected a code {expected_status}, got a {response.status_code} error instead: {response.text}"
                )
            if response_type == "json":
                return response.json()
            elif response_type == "raw":
                return response
            else:
                raise ValueError(f"Invalid response_type: {response_type}")

        return decorator
