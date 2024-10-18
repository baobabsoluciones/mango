import time
import warnings
from abc import ABC, abstractmethod
from typing import Union, Literal, Type, List

import requests
from pydantic import BaseModel
from requests import HTTPError


class RESTClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def connect(self, *args, **kwargs):
        """
        This method must be implemented in the child class and must connect to the API
        Can be used to check if the API is available and API keys are valid
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
        This function will handle the request to a URL, implements the wait time and checks for errors.
        :param url: URL to make the request to
        :param params: Parameters to pass to the request
        :param wait_time: Wait time in seconds. Default: 0.5 seconds
        :param if_error: What to do if an error is found. Options: raise, warn, ignore
        :param expected_schema: Pydantic schema to validate the response or generate a default response
        :return: Dictionary with the response
        :doc-author: baobab soluciones
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
        Decorator for functions that return a response from the server using requests library. It will check the status
        code of the response and raise an exception if the status of the response is not the expected
        and raise an exception if the status of the response is not the expected
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
