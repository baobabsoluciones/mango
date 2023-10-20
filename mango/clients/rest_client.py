import warnings
from abc import ABC, abstractmethod
import time
from typing import Union, Literal, Type

import requests
from pydantic import BaseModel


class RestClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    @staticmethod
    def _request_handler(
        url: str,
        params: dict,
        wait_time: Union[float, int] = 0.5,
        if_error: Literal["raise", "warn", "ignore"] = "raise",
        expected_schema: Type[BaseModel] = None,
    ) -> Union[dict, list[dict]]:
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
                # If not valid option keep default behaviour
                raise e
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
                # If not valid option keep default behaviour
                raise e
        if expected_schema:
            if parsed:
                if isinstance(parsed, dict):
                    parsed = expected_schema(**parsed).model_dump()
                elif isinstance(parsed, list):
                    parsed = expected_schema(parsed).model_dump()
            else:
                # Return Dict with keys and None values
                parsed = {key: None for key in expected_schema.model_fields}
        return parsed
