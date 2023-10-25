import time
import warnings
from abc import ABC, abstractmethod
from typing import Union, Literal, Type, List

import requests
from pydantic import BaseModel


class RestClient(ABC):
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
    def _request_handler(
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
        if if_error not in ["raise", "warn", "ignore"]:
            if_error = "raise"
            warnings.warn(
                f"Invalid option for if_error. Valid options are: raise, warn, ignore. "
                f"Setting if_error to {if_error}"
            )
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
