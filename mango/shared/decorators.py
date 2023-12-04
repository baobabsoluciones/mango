from collections.abc import Mapping
from functools import wraps

from fastjsonschema import compile
from fastjsonschema.exceptions import JsonSchemaValueException
from mango.processing import load_json
from .exceptions import ValidationError
from pydantic_core import ValidationError as ValidationErrorPydantic


def validate_args(**schemas):
    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper evaluator validates the arguments against a schema and then calls
            the original evaluator.

            :param args: Pass a non-keyworded, variable-length argument list to the evaluator
            :param kwargs: Pass the arguments to the evaluator
            :return: The same as the original evaluator
            :doc-author: baobab soluciones
            """

            for key, value in schemas.items():
                if isinstance(value, str):
                    schema = load_json(value)
                elif isinstance(value, dict):
                    schema = value
                else:
                    raise ValueError(f"Schema for {key} is not a dict or a string")
                validate = compile(schema)
                try:
                    validate(kwargs[key])
                except JsonSchemaValueException as e:
                    raise ValidationError(
                        f"Error validating {key}: {e.message}"
                    ) from None
            return func(*args, **kwargs)

        return wrapper

    return decorator


def pydantic_validation(**validators):
    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper evaluator validates the arguments against a schema and then calls
            the original evaluator.

            :param args: Pass a non-keyworded, variable-length argument list to the evaluator
            :param kwargs: Pass the arguments to the evaluator
            :return: The same as the original evaluator
            :doc-author: baobab soluciones
            """
            try:
                for key, value in validators.items():
                    if isinstance(kwargs[key], Mapping):
                        value(**kwargs[key])
                    else:
                        value(kwargs[key])
            except ValidationErrorPydantic as e:
                raise ValidationError(
                    f"There is {e.error_count()} validation errors"
                ) from None
            return func(*args, **kwargs)

        return wrapper

    return decorator
