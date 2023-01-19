from functools import wraps

from fastjsonschema import compile
from fastjsonschema.exceptions import JsonSchemaValueException
from mango.processing import load_json
from .exceptions import ValidationError


def validate_args(**schemas):
    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper function validates the arguments against a schema and then calls
            the original function.

            :param args: Pass a non-keyworded, variable-length argument list to the function
            :param kwargs: Pass the arguments to the function
            :return: The same as the original function
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
