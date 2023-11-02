import warnings
from collections.abc import Mapping
from functools import wraps
from typing import Literal

from fastjsonschema import compile
from fastjsonschema.exceptions import JsonSchemaValueException
from pydantic_core import ValidationError as ValidationErrorPydantic

from mango.processing import load_json
from mango.shared.exceptions import ValidationError


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


def pydantic_validation(
    *validator,
    on_validation_error: Literal["raise", "warn", "ignore"] = "raise",
    strict_validation: bool = True,
    **named_validators,
):
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
            if args:
                raise ValueError(
                    "This decorator only works if function takes keyword arguments"
                )
            validation_errors = []
            if validator:
                try:
                    validator[0].model_validate(kwargs, strict=strict_validation)
                except ValidationErrorPydantic as e:
                    validation_errors.extend(e.errors())
            if named_validators:
                try:
                    for key, value in named_validators.items():
                        if isinstance(kwargs[key], Mapping):
                            value.model_validate(kwargs[key], strict=strict_validation)
                        else:
                            value.model_validate(kwargs[key], strict=strict_validation)
                except ValidationErrorPydantic as e:
                    validation_errors.extend(e.errors())
            if validation_errors:
                more_than_one_error = len(validation_errors) > 1
                if more_than_one_error:
                    msg = f"There are {len(validation_errors)} validation errors: {validation_errors}"
                else:
                    msg = f"There is {len(validation_errors)} validation error: {validation_errors[0]}"
                if on_validation_error == "raise":
                    raise ValidationError(msg) from None
                elif on_validation_error == "warn":
                    warnings.warn(msg)
                elif on_validation_error == "ignore":
                    pass
                else:
                    raise ValueError(
                        f"on_validation_error must be one of 'raise', 'warn', 'ignore' but it is {on_validation_error}"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator
