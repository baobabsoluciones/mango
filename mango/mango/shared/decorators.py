import warnings
from collections.abc import Mapping
from functools import wraps
from typing import Literal

from fastjsonschema import compile
from fastjsonschema.exceptions import JsonSchemaValueException
from mango.processing import load_json
from mango.shared.exceptions import ValidationError
from pydantic_core import ValidationError as ValidationErrorPydantic


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
    """
    Decorator to validate the arguments of a function using pydantic models. The decorator can be used in two ways:
    1. Passing a single positional argument: the pydantic model to validate the arguments against.
    2. Passing multiple keyword arguments: the name of the argument to validate and the pydantic model to validate (useful
    when the argument is a dictionary or a list of dictionaries).

    :param validator: The pydantic model to validate all the arguments against
    :param on_validation_error: What to do when there is a validation error. One of 'raise', 'warn', 'ignore'
    :param strict_validation: Whether to validate the arguments strictly or not (if false pydantic will try to convert)
    :param named_validators: The name of the argument to validate and the pydantic model to validate
    :return: The original function with arguments validated


    Usage
    -----

    1. Passing a single positional argument:

    >>> from mango.shared import pydantic_validation
    >>> from pydantic import BaseModel
    >>> class DummyArgs(BaseModel):
    ...     name: str
    ...     x: int
    ...     y: int
    >>> @pydantic_validation(DummyArgs)
    ... def do_nothing(*, name: str, x: int, y: int):
    ...     return True
    >>> do_nothing(name="random", x=0, y=1) # No error
    ... True
    >>> do_nothing(name="random", x=0, y="1") # Error (validation is strict by default can be changed with strict_validation=False)
    ... Raises ValidationError

    2. Passing multiple keyword arguments for JSON arguments (For compatibility with the previous version of the decorator) or if
    just want to validate some of the arguments:

    >>> from mango.shared import pydantic_validation
    >>> from pydantic import BaseModel
    >>> class JSONModel(BaseModel):
    ...     self.model_config = {"extra": "forbid"} # This is to avoid extra arguments in the JSON
    ...     c_1: str
    ...     c_2: int
    >>> @pydantic_validation(c=JSONModel)
    ... def dummy(a: str, b: int, c: dict):
    ...     return True
    >>> dummy(a="a", b=1, c={"c_1": "a", "c_2": 1}) # No error
    ... True
    >>> dummy(a="a", b=1, c={"c_1": "a", "c_2": "1"}) # Error
    ... Raises ValidationError

    Intended usage:

    >>> from mango.shared import pydantic_validation
    >>> from pydantic import BaseModel
    >>> class JSONModel(BaseModel):
    ...     self.model_config = {"extra": "forbid"}
    ...     c_1: str
    ...     c_2: int
    >>> class DummyArgs(BaseModel):
    ...     name: str
    ...     x: int
    ...     y: int
    ...     c: JSONModel
    >>> @pydantic_validation(DummyArgs)
    ... def do_nothing(*, name: str, x: int, y: int, c: dict):
    ...     return True
    >>> do_nothing(name="random", x=0, y=1, c={"c_1": "a", "c_2": 1}) # No error
    ... True
    >>> do_nothing(name="random", x=0, y=1, c={"c_1": "a", "c_2": "1"}) # Error
    ... Raises ValidationError
    """

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
