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
    """
    Decorator to validate function arguments against JSON schemas.

    Validates keyword arguments against provided JSON schemas before
    executing the decorated function. Schemas can be provided as
    dictionaries or file paths to JSON schema files.

    :param schemas: Dictionary mapping argument names to their validation schemas
    :type schemas: dict
    :return: Decorator function
    :rtype: callable
    :raises ValidationError: If any argument fails validation
    :raises ValueError: If schema format is invalid

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer", "minimum": 0}
        ...     },
        ...     "required": ["name"]
        ... }
        >>> @validate_args(user=schema)
        ... def create_user(user):
        ...     return f"Created user: {user['name']}"
        >>> create_user(user={"name": "John", "age": 30})
        'Created user: John'
    """

    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Validate arguments against schemas and execute the original function.

            Validates each specified argument against its corresponding schema
            before calling the original function. If validation fails, raises
            a ValidationError with details about the validation failure.

            :param args: Positional arguments passed to the function
            :type args: tuple
            :param kwargs: Keyword arguments passed to the function
            :type kwargs: dict
            :return: Result of the original function
            :rtype: Any
            :raises ValidationError: If argument validation fails
            :raises ValueError: If schema format is invalid
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
    Decorator to validate function arguments using Pydantic models.

    Provides comprehensive argument validation using Pydantic models with
    flexible error handling options. Supports both global validation (all
    arguments) and selective validation (specific arguments only).

    :param validator: Pydantic model to validate all arguments against
    :type validator: BaseModel, optional
    :param on_validation_error: Action to take on validation failure
    :type on_validation_error: Literal["raise", "warn", "ignore"]
    :param strict_validation: Whether to use strict validation (no type coercion)
    :type strict_validation: bool
    :param named_validators: Dictionary mapping argument names to Pydantic models
    :type named_validators: dict[str, BaseModel]
    :return: Decorator function
    :rtype: callable
    :raises ValidationError: If validation fails and on_validation_error="raise"
    :raises ValueError: If function has positional arguments or invalid error handling option

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class UserModel(BaseModel):
        ...     name: str
        ...     age: int
        ...     email: str
        >>>
        >>> @pydantic_validation(UserModel)
        ... def create_user(*, name: str, age: int, email: str):
        ...     return f"User {name} created"
        >>>
        >>> create_user(name="John", age=30, email="john@example.com")
        'User John created'
        >>>
        >>> # With selective validation
        >>> @pydantic_validation(user_data=UserModel)
        ... def process_user(user_data: dict, action: str):
        ...     return f"Processing {user_data['name']} with action {action}"
    """

    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Validate arguments using Pydantic models and execute the original function.

            Performs validation on specified arguments using Pydantic models.
            Handles validation errors according to the configured error handling
            strategy (raise, warn, or ignore).

            :param args: Positional arguments (not supported, must be empty)
            :type args: tuple
            :param kwargs: Keyword arguments to validate
            :type kwargs: dict
            :return: Result of the original function
            :rtype: Any
            :raises ValueError: If positional arguments are provided
            :raises ValidationError: If validation fails and error handling is set to raise
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
