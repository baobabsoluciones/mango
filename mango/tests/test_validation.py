from unittest import TestCase

from pydantic import BaseModel

from mango.shared import validate_args, ValidationError
from mango.shared.decorators import pydantic_validation
from mango.tests.const import VALIDATION_SCHEMA, normalize_path
from mango.validators.arcgis import LocationsList, Locations


class ValidationTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_validation_correct_path(self):
        @validate_args(argument=normalize_path("../schemas/location.json"))
        def do_nothing(*, argument):
            return True

        data = [{"name": "random", "x": 0, "y": 1}]

        result = do_nothing(argument=data)
        self.assertEqual(result, True)

    def test_pydantic_validation_list(self):
        @pydantic_validation(argument=LocationsList)
        def do_nothing(*, argument):
            return True

        data = [{"name": "random", "x": 0, "y": 1}]

        result = do_nothing(argument=data)
        self.assertEqual(result, True)

    def test_pydantic_validation_dict(self):
        @pydantic_validation(argument=Locations)
        def do_nothing(*, argument):
            return True

        data = {"name": "random", "x": 0, "y": 1}

        result = do_nothing(argument=data)
        self.assertEqual(result, True)

    def test_pydantic_error_value(self):
        @pydantic_validation(argument=Locations)
        def do_nothing(*, argument):
            return True

        data = {"name": "random"}

        self.assertRaises(ValidationError, do_nothing, argument=data)

    def test_pydantic_error_list(self):
        @pydantic_validation(argument=LocationsList)
        def do_nothing(*, argument):
            return True

        data = [{"name": "random"}]

        self.assertRaises(ValidationError, do_nothing, argument=data)

    def test_validation_correct_object(self):
        schema = VALIDATION_SCHEMA

        @validate_args(argument=schema)
        def do_nothing(*, argument):
            return True

        data = [{"name": "random", "x": 0, "y": 1}, {"name": "random", "x": 0, "y": 1}]

        result = do_nothing(argument=data)
        self.assertEqual(result, True)

    def test_validation_error(self):
        schema = VALIDATION_SCHEMA

        @validate_args(argument=schema)
        def do_nothing(*, argument):
            return True

        data = [{"name": "random"}]

        self.assertRaises(ValidationError, do_nothing, argument=data)

    def test_invalid_schema(self):
        @validate_args(argument=[])
        def do_nothing(*, argument):
            return True

        data = [{"name": "random"}]

        self.assertRaises(ValueError, do_nothing, argument=data)
