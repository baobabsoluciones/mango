from unittest import TestCase
from artic.shared import validate_args, ValidationError
from artic.tests.const import normalize_path, VALIDATION_SCHEMA


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

