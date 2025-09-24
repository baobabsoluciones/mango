from unittest import TestCase

from mango.shared import validate_args, ValidationError, pydantic_validation
from mango.tests.const import VALIDATION_SCHEMA, normalize_path
from mango.validators.arcgis import LocationsList, Locations
from pydantic import BaseModel


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

    def test_pydantic_argument_validation(self):
        class DummyValidator(BaseModel):
            a: str
            b: int

        @pydantic_validation(DummyValidator, on_validation_error="warn")
        def dummy2(a: str, b: int):
            return True

        dummy2(a="a", b=1)
        dummy2(a="a", b="1")
        dummy2(a="a", b=1.0)
        dummy2(a=1, b="1")

        class JSONModel(BaseModel):
            self.model_config = {"extra": "forbid"}
            c_1: str
            c_2: int

        class DummyValidator2(BaseModel):
            a: str
            b: int
            c: JSONModel

        # Commented due to conflict with pyomo
        # @pydantic_validation(DummyValidator2, c=JSONModel, on_validation_error="warn")
        # def dummy3(a: str, b: int, c: dict):
        #     return True
        #
        # dummy3(a="a", b=1, c={"c_1": "a", "c_2": 1})
        # with self.assertWarns(Warning):
        #     dummy3(a="a", b=1, c={"c_1": "a", "c_2": "1"})

        # With raise
        @pydantic_validation(DummyValidator2, c=JSONModel, on_validation_error="raise")
        def dummy4(a: str, b: int, c: dict):
            return True

        # No error
        dummy4(a="a", b=1, c={"c_1": "a", "c_2": 1})
        self.assertRaises(
            ValidationError, dummy4, a="a", b=1, c={"c_1": "a", "c_2": "1"}
        )

        # Invalid on_validation_error
        @pydantic_validation(DummyValidator2, c=JSONModel, on_validation_error="random")
        def dummy5(a: str, b: int, c: dict):
            return True

        self.assertRaises(ValueError, dummy5, a="a", b=1, c={"c_1": "a", "c_2": "!"})

        # Without strict_validation
        @pydantic_validation(
            DummyValidator2,
            c=JSONModel,
            on_validation_error="raise",
            strict_validation=False,
        )
        def dummy6(a: str, b: int, c: dict):
            return True

        dummy6(a="a", b=1, c={"c_1": "a", "c_2": "1"})
        dummy6(a="a", b=1, c={"c_1": "a", "c_2": 1.0})
        self.assertRaises(
            ValidationError, dummy6, a="a", b=1, c={"c_1": "a", "c_2": "!"}
        )

        # args instead of kwargs
        class DummyValidator3(BaseModel):
            a: str
            b: int

        @pydantic_validation(DummyValidator3, on_validation_error="raise")
        def dummy7(a: str, b: int):
            return True

        self.assertRaises(ValueError, dummy7, "a", 1)

        # Ignore validation
        @pydantic_validation(DummyValidator3, on_validation_error="ignore")
        def dummy8(a: str, b: int):
            return True

        dummy8(a="a", b="1")

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
