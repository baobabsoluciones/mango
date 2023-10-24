from unittest import TestCase

from mango.config import BaseConfig, ConfigParameter
from mango.tests.const import normalize_path


class TestConfig(BaseConfig):
    __params = {
        "main": [
            ConfigParameter("some_parameter", int),
            ConfigParameter("other_parameter", str, validate=["hello", "world"]),
        ],
        "other_section": [
            ConfigParameter("float_parameter", float, default=1.0),
            ConfigParameter("bool_parameter", bool),
            ConfigParameter("list_parameter", list, secondary_type=int),
            ConfigParameter("another_list", list, secondary_type=str),
            ConfigParameter("has_default", str, default="default"),
            ConfigParameter("bad_value", float, default=1.0),
        ],
        "missing_section": [
            ConfigParameter("missing_parameter", str, default="default")
        ],
    }

    def __init__(self, file_name):
        super().__init__(file_name, self.__params)


class AnotherTestConfig(BaseConfig):
    __params = {
        "main": [
            ConfigParameter("some_parameter", dict),
            ConfigParameter("other_parameter", str, validate=["hello", "world"]),
        ],
        "other_section": [
            ConfigParameter("float_parameter", float),
            ConfigParameter("bool_parameter", bool),
            ConfigParameter("list_parameter", list, secondary_type=int),
            ConfigParameter("another_list", list, secondary_type=str),
            ConfigParameter("has_default", str, default="default"),
            ConfigParameter("bad_value", float, default=1.0),
        ],
    }

    def __init__(self, file_name):
        super().__init__(file_name, self.__params)


class ConfigTest(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_parameter_config_parsing(self):
        config = TestConfig(normalize_path("data/test_good.cfg"))
        self.assertEqual(config("some_parameter"), 1)
        self.assertEqual(config("other_parameter"), "hello")
        self.assertEqual(config("float_parameter"), 1.0)
        self.assertEqual(config("bool_parameter"), True)
        self.assertEqual(config("list_parameter"), [1, 2, 3, 4, 5])
        self.assertEqual(config("another_list"), ["hello", "world"])
        self.assertEqual(config("has_default"), "default")

    def test_no_file(self):
        with self.assertRaises(FileNotFoundError):
            TestConfig("not_a_file.cfg")

    def test_not_valid_value(self):
        with self.assertRaises(ValueError):
            TestConfig(normalize_path("data/test_bad_value.cfg"))

    def test_not_valid_type(self):
        with self.assertRaises(ValueError):
            TestConfig(normalize_path("data/test_bad_type.cfg"))

    def test_not_valid_parsing_type(self):
        with self.assertRaises(TypeError):
            AnotherTestConfig(normalize_path("data/test_bad_type.cfg"))

    def test_parameter_repr(self):
        param = ConfigParameter("some_parameter", int)
        self.assertEqual(
            param.__repr__(), "ConfigParameter('some_parameter', <class 'int'>)"
        )
        other_param = ConfigParameter("some_parameter", int, default=1)
        self.assertEqual(
            other_param.__repr__(),
            "ConfigParameter('some_parameter', <class 'int'>, 1)",
        )
