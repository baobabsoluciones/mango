import os
from unittest import TestCase, mock

from mango.shared.requirement_check import check_dependencies
from pkg_resources import DistributionNotFound



def raise_exception(*args, **kwargs):
    raise DistributionNotFound()


class TestRequirementCheck(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Create mockup.toml
        toml_data = (
            "[project.optional-dependencies]\n"
            "option_1 = ['mocklib<=1.12']\n"
            "option_2 = ['impossible_package_name<=1.12']"
        )
        with open("mockup.toml", "w") as toml_file:
            toml_file.write(toml_data)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove("mockup.toml")

    @mock.patch("mango.shared.requirement_check.pkg_resources")
    def test_check_dependencies(self, mock_pkg_resources):
        mock_pkg_resources.require.value = None
        flag = check_dependencies("option_1", pyproject_path="mockup.toml")
        self.assertTrue(flag)

    @mock.patch("mango.shared.requirement_check.pkg_resources", autospec=True)
    def test_missing_dependency(self, mock_pkg_resources):
        mock_pkg_resources.require.side_effect = raise_exception
        flag = check_dependencies("option_2", pyproject_path="mockup.toml")
        self.assertFalse(flag)

    def test_invalid_option_name(self):
        self.assertRaises(
            KeyError, check_dependencies, "invalid_option", pyproject_path="mockup.toml"
        )

    def test_invalid_path(self):
        self.assertRaises(
            FileNotFoundError,
            check_dependencies,
            "some_option",
            pyproject_path="invalid_path.toml",
        )
