"""
Test utilities for CatastroData tests.

This module provides utility functions and base classes that can be used
across different test files to reduce code duplication and provide
common functionality similar to pytest fixtures but for unittest.
"""

import os
import tempfile
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pandas as pd

from .fixtures.sample_data import (
    MOCK_MULTIPLE_MUNICIPALITIES_DATA,
    MOCK_MUNICIPALITY_DATA,
    MOCK_SUCCESS_RESPONSE,
    TEST_MUNICIPALITY_CODES,
)
from .fixtures.sample_files import create_mock_building_gml, create_mock_zip_with_gml


class BaseTestCase(unittest.TestCase):
    """
    Base test case with common setup and utility methods.

    Provides common functionality that can be inherited by other test classes.
    """

    def setUp(self):
        """Set up common test fixtures."""
        super().setUp()

        # Common mock DataFrames
        self.mock_municipality_df = pd.DataFrame(MOCK_MUNICIPALITY_DATA)
        self.mock_multiple_municipalities_df = pd.DataFrame(
            MOCK_MULTIPLE_MUNICIPALITIES_DATA
        )

        # Test municipality codes
        self.test_municipality_codes = TEST_MUNICIPALITY_CODES

        # Temporary files list for cleanup
        self._temp_files = []
        self._temp_dirs = []

    def tearDown(self):
        """Clean up temporary files and directories."""
        super().tearDown()

        # Clean up temporary files
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        # Clean up temporary directories
        for temp_dir in self._temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                except OSError:
                    pass

    def create_temp_cache_file(self) -> str:
        """
        Create a temporary cache file for testing.

        :return: Path to temporary cache file.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            cache_file = tmp.name

        self._temp_files.append(cache_file)
        return cache_file

    def create_temp_directory(self) -> str:
        """
        Create a temporary directory for testing.

        :return: Path to temporary directory.
        """
        temp_dir = tempfile.mkdtemp()
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def create_mock_http_response(self, **kwargs) -> MagicMock:
        """
        Create a mock HTTP response.

        :param kwargs: Override default response attributes.
        :return: MagicMock configured as HTTP response.
        """
        mock_response = MagicMock()

        # Default values
        defaults = {
            "status_code": MOCK_SUCCESS_RESPONSE["status_code"],
            "content": MOCK_SUCCESS_RESPONSE["content"],
            "headers": MOCK_SUCCESS_RESPONSE["headers"],
        }
        defaults.update(kwargs)

        for attr, value in defaults.items():
            setattr(mock_response, attr, value)

        mock_response.raise_for_status.return_value = None
        return mock_response

    def create_mock_zip_content(self, gml_type: str = "buildings") -> bytes:
        """
        Create mock ZIP content for testing.

        :param gml_type: Type of GML content to create.
        :return: ZIP file content as bytes.
        """
        if gml_type == "buildings":
            return create_mock_zip_with_gml(
                create_mock_building_gml(), "test.building.gml"
            )
        else:
            return create_mock_zip_with_gml(
                f"<gml>{gml_type} content</gml>", f"test.{gml_type}.gml"
            )


class NetworkTestMixin:
    """
    Mixin class for tests that require network access.

    Provides methods to check network availability and skip tests accordingly.
    """

    def skip_if_no_network(self):
        """
        Skip test if no network connection is available.

        Tests connectivity using httpbin.org as a reliable test endpoint.
        """
        import requests

        try:
            response = requests.get("http://httpbin.org/status/200", timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            self.skipTest("No network connection available")

    def skip_if_integration_disabled(self):
        """
        Skip test if integration tests are disabled.

        Checks for RUN_INTEGRATION_TESTS environment variable.
        """
        if not os.environ.get("RUN_INTEGRATION_TESTS"):
            self.skipTest(
                "Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable"
            )


class IntegrationTestCase(BaseTestCase, NetworkTestMixin):
    """
    Base class for integration tests.

    Combines BaseTestCase with NetworkTestMixin and provides
    integration-specific setup.
    """

    def setUp(self):
        """Set up integration test environment."""
        super().setUp()

        # Check prerequisites for integration tests
        self.skip_if_no_network()
        self.skip_if_integration_disabled()

        # Integration test configuration
        self.integration_config = {
            "cache": False,
            "request_timeout": 30,
            "request_interval": 0.5,
            "verbose": False,
        }


def create_test_suite_for_class(test_class, test_methods: Optional[list] = None):
    """
    Create a test suite for a specific test class.

    :param test_class: The test class to create suite for.
    :param test_methods: Optional list of specific test methods to include.
    :return: unittest.TestSuite instance.
    """
    suite = unittest.TestSuite()

    if test_methods:
        for method in test_methods:
            suite.addTest(test_class(method))
    else:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))

    return suite


def run_test_suite(suite, verbosity: int = 2):
    """
    Run a test suite with specified verbosity.

    :param suite: unittest.TestSuite to run.
    :param verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose).
    :return: unittest.TestResult instance.
    """
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


class ParametrizedTestCase(unittest.TestCase):
    """
    Base class for creating parametrized tests in unittest.

    Provides functionality similar to pytest's parametrize decorator.
    """

    @classmethod
    def parametrize(cls, parameter_name: str, values: list):
        """
        Create parametrized test methods.

        :param parameter_name: Name of the parameter.
        :param values: List of values to test with.
        """

        def decorator(test_method):
            for i, value in enumerate(values):
                test_name = f"{test_method.__name__}_{parameter_name}_{i}"

                def create_test(val):
                    def test_func(self):
                        setattr(self, parameter_name, val)
                        return test_method(self)

                    return test_func

                setattr(cls, test_name, create_test(value))

            # Remove original test method
            delattr(cls, test_method.__name__)
            return test_method

        return decorator


# Utility functions for common test patterns


def assert_dataframe_equal(
    test_case: unittest.TestCase, df1: pd.DataFrame, df2: pd.DataFrame
):
    """
    Assert that two DataFrames are equal.

    :param test_case: unittest.TestCase instance for assertions.
    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    """
    try:
        pd.testing.assert_frame_equal(df1, df2)
    except AssertionError as e:
        test_case.fail(f"DataFrames are not equal: {e}")


def assert_gdf_has_geometry(test_case: unittest.TestCase, gdf):
    """
    Assert that a GeoDataFrame has valid geometry.

    :param test_case: unittest.TestCase instance for assertions.
    :param gdf: GeoDataFrame to check.
    """
    test_case.assertIn("geometry", gdf.columns)
    test_case.assertTrue(gdf.geometry.notna().any())
    test_case.assertIsNotNone(gdf.crs)


def create_mock_catastro_instance(**kwargs) -> MagicMock:
    """
    Create a mock CatastroData instance.

    :param kwargs: Attributes to override in the mock.
    :return: MagicMock configured to behave like CatastroData.
    """
    mock_catastro = MagicMock()

    # Default attributes
    mock_catastro._index_loaded = True
    mock_catastro.cache = False
    mock_catastro.municipalities_links = pd.DataFrame(MOCK_MUNICIPALITY_DATA)

    # Override with provided kwargs
    for attr, value in kwargs.items():
        setattr(mock_catastro, attr, value)

    return mock_catastro
