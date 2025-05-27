"""
Unit tests for CatastroData with mocked dependencies.

These tests run quickly and don't require network access.
All external dependencies are mocked to ensure reliability and speed.
"""

import unittest
import zipfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from mango.clients.catastro import CatastroData

from .fixtures.sample_data import (
    CACHE_CONFIGS,
    INVALID_DATATYPES,
    MOCK_ERROR_RESPONSES,
    MOCK_FEED_DATA,
    MOCK_MULTIPLE_MUNICIPALITIES_DATA,
    MOCK_MUNICIPALITY_DATA,
    MOCK_SUCCESS_RESPONSE,
    TEST_MUNICIPALITY_CODES,
    VALID_DATATYPES,
)
from .fixtures.sample_files import (
    MOCK_ZIP_FILES,
    create_invalid_zip,
    create_mock_building_gml,
    create_mock_zip_with_gml,
    create_mock_zip_without_gml,
)


class TestCatastroDataUnit(unittest.TestCase):
    """Unit tests for CatastroData with mocked dependencies."""

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Common mock DataFrames
        self.mock_df = pd.DataFrame(MOCK_MUNICIPALITY_DATA)
        self.mock_multiple_df = pd.DataFrame(MOCK_MULTIPLE_MUNICIPALITIES_DATA)

        # Common test file paths
        self.test_cache_file = "test_cache.json"
        self.fake_cache_file = "fake_path.json"

        # Test municipality codes
        self.test_municipality_code = TEST_MUNICIPALITY_CODES["small"]
        self.nonexistent_code = TEST_MUNICIPALITY_CODES["nonexistent"]

    # ==================== INITIALIZATION TESTS ====================

    @patch("mango.clients.catastro.os.path.exists", return_value=False)
    @patch("mango.clients.catastro.CatastroData._fetch_and_parse_links")
    def test_initialization_without_cache(self, mock_fetch, _):
        """
        Test CatastroData initialization when cache file doesn't exist.

        Verifies that:
        - Cache setting is preserved
        - Index is loaded from fresh data
        - _fetch_and_parse_links is called exactly once
        """
        mock_fetch.return_value = self.mock_df

        catastro = CatastroData(cache=True, cache_file_path=self.fake_cache_file)

        self.assertTrue(catastro.cache)
        self.assertTrue(catastro._index_loaded)
        mock_fetch.assert_called_once()

    @patch("mango.clients.catastro.pd.read_json")
    @patch("mango.clients.catastro.os.path.exists", return_value=True)
    def test_initialization_with_cache(self, _, mock_read_json):
        """
        Test CatastroData initialization when cache file exists.

        Verifies that:
        - Index is loaded from cache file
        - pd.read_json is called exactly once
        - No fresh data fetching occurs
        """
        mock_read_json.return_value = self.mock_df

        catastro = CatastroData(cache=True, cache_file_path=self.test_cache_file)

        self.assertTrue(catastro._index_loaded)
        mock_read_json.assert_called_once()

    def test_initialization_without_cache_disabled(self):
        """Test initialization with cache disabled."""
        with patch(
            "mango.clients.catastro.CatastroData._fetch_and_parse_links"
        ) as mock_fetch:
            mock_fetch.return_value = self.mock_df

            catastro = CatastroData(cache=False)

            self.assertFalse(catastro.cache)
            self.assertTrue(catastro._index_loaded)
            mock_fetch.assert_called_once()

    # ==================== DOWNLOAD TESTS ====================

    @patch("mango.clients.catastro.requests.get")
    def test_download_zip_content_success(self, mock_get):
        """Test successful zip content download."""
        mock_response = MagicMock()
        mock_response.status_code = MOCK_SUCCESS_RESPONSE["status_code"]
        mock_response.content = MOCK_SUCCESS_RESPONSE["content"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        catastro = CatastroData()
        content = catastro._download_zip_content("http://fake-url.com")

        self.assertEqual(content, MOCK_SUCCESS_RESPONSE["content"])
        mock_get.assert_called_once_with("http://fake-url.com")

    @patch("mango.clients.catastro.requests.get")
    def test_download_zip_content_timeout(self, mock_get):
        """Test download timeout handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        catastro = CatastroData()
        content = catastro._download_zip_content("http://fake-url.com")

        self.assertIsNone(content)

    @patch("mango.clients.catastro.requests.get")
    def test_download_zip_content_http_error(self, mock_get):
        """Test HTTP error handling."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_get.return_value = mock_response

        catastro = CatastroData()
        content = catastro._download_zip_content("http://fake-url.com")

        self.assertIsNone(content)

    @patch("mango.clients.catastro.requests.get")
    def test_download_zip_content_connection_error(self, mock_get):
        """Test connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        catastro = CatastroData()
        content = catastro._download_zip_content("http://fake-url.com")

        self.assertIsNone(content)

    # ==================== FILE PROCESSING TESTS ====================

    def test_extract_gml_from_zip_buildings_success(self):
        """Test successful GML extraction from ZIP for Buildings."""
        zip_content = create_mock_zip_with_gml(
            create_mock_building_gml(), "test.building.gml"
        )

        catastro = CatastroData()
        result = catastro._extract_gml_from_zip(
            zip_content, "Buildings", "Buildings", "http://test.com"
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, BytesIO)
        # Verify content can be read
        content = result.read()
        self.assertIn(b"<bu:Building", content)

    def test_extract_gml_from_zip_invalid_datatype(self):
        """Test error handling for invalid datatype in extract_gml_from_zip."""
        catastro = CatastroData()

        with self.assertRaises(ValueError) as context:
            catastro._extract_gml_from_zip(
                b"fake-zip", "INVALID_TYPE", None, "http://test.com"
            )

        self.assertIn("Invalid datatype", str(context.exception))

    def test_extract_gml_from_zip_file_not_found(self):
        """Test error when expected GML file not found in zip."""
        zip_content = create_mock_zip_without_gml()

        catastro = CatastroData()

        with self.assertRaises(FileNotFoundError) as context:
            catastro._extract_gml_from_zip(
                zip_content, "Buildings", "Buildings", "http://test.com"
            )

        self.assertIn("No file with suffix", str(context.exception))

    def test_extract_gml_from_zip_bad_zip(self):
        """Test error handling for invalid ZIP content."""
        invalid_zip = create_invalid_zip()

        catastro = CatastroData()

        with self.assertRaises(zipfile.BadZipFile):
            catastro._extract_gml_from_zip(
                invalid_zip, "Buildings", "Buildings", "http://test.com"
            )

    @patch("mango.clients.catastro.os.makedirs")
    def test_extract_zip_to_folder_success(self, mock_makedirs):
        """Test successful extraction of zip to folder."""
        zip_content = MOCK_ZIP_FILES["buildings"]

        catastro = CatastroData()

        with patch("zipfile.ZipFile.extract") as mock_extract:
            result = catastro.extract_zip_to_folder(zip_content, "/fake/path")

            self.assertTrue(result)
            mock_makedirs.assert_called_once_with("/fake/path", exist_ok=True)

    def test_extract_zip_to_folder_bad_zip(self):
        """Test error handling for bad zip file in folder extraction."""
        catastro = CatastroData()
        result = catastro.extract_zip_to_folder(b"not-a-zip", "/fake/path")

        self.assertFalse(result)

    # ==================== BUSINESS LOGIC TESTS ====================

    def test_find_link_success(self):
        """Test finding download link for municipality and datatype."""
        catastro = CatastroData()
        catastro.municipalities_links = self.mock_df

        link = catastro._find_link(self.test_municipality_code, "Buildings")

        expected_link = MOCK_MUNICIPALITY_DATA["zip_link"]["0"]
        self.assertEqual(link, expected_link)

    def test_find_link_not_found(self):
        """Test finding download link when municipality/datatype not found."""
        catastro = CatastroData()
        catastro.municipalities_links = self.mock_df

        link = catastro._find_link(self.nonexistent_code, "Buildings")

        self.assertIsNone(link)

    def test_available_municipalities_valid_type(self):
        """Test getting available municipalities for valid datatype."""
        catastro = CatastroData()
        catastro._index_loaded = True
        catastro.municipalities_links = self.mock_multiple_df

        result = catastro.available_municipalities("Buildings")

        self.assertEqual(len(result), 2)  # Two buildings entries in mock data
        self.assertIn("catastro_municipality_code", result.columns)
        self.assertIn("catastro_municipality_name", result.columns)

    def test_available_municipalities_invalid_type(self):
        """Test error handling for invalid datatype in available_municipalities."""
        catastro = CatastroData()
        catastro._index_loaded = True

        for invalid_type in INVALID_DATATYPES:
            with self.subTest(datatype=invalid_type):
                with self.assertRaises(ValueError) as context:
                    catastro.available_municipalities(invalid_type)

                self.assertIn("Invalid datatype", str(context.exception))

    def test_ensure_index_loaded_success(self):
        """Test successful index loading check."""
        catastro = CatastroData()
        catastro._index_loaded = True

        # Should not raise any exception
        catastro._ensure_index_loaded()

    def test_ensure_index_loaded_failure(self):
        """Test error when index is not loaded."""
        catastro = CatastroData()
        catastro._index_loaded = False

        with self.assertRaises(RuntimeError) as context:
            catastro._ensure_index_loaded()

        self.assertIn("Municipality index could not be loaded", str(context.exception))

    # ==================== FEED PARSING TESTS ====================

    def test_parse_territorial_entry_success(self):
        """Test successful parsing of territorial entry."""
        mock_entry = MagicMock()
        mock_entry.title = MOCK_FEED_DATA["territorial_entry"]["title"]
        mock_entry.link = MOCK_FEED_DATA["territorial_entry"]["link"]

        catastro = CatastroData()
        result = catastro._parse_territorial_entry(mock_entry)

        self.assertIsNotNone(result)
        self.assertEqual(result["territorial_code"], "02")
        self.assertEqual(result["territorial_name"], "Albacete")
        self.assertEqual(result["link"], MOCK_FEED_DATA["territorial_entry"]["link"])

    def test_parse_territorial_entry_invalid_title(self):
        """Test handling of invalid territorial entry title."""
        mock_entry = MagicMock()
        mock_entry.title = MOCK_FEED_DATA["invalid_territorial"]["title"]
        mock_entry.link = MOCK_FEED_DATA["invalid_territorial"]["link"]

        catastro = CatastroData()
        result = catastro._parse_territorial_entry(mock_entry)

        self.assertIsNone(result)

    def test_parse_municipality_entry_success(self):
        """Test successful parsing of municipality entry."""
        mock_entry = MagicMock()
        mock_entry.title = MOCK_FEED_DATA["municipality_entry"]["title"]
        mock_entry.link = MOCK_FEED_DATA["municipality_entry"]["link"]

        catastro = CatastroData()
        result = catastro._parse_municipality_entry(mock_entry)

        self.assertIsNotNone(result)
        self.assertEqual(result["municipality_code"], "02001")
        self.assertEqual(result["municipality_name"], "ABENGIBRE")
        self.assertEqual(result["link"], MOCK_FEED_DATA["municipality_entry"]["link"])

    def test_parse_municipality_entry_invalid_format(self):
        """Test handling of invalid municipality entry format."""
        mock_entry = MagicMock()
        mock_entry.title = MOCK_FEED_DATA["invalid_municipality"]["title"]
        mock_entry.link = MOCK_FEED_DATA["invalid_municipality"]["link"]

        catastro = CatastroData()
        result = catastro._parse_municipality_entry(mock_entry)

        self.assertIsNone(result)

    @patch("mango.clients.catastro.feedparser.parse")
    def test_fetch_feed_success(self, mock_parse):
        """Test successful feed fetching."""
        mock_feed = MagicMock()
        mock_feed.entries = [MagicMock()]
        mock_feed.feed = MagicMock()
        mock_parse.return_value = mock_feed

        catastro = CatastroData()
        result = catastro._fetch_feed("http://test.com/feed.xml")

        self.assertIsNotNone(result)
        mock_parse.assert_called_once_with("http://test.com/feed.xml")

    @patch("mango.clients.catastro.feedparser.parse")
    def test_fetch_feed_empty(self, mock_parse):
        """Test handling of empty feed."""
        mock_feed = MagicMock()
        mock_feed.entries = []
        mock_feed.feed = None
        mock_parse.return_value = mock_feed

        catastro = CatastroData()
        result = catastro._fetch_feed("http://test.com/feed.xml")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
