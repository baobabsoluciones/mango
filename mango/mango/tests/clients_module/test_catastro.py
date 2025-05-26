import unittest
from unittest.mock import patch

import pandas as pd

from mango.clients.catastro import CatastroData


class TestCatastroData(unittest.TestCase):

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Common mock data for municipality links
        self.mock_municipality_data = {
            "territorial_office_code": {"0": "02"},
            "territorial_office_name": {"0": "Albacete"},
            "catastro_municipality_code": {"0": "02001"},
            "catastro_municipality_name": {"0": "ABENGIBRE"},
            "datatype": {"0": "Buildings"},
            "zip_link": {
                "0": r"http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02001-ABENGIBRE/A.ES.SDGC.BU.02001.zip",
            },
        }

        # Common test file paths
        self.test_cache_file = "test_cache.json"
        self.fake_cache_file = "fake_path.json"

        # Common mock DataFrame
        self.mock_df = pd.DataFrame(self.mock_municipality_data)

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


if __name__ == "__main__":
    unittest.main()
