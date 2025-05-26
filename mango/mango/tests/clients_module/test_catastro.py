import unittest
from unittest.mock import patch

import pandas as pd

from mango.clients.catastro import CatastroData


class TestCatastroData(unittest.TestCase):

    @patch("mango.clients.catastro.os.path.exists", return_value=False)
    @patch("mango.clients.catastro.CatastroData._fetch_and_parse_links")
    def test_initialization_without_cache(self, mock_fetch, _):
        """Test CatastroData initialization when cache file doesn't exist."""
        mock_df = pd.DataFrame(
            {
                "territorial_office_code": {"0": "02"},
                "territorial_office_name": {"0": "Albacete"},
                "catastro_municipality_code": {"0": "02001"},
                "catastro_municipality_name": {"0": "ABENGIBRE"},
                "datatype": {"0": "Buildings"},
                "zip_link": {
                    "0": "http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02001-ABENGIBRE/A.ES.SDGC.BU.02001.zip",
                },
            }
        )
        mock_fetch.return_value = mock_df

        catastro = CatastroData(cache=True, cache_file_path="fake_path.json")

        self.assertTrue(catastro.cache)
        self.assertTrue(catastro._index_loaded)
        mock_fetch.assert_called_once()

    @patch("mango.clients.catastro.pd.read_json")
    @patch("mango.clients.catastro.os.path.exists", return_value=True)
    def test_initialization_with_cache(self, _, mock_read_json):
        """Test CatastroData initialization when cache file exists."""
        mock_df = pd.DataFrame(
            {
                "territorial_office_code": {"0": "02"},
                "territorial_office_name": {"0": "Albacete"},
                "catastro_municipality_code": {"0": "02001"},
                "catastro_municipality_name": {"0": "ABENGIBRE"},
                "datatype": {"0": "Buildings"},
                "zip_link": {
                    "0": "http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02001-ABENGIBRE/A.ES.SDGC.BU.02001.zip",
                },
            }
        )
        mock_read_json.return_value = mock_df

        catastro = CatastroData(cache=True, cache_file_path="cache.json")

        self.assertTrue(catastro._index_loaded)
        mock_read_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
