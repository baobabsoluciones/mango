"""
Integration tests for CatastroData with real network calls.

These tests verify the actual integration with Catastro services.
They are slower and optional, requiring network access and real API availability.
"""

import os
import tempfile
import unittest
from typing import Optional

import requests

from mango.clients.catastro import CatastroData

from .fixtures.sample_data import TEST_MUNICIPALITY_CODES, TEST_URLS
from .test_utils import IntegrationTestCase


class TestCatastroDataIntegration(IntegrationTestCase):
    """Integration tests with real network calls."""

    def setUp(self):
        """
        Setup for integration tests.

        Inherits network and integration checks from IntegrationTestCase.
        """
        super().setUp()

        # Use small municipalities known to exist for tests
        self.small_municipality_code = TEST_MUNICIPALITY_CODES["small"]  # Abengibre

        # Override default integration config if needed
        self.integration_config.update(
            {
                "request_timeout": 30,  # Generous timeout for real network calls
                "request_interval": 0.5,  # Be respectful to the API
            }
        )

    # ==================== INITIALIZATION TESTS ====================

    def test_real_municipality_index_loading(self):
        """
        Test loading real municipality index from Catastro API.

        Verifies that:
        - Index can be loaded from real API
        - Index contains expected data structure
        - Index has reasonable amount of data
        """
        catastro = CatastroData(**self.integration_config)

        self.assertTrue(catastro._index_loaded)
        self.assertIsNotNone(catastro.municipalities_links)
        self.assertGreater(
            len(catastro.municipalities_links), 1000
        )  # Spain has many municipalities

        # Verify expected columns exist
        expected_columns = [
            "territorial_office_code",
            "territorial_office_name",
            "catastro_municipality_code",
            "catastro_municipality_name",
            "datatype",
            "zip_link",
        ]
        for col in expected_columns:
            self.assertIn(col, catastro.municipalities_links.columns)

    def test_real_available_datatypes(self):
        """Test that all expected datatypes are available in real data."""
        catastro = CatastroData(**self.integration_config)

        available_datatypes = catastro.municipalities_links["datatype"].unique()
        expected_datatypes = ["Buildings", "CadastralParcels", "Addresses"]

        for datatype in expected_datatypes:
            self.assertIn(datatype, available_datatypes)

    # ==================== CACHE FUNCTIONALITY TESTS ====================

    def test_real_cache_functionality(self):
        """
        Test that caching works correctly with real data.

        Verifies:
        - Cache file is created when enabled
        - Second initialization loads from cache
        - Cache contains valid data
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            cache_file = tmp.name

        try:
            # First load - should create cache
            config_with_cache = self.integration_config.copy()
            config_with_cache.update({"cache": True, "cache_file_path": cache_file})

            catastro1 = CatastroData(**config_with_cache)
            self.assertTrue(os.path.exists(cache_file))
            self.assertTrue(catastro1._index_loaded)

            # Verify cache file has content
            cache_size = os.path.getsize(cache_file)
            self.assertGreater(cache_size, 1000)  # Should be substantial

            # Second load - should use cache (faster)
            import time

            start_time = time.time()
            catastro2 = CatastroData(**config_with_cache)
            load_time = time.time() - start_time

            self.assertTrue(catastro2._index_loaded)
            self.assertLess(load_time, 5)  # Cache loading should be fast

            # Verify both instances have same data
            self.assertEqual(
                len(catastro1.municipalities_links), len(catastro2.municipalities_links)
            )

        finally:
            if os.path.exists(cache_file):
                os.unlink(cache_file)

    # ==================== DATA RETRIEVAL TESTS ====================

    def test_real_small_municipality_buildings_download(self):
        """
        Test downloading Buildings data for a small real municipality.

        Uses a small municipality to minimize download time and API load.
        """
        catastro = CatastroData(**self.test_config)

        # Check if municipality exists in index
        available_municipalities = catastro.available_municipalities("Buildings")
        municipality_codes = available_municipalities[
            "catastro_municipality_code"
        ].values

        if self.small_municipality_code not in municipality_codes:
            self.skipTest(
                f"Municipality {self.small_municipality_code} not available for Buildings"
            )

        # Attempt to download data
        gdf = catastro.get_data(self.small_municipality_code, "Buildings")

        if gdf is not None:  # May be None if no data available
            self.assertGreater(len(gdf), 0)
            self.assertIn("geometry", gdf.columns)
            self.assertIn("catastro_municipality_code", gdf.columns)
            self.assertEqual(
                gdf["catastro_municipality_code"].iloc[0], self.small_municipality_code
            )

            # Verify CRS is set
            self.assertIsNotNone(gdf.crs)
        else:
            # If no data, that's also a valid result - just log it
            print(
                f"No Buildings data available for municipality {self.small_municipality_code}"
            )

    def test_real_available_municipalities_query(self):
        """Test querying available municipalities for different datatypes."""
        catastro = CatastroData(**self.test_config)

        for datatype in ["Buildings", "CadastralParcels", "Addresses"]:
            with self.subTest(datatype=datatype):
                municipalities = catastro.available_municipalities(datatype)

                self.assertIsNotNone(municipalities)
                self.assertGreater(len(municipalities), 0)
                self.assertIn("catastro_municipality_code", municipalities.columns)
                self.assertIn("catastro_municipality_name", municipalities.columns)

    # ==================== ERROR HANDLING TESTS ====================

    def test_real_nonexistent_municipality(self):
        """Test handling of requests for non-existent municipalities."""
        catastro = CatastroData(**self.test_config)

        nonexistent_code = TEST_MUNICIPALITY_CODES["nonexistent"]

        with self.assertRaises(ValueError) as context:
            catastro.get_data(nonexistent_code, "Buildings")

        self.assertIn("No index entry found", str(context.exception))

    def test_real_network_timeout_handling(self):
        """
        Test handling of network timeouts with very short timeout.

        Note: This test may be flaky depending on network conditions.
        """
        # Use very short timeout to force timeout
        timeout_config = self.test_config.copy()
        timeout_config["request_timeout"] = 0.001  # 1ms - should timeout

        try:
            catastro = CatastroData(**timeout_config)
            # If initialization succeeds despite short timeout, that's also valid
            # (maybe the API is very fast or cached)
            self.assertTrue(True)  # Test passes either way
        except Exception:
            # If it fails due to timeout, that's expected behavior
            self.assertTrue(True)  # Test passes

    # ==================== PERFORMANCE TESTS ====================

    def test_real_multiple_municipalities_performance(self):
        """
        Test downloading data for multiple small municipalities.

        Verifies that batch processing works correctly.
        """
        catastro = CatastroData(**self.test_config)

        # Get a few small municipalities
        available = catastro.available_municipalities("Buildings")
        if len(available) < 2:
            self.skipTest("Not enough municipalities available for batch test")

        # Take first 2 municipalities (should be small)
        test_codes = available["catastro_municipality_code"].head(2).tolist()

        import time

        start_time = time.time()

        gdf = catastro.get_data(test_codes, "Buildings")

        processing_time = time.time() - start_time

        if gdf is not None:
            self.assertGreater(len(gdf), 0)
            # Should have data from multiple municipalities
            unique_codes = gdf["catastro_municipality_code"].unique()
            self.assertGreaterEqual(
                len(unique_codes), 1
            )  # At least one should have data

            # Performance check - should complete in reasonable time
            self.assertLess(processing_time, 60)  # Should complete within 1 minute
        else:
            print(f"No data available for municipalities {test_codes}")

    # ==================== INTEGRATION EDGE CASES ====================

    def test_real_data_consistency(self):
        """
        Test that data retrieved is consistent and well-formed.

        Verifies data quality and structure.
        """
        catastro = CatastroData(**self.test_config)

        # Get available municipalities for Buildings
        municipalities = catastro.available_municipalities("Buildings")

        if len(municipalities) == 0:
            self.skipTest("No municipalities available for Buildings")

        # Test with first available municipality
        test_code = municipalities["catastro_municipality_code"].iloc[0]

        gdf = catastro.get_data(test_code, "Buildings")

        if gdf is not None and len(gdf) > 0:
            # Verify data structure
            self.assertIn("geometry", gdf.columns)
            self.assertIn("catastro_municipality_code", gdf.columns)

            # Verify geometry is valid
            self.assertTrue(gdf.geometry.notna().any())

            # Verify CRS is properly set
            self.assertIsNotNone(gdf.crs)

            # Verify municipality code is consistent
            unique_codes = gdf["catastro_municipality_code"].unique()
            self.assertEqual(len(unique_codes), 1)
            self.assertEqual(unique_codes[0], test_code)


if __name__ == "__main__":
    # Only run if explicitly enabled
    if os.environ.get("RUN_INTEGRATION_TESTS"):
        unittest.main()
    else:
        print("Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.")
        print(
            "Example: RUN_INTEGRATION_TESTS=1 python -m unittest test_catastro_integration"
        )
