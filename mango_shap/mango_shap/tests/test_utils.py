"""Tests for utility modules."""

import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from mango_shap.utils import DataProcessor, ExportUtils, InputValidator


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()

        # Create test data
        self.test_dataframe = pd.DataFrame(
            {
                "feature_1": [1, 2, 3, 4, 5],
                "feature_2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "categorical": ["A", "B", "A", "C", "B"],
                "missing": [1, 2, np.nan, 4, 5],
            }
        )

        self.test_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_process_dataframe_drop_missing(self):
        """Test processing DataFrame with drop missing strategy."""
        processed = self.processor.process_data(
            self.test_dataframe, handle_missing="drop"
        )

        # Should drop rows with missing values
        self.assertEqual(len(processed), 4)
        self.assertFalse(processed.isnull().any().any())

    def test_process_dataframe_fill_missing(self):
        """Test processing DataFrame with fill missing strategy."""
        processed = self.processor.process_data(
            self.test_dataframe, handle_missing="fill"
        )

        # Should fill missing values
        # No rows dropped
        self.assertEqual(len(processed), 5)
        self.assertFalse(processed.isnull().any().any())

    def test_process_dataframe_error_missing(self):
        """Test processing DataFrame with error missing strategy."""
        with self.assertRaises(ValueError):
            self.processor.process_data(self.test_dataframe, handle_missing="error")

    def test_process_array(self):
        """Test processing numpy array."""
        processed = self.processor.process_data(self.test_array, handle_missing="fill")

        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, self.test_array.shape)

    def test_encode_categorical(self):
        """Test categorical encoding."""
        processed = self.processor._encode_categorical(self.test_dataframe)

        # Categorical column should be encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(processed["categorical"]))

    def test_get_feature_names_dataframe(self):
        """Test getting feature names from DataFrame."""
        names = DataProcessor.get_feature_names(self.test_dataframe)
        expected = ["feature_1", "feature_2", "categorical", "missing"]
        self.assertEqual(names, expected)

    def test_get_feature_names_array(self):
        """Test getting feature names from array."""
        names = DataProcessor.get_feature_names(self.test_array)
        expected = ["feature_0", "feature_1", "feature_2"]
        self.assertEqual(names, expected)

    def test_get_feature_names_custom(self):
        """Test getting custom feature names."""
        custom_names = ["custom_1", "custom_2", "custom_3"]
        names = DataProcessor.get_feature_names(self.test_array, custom_names)
        self.assertEqual(names, custom_names)

    def test_unsupported_data_type(self):
        """Test error handling for unsupported data types."""
        with self.assertRaises(ValueError):
            self.processor.process_data("invalid_data")


class TestExportUtils(unittest.TestCase):
    """Test cases for ExportUtils class."""

    def setUp(self):
        """Set up test fixtures."""
        self.exporter = ExportUtils()

        # Create test data
        self.shap_values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

        self.data = pd.DataFrame(
            {"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "feature_3": [7, 8, 9]}
        )

        self.feature_names = ["feature_1", "feature_2", "feature_3"]

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_csv(self):
        """Test CSV export."""
        output_path = Path(self.temp_dir) / "test_export"

        self.exporter.export(
            self.shap_values, self.data, self.feature_names, str(output_path), "csv"
        )

        # Check if file was created
        csv_file = output_path.with_suffix(".csv")
        self.assertTrue(csv_file.exists())

        # Check file content
        df = pd.read_csv(csv_file)
        # 3 instances
        self.assertEqual(len(df), 3)

    def test_export_json(self):
        """Test JSON export."""
        output_path = Path(self.temp_dir) / "test_export"

        self.exporter.export(
            self.shap_values, self.data, self.feature_names, str(output_path), "json"
        )

        # Check if file was created
        json_file = output_path.with_suffix(".json")
        self.assertTrue(json_file.exists())

    def test_export_html(self):
        """Test HTML export."""
        output_path = Path(self.temp_dir) / "test_export"

        self.exporter.export(
            self.shap_values, self.data, self.feature_names, str(output_path), "html"
        )

        # Check if file was created
        html_file = output_path.with_suffix(".html")
        self.assertTrue(html_file.exists())

    def test_export_unsupported_format(self):
        """Test error handling for unsupported format."""
        with self.assertRaises(ValueError):
            self.exporter.export(
                self.shap_values, self.data, self.feature_names, "test", "unsupported"
            )

    def test_export_with_numpy_data(self):
        """Test export with numpy array data."""
        output_path = Path(self.temp_dir) / "test_export"

        self.exporter.export(
            self.shap_values,
            # Convert to numpy array
            self.data.values,
            self.feature_names,
            str(output_path),
            "csv",
        )

        csv_file = output_path.with_suffix(".csv")
        self.assertTrue(csv_file.exists())

    def test_export_without_feature_names(self):
        """Test export without feature names."""
        output_path = Path(self.temp_dir) / "test_export"

        self.exporter.export(
            self.shap_values,
            self.data,
            # No feature names
            None,
            str(output_path),
            "csv",
        )

        csv_file = output_path.with_suffix(".csv")
        self.assertTrue(csv_file.exists())


class TestInputValidator(unittest.TestCase):
    """Test cases for InputValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

        # Create test data
        self.valid_data = pd.DataFrame(
            {"feature_1": [1, 2, 3, 4, 5], "feature_2": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )

        self.valid_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Create a simple model for testing
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        self.model.fit(self.valid_data, [1, 2, 3, 4, 5])

    def test_validate_model_success(self):
        """Test successful model validation."""
        # Should not raise any exception
        self.validator.validate_model(self.model)

    def test_validate_model_none(self):
        """Test model validation with None model."""
        with self.assertRaises(ValueError):
            self.validator.validate_model(None)

    def test_validate_model_no_predict(self):
        """Test model validation with model without predict method."""

        class BadModel:
            pass

        with self.assertRaises(ValueError):
            self.validator.validate_model(BadModel())

    def test_validate_data_success(self):
        """Test successful data validation."""
        # Should not raise any exception
        self.validator.validate_data(self.valid_data)
        self.validator.validate_data(self.valid_array)

    def test_validate_data_none(self):
        """Test data validation with None data."""
        with self.assertRaises(ValueError):
            self.validator.validate_data(None)

    def test_validate_data_wrong_type(self):
        """Test data validation with wrong data type."""
        with self.assertRaises(ValueError):
            self.validator.validate_data("invalid_data")

    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.validator.validate_data(empty_data)

    def test_validate_data_no_features(self):
        """Test data validation with no features."""
        no_features = pd.DataFrame([[], [], []])
        with self.assertRaises(ValueError):
            self.validator.validate_data(no_features)

    def test_validate_data_infinite_values(self):
        """Test data validation with infinite values."""
        data_with_inf = self.valid_data.copy()
        data_with_inf.iloc[0, 0] = np.inf

        with self.assertRaises(ValueError):
            self.validator.validate_data(data_with_inf)

    def test_validate_background_data_success(self):
        """Test successful background data validation."""
        # Should not raise any exception
        self.validator.validate_background_data(self.valid_data)

    def test_validate_background_data_insufficient_samples(self):
        """Test background data validation with insufficient samples."""
        single_sample = self.valid_data.iloc[:1]
        with self.assertRaises(ValueError):
            self.validator.validate_background_data(single_sample)

    def test_validate_feature_names_success(self):
        """Test successful feature names validation."""
        feature_names = ["feature_1", "feature_2"]
        data_shape = (10, 2)

        # Should not raise any exception
        InputValidator.validate_feature_names(feature_names, data_shape)

    def test_validate_feature_names_none(self):
        """Test feature names validation with None."""
        data_shape = (10, 2)

        # Should not raise any exception
        InputValidator.validate_feature_names(None, data_shape)

    def test_validate_feature_names_wrong_type(self):
        """Test feature names validation with wrong type."""
        data_shape = (10, 2)

        with self.assertRaises(ValueError):
            InputValidator.validate_feature_names("not_a_list", data_shape)

    def test_validate_feature_names_wrong_length(self):
        """Test feature names validation with wrong length."""
        # 3 names
        feature_names = ["feature_1", "feature_2", "feature_3"]
        # 2 features
        data_shape = (10, 2)

        with self.assertRaises(ValueError):
            InputValidator.validate_feature_names(feature_names, data_shape)

    def test_validate_feature_names_duplicates(self):
        """Test feature names validation with duplicates."""
        # Duplicates
        feature_names = ["feature_1", "feature_1"]
        data_shape = (10, 2)

        with self.assertRaises(ValueError):
            InputValidator.validate_feature_names(feature_names, data_shape)

    def test_validate_shap_values_success(self):
        """Test successful SHAP values validation."""
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        # Should not raise any exception
        self.validator.validate_shap_values(shap_values, self.valid_data)

    def test_validate_shap_values_none(self):
        """Test SHAP values validation with None."""
        with self.assertRaises(ValueError):
            self.validator.validate_shap_values(None, self.valid_data)

    def test_validate_shap_values_wrong_type(self):
        """Test SHAP values validation with wrong type."""
        with self.assertRaises(ValueError):
            self.validator.validate_shap_values("not_an_array", self.valid_data)

    def test_validate_shap_values_wrong_dimensions(self):
        """Test SHAP values validation with wrong dimensions."""
        # 1D instead of 2D
        shap_values = np.array([0.1, 0.2, 0.3])

        with self.assertRaises(ValueError):
            self.validator.validate_shap_values(shap_values, self.valid_data)

    def test_validate_shap_values_wrong_shape(self):
        """Test SHAP values validation with wrong shape."""
        shap_values = np.array(
            # 3 features instead of 2
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        with self.assertRaises(ValueError):
            self.validator.validate_shap_values(shap_values, self.valid_data)

    def test_validate_shap_values_infinite_values(self):
        """Test SHAP values validation with infinite values."""
        shap_values = np.array(
            # Infinite value
            [[0.1, 0.2], [np.inf, 0.4], [0.5, 0.6]]
        )

        with self.assertRaises(ValueError):
            self.validator.validate_shap_values(shap_values, self.valid_data)


if __name__ == "__main__":
    unittest.main()
