"""Tests for the main SHAP explainer class."""

import shutil
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from mango_shap.explainer import SHAPExplainer
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Suppress XGBoost deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")


class TestSHAPExplainer(unittest.TestCase):
    """Test cases for SHAPExplainer class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Classification models
        model_1 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )
        model_2 = RandomForestClassifier(random_state=42)
        model_3 = LGBMClassifier(random_state=42)
        cls._model_error = XGBClassifier(random_state=42)

        # Create a synthetic dataset for classification
        X, y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=42
        )

        # Assign names to the features
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        # Split the dataset into train and test sets
        cls.X_train, X_test, cls.y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models
        cls._classification_models = [model_1, model_2, model_3]
        for model in cls._classification_models:
            model.fit(cls.X_train, cls.y_train)

        # Train XGBoost model with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cls._model_error.fit(cls.X_train, cls.y_train)

        # Regression models
        model_4 = Pipeline([("classifier", RandomForestRegressor(random_state=42))])
        model_5 = RandomForestRegressor(random_state=42)
        model_6 = LGBMRegressor(random_state=42)

        # Create a synthetic dataset for regression
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)

        # Assign names to the features
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        # X Reset index and rename column with "metadata_index" name
        X.reset_index(drop=False, inplace=True)
        X.rename(columns={"index": "metadata_index"}, inplace=True)

        # Split the dataset into train and test sets
        cls.X_train_reg, X_test_reg, cls.y_train_reg, y_test_reg = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models
        cls._regression_models = [model_4, model_5, model_6]
        for model in cls._regression_models:
            model.fit(cls.X_train_reg, cls.y_train_reg)

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42
        )
        self.X = X
        self.y = y

        # Create a trained model for testing
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove any test directories that might have been created
        test_dirs = ["shap_module", "test_shap"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_explainer_initialization(self):
        """Test SHAP explainer initialization."""
        explainer = SHAPExplainer(
            model=self.model,
            data=self.X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        self.assertEqual(explainer._model, self.model)
        self.assertEqual(explainer._x_transformed.shape, (50, 5))
        self.assertIsNotNone(explainer._explainer)

    def test_explain_method(self):
        """Test SHAP values generation."""
        explainer = SHAPExplainer(
            model=self.model,
            data=self.X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        shap_values = explainer.explain(self.X[50:60])

        # For binary classification, SHAP returns values for both classes (10, 5, 2)
        # For regression, it would be (10, 5)
        if explainer._problem_type == "binary_classification":
            self.assertEqual(shap_values.shape, (10, 5, 2))
        else:
            self.assertEqual(shap_values.shape, (10, 5))

        self.assertFalse(np.isnan(shap_values).any())
        self.assertFalse(np.isinf(shap_values).any())

    def test_model_type_detection(self):
        """Test automatic model type detection."""
        explainer = SHAPExplainer(
            model=self.model,
            data=self.X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        # Should detect tree-based model
        self.assertTrue(hasattr(explainer._explainer, "shap_values"))

    def test_feature_names_handling(self):
        """Test feature names handling."""
        explainer = SHAPExplainer(
            model=self.model,
            data=self.X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        self.assertEqual(len(explainer._feature_names), self.X.shape[1])

    def test_invalid_model(self):
        """Test error handling for invalid model."""
        with self.assertRaises(ValueError):
            SHAPExplainer(
                model=None,
                data=self.X[:50],
                problem_type="binary_classification",
                model_name="test_model",
            )

    def test_invalid_data(self):
        """Test error handling for invalid data."""
        with self.assertRaises(ValueError):
            SHAPExplainer(
                model=self.model,
                data=None,
                problem_type="binary_classification",
                model_name="test_model",
            )

    def test_empty_data(self):
        """Test error handling for empty data."""
        empty_data = np.array([]).reshape(0, 5)

        with self.assertRaises(ValueError):
            SHAPExplainer(
                model=self.model,
                data=empty_data,
                problem_type="binary_classification",
                model_name="test_model",
            )

    def test_problem_type_validation(self):
        """Test problem type validation."""
        with self.assertRaises(ValueError):
            SHAPExplainer(
                model=self.model,
                data=self.X[:50],
                problem_type="invalid_type",
                model_name="test_model",
            )

    def test_metadata_handling(self):
        """Test metadata handling."""
        # Add metadata column
        X_with_metadata = pd.DataFrame(
            self.X, columns=[f"feature_{i}" for i in range(self.X.shape[1])]
        )
        X_with_metadata["id"] = range(len(X_with_metadata))

        explainer = SHAPExplainer(
            model=self.model,
            data=X_with_metadata,
            problem_type="binary_classification",
            model_name="test_model",
            metadata=["id"],
        )

        # Should exclude metadata from SHAP calculations
        self.assertEqual(explainer._x_transformed.shape[1], self.X.shape[1])
        self.assertNotIn("id", explainer._feature_names)

    def test_binary_classification_with_multiple_models(self):
        """Test binary classification with multiple model types."""
        for model in self._classification_models:
            with self.subTest(model=type(model).__name__):
                explainer = SHAPExplainer(
                    model=model,
                    data=self.X_train,
                    problem_type="binary_classification",
                    model_name=type(model).__name__,
                )

                # Test basic properties
                self.assertEqual(explainer._problem_type, "binary_classification")
                self.assertEqual(explainer._model_name, type(model).__name__)
                self.assertEqual(explainer._data.shape, self.X_train.shape)
                self.assertIsNotNone(explainer._explainer)

                # Test SHAP values generation
                shap_values = explainer.explain(self.X_train[:10])
                self.assertIsNotNone(shap_values)
                self.assertFalse(np.isnan(shap_values).any())
                self.assertFalse(np.isinf(shap_values).any())

    def test_regression_with_multiple_models(self):
        """Test regression with multiple model types."""
        for model in self._regression_models:
            with self.subTest(model=type(model).__name__):
                explainer = SHAPExplainer(
                    model=model,
                    data=self.X_train_reg,
                    problem_type="regression",
                    model_name=type(model).__name__,
                )

                # Test basic properties
                self.assertEqual(explainer._problem_type, "regression")
                self.assertEqual(explainer._model_name, type(model).__name__)
                self.assertEqual(explainer._data.shape, self.X_train_reg.shape)
                self.assertIsNotNone(explainer._explainer)

                # Test SHAP values generation
                shap_values = explainer.explain(self.X_train_reg[:10])
                self.assertIsNotNone(shap_values)
                self.assertFalse(np.isnan(shap_values).any())
                self.assertFalse(np.isinf(shap_values).any())

    def test_regression_with_metadata(self):
        """Test regression with metadata handling."""
        model = self._regression_models[0]
        explainer = SHAPExplainer(
            model=model,
            data=self.X_train_reg,
            problem_type="regression",
            model_name=type(model).__name__,
            metadata=["metadata_index"],
        )

        # Test metadata handling
        self.assertEqual(explainer._metadata, ["metadata_index"])
        self.assertEqual(explainer._x_transformed.shape[1], 21)
        self.assertNotIn("metadata_index", explainer._feature_names)

    def test_feature_names_with_different_models(self):
        """Test feature names handling with different model types."""
        # Test with RandomForest (should use fallback)
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(self.X_train, self.y_train)

        explainer = SHAPExplainer(
            model=rf_model,
            data=self.X_train,
            problem_type="binary_classification",
            model_name="RandomForest",
        )

        self.assertEqual(len(explainer._feature_names), 20)
        self.assertTrue(
            all(name.startswith("feature_") for name in explainer._feature_names)
        )

    def test_pipeline_model_handling(self):
        """Test handling of sklearn Pipeline models."""
        pipeline_model = self._classification_models[0]

        explainer = SHAPExplainer(
            model=pipeline_model,
            data=self.X_train,
            problem_type="binary_classification",
            model_name="Pipeline",
        )

        # Should work with Pipeline models
        self.assertIsNotNone(explainer._explainer)
        shap_values = explainer.explain(self.X_train[:5])
        self.assertIsNotNone(shap_values)

    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create a larger dataset
        X_large, y_large = make_classification(
            n_samples=5000, n_features=50, n_classes=2, random_state=42
        )
        X_large = pd.DataFrame(X_large, columns=[f"feature_{i}" for i in range(50)])

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_large, y_large)

        explainer = SHAPExplainer(
            model=model,
            data=X_large[:1000],
            problem_type="binary_classification",
            model_name="LargeDataset",
        )

        # Should handle large datasets
        self.assertEqual(explainer._x_transformed.shape, (1000, 50))
        shap_values = explainer.explain(X_large[1000:1010])

        # For binary classification, SHAP returns values for both classes (10, 50, 2)
        # For regression, it would be (10, 50)
        if explainer._problem_type == "binary_classification":
            self.assertEqual(shap_values.shape, (10, 50, 2))
        else:
            self.assertEqual(shap_values.shape, (10, 50))

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single sample using the correct model
        single_sample = self.X_train[:1]
        model_for_test = self._classification_models[
            1
        ]  # Use RandomForest from setUpClass
        explainer = SHAPExplainer(
            model=model_for_test,
            data=single_sample,
            problem_type="binary_classification",
            model_name="SingleSample",
        )

        # Should handle single sample
        self.assertEqual(explainer._x_transformed.shape, (1, 20))

        # Test with single feature (if possible)
        X_single_feature = self.X_train.iloc[:, :1]
        model_single = RandomForestClassifier(n_estimators=5, random_state=42)
        model_single.fit(X_single_feature, self.y_train)

        explainer_single = SHAPExplainer(
            model=model_single,
            data=X_single_feature,
            problem_type="binary_classification",
            model_name="SingleFeature",
        )

        self.assertEqual(explainer_single._x_transformed.shape[1], 1)

    def test_model_type_detection_comprehensive(self):
        """Test model type detection for various model types."""
        # Test RandomForest
        rf_explainer = SHAPExplainer(
            model=self._classification_models[1],  # RandomForest
            data=self.X_train,
            problem_type="binary_classification",
            model_name="RandomForest",
        )
        self.assertIsNotNone(rf_explainer._explainer)

        # Test LightGBM
        lgb_explainer = SHAPExplainer(
            model=self._classification_models[2],  # LightGBM
            data=self.X_train,
            problem_type="binary_classification",
            model_name="LightGBM",
        )
        self.assertIsNotNone(lgb_explainer._explainer)

    def test_data_validation_comprehensive(self):
        """Test comprehensive data validation."""
        # Use a model that matches the data dimensions
        model_for_test = self._classification_models[1]  # RandomForest from setUpClass

        # Test with numpy array
        X_numpy = self.X_train.values
        explainer_numpy = SHAPExplainer(
            model=model_for_test,
            data=X_numpy,
            problem_type="binary_classification",
            model_name="NumpyData",
        )
        self.assertIsNotNone(explainer_numpy._explainer)

        # Test with pandas DataFrame
        explainer_pandas = SHAPExplainer(
            model=model_for_test,
            data=self.X_train,
            problem_type="binary_classification",
            model_name="PandasData",
        )
        self.assertIsNotNone(explainer_pandas._explainer)

    def test_problem_type_validation_comprehensive(self):
        """Test comprehensive problem type validation."""
        # Use a model that matches the data dimensions
        model_for_test = self._classification_models[1]  # RandomForest from setUpClass

        valid_types = [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]

        for problem_type in valid_types:
            with self.subTest(problem_type=problem_type):
                explainer = SHAPExplainer(
                    model=model_for_test,
                    data=self.X_train,
                    problem_type=problem_type,
                    model_name="TestModel",
                )
                self.assertEqual(explainer._problem_type, problem_type)

        # Test invalid problem type
        with self.assertRaises(ValueError):
            SHAPExplainer(
                model=model_for_test,
                data=self.X_train,
                problem_type="invalid_type",
                model_name="TestModel",
            )

    def test_visualization_methods(self):
        """Test visualization methods."""
        # Test summary plot
        explainer = SHAPExplainer(
            self._classification_models[1], self.X_train, problem_type="classification"
        )

        # Test that visualization methods don't raise errors
        try:
            explainer.summary_plot(show=False)
            explainer.bar_summary_plot(show=False)
            explainer.waterfall_plot(instance_idx=0, show=False)
            explainer.partial_dependence_plot(feature_idx=0, show=False)
        except Exception as e:
            self.fail(f"Visualization methods failed: {e}")

    def test_analysis_and_export_methods(self):
        """Test analysis and export methods."""
        explainer = SHAPExplainer(
            self._classification_models[1], self.X_train, problem_type="classification"
        )

        # Test make_shap_analysis
        try:
            analysis_results = explainer.make_shap_analysis()
            self.assertIsInstance(analysis_results, dict)
            self.assertIn("summary_stats", analysis_results)
            self.assertIn("feature_importance", analysis_results)
        except Exception as e:
            self.fail(f"make_shap_analysis failed: {e}")

        # Test export_explanations
        try:
            explainer.export_explanations("test_export", format="csv")
            # Check if file was created
            self.assertTrue(Path("test_export.csv").exists())
            # Clean up
            Path("test_export.csv").unlink()
        except Exception as e:
            self.fail(f"export_explanations failed: {e}")

    def test_get_sample_by_shap_value(self):
        """Test get_sample_by_shap_value method."""
        explainer = SHAPExplainer(
            self._classification_models[1], self.X_train, problem_type="classification"
        )

        # Test getting sample with highest SHAP value for first feature
        try:
            sample = explainer.get_sample_by_shap_value(feature_idx=0, highest=True)
            self.assertIsInstance(sample, dict)
            self.assertIn("instance_idx", sample)
            self.assertIn("shap_value", sample)
            self.assertIn("feature_value", sample)
        except Exception as e:
            self.fail(f"get_sample_by_shap_value failed: {e}")

    def test_different_explainer_types(self):
        """Test different explainer types."""
        # Test with linear model (should use LinearExplainer)
        from sklearn.linear_model import LinearRegression

        linear_model = LinearRegression()
        linear_model.fit(self.X_train, self.y_train)

        explainer = SHAPExplainer(linear_model, self.X_train, problem_type="regression")

        # Should detect as linear model
        self.assertEqual(explainer._model_type, "linear")

        # Test SHAP values generation
        shap_values = explainer.explain(self.X_test[:5])
        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (5, self.X_train.shape[1]))

    def test_property_setters_and_getters(self):
        """Test property setters and getters."""
        explainer = SHAPExplainer(
            self._classification_models[1], self.X_train, problem_type="classification"
        )

        # Test problem_type property
        self.assertEqual(explainer.problem_type, "classification")
        explainer.problem_type = "regression"
        self.assertEqual(explainer.problem_type, "regression")

        # Test model_name property
        explainer.model_name = "test_model"
        self.assertEqual(explainer.model_name, "test_model")

        # Test shap_folder property
        explainer.shap_folder = "test_folder"
        self.assertEqual(explainer.shap_folder, "test_folder")

    def test_data_property_with_metadata(self):
        """Test data property with metadata columns."""
        # Create data with metadata
        data_with_metadata = self.X_train.copy()
        data_with_metadata["metadata_col"] = np.random.random(len(data_with_metadata))

        explainer = SHAPExplainer(
            self._classification_models[1],
            data_with_metadata,
            problem_type="classification",
        )

        # Test that metadata is handled correctly
        self.assertIn("metadata_col", explainer.metadata)
        self.assertNotIn("metadata_col", explainer._feature_names)

    def test_multi_class_classification(self):
        """Test multi-class classification."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        # Create multi-class dataset
        X_multi, y_multi = make_classification(
            n_samples=200, n_features=5, n_classes=3, random_state=42
        )
        X_multi = pd.DataFrame(X_multi, columns=[f"feature_{i}" for i in range(5)])

        # Train model
        model_multi = RandomForestClassifier(random_state=42)
        model_multi.fit(X_multi, y_multi)

        # Create explainer
        explainer = SHAPExplainer(model_multi, X_multi, problem_type="classification")

        # Test SHAP values for multi-class
        shap_values = explainer.explain(X_multi[:10])
        self.assertEqual(shap_values.shape, (10, 5, 3))  # (samples, features, classes)

    def test_custom_feature_names(self):
        """Test custom feature names."""
        custom_names = [f"custom_feature_{i}" for i in range(self.X_train.shape[1])]

        explainer = SHAPExplainer(
            self._classification_models[1],
            self.X_train,
            problem_type="classification",
            feature_names=custom_names,
        )

        self.assertEqual(explainer._feature_names, custom_names)

    def test_save_fig_method(self):
        """Test _save_fig method."""
        explainer = SHAPExplainer(
            self._classification_models[1], self.X_train, problem_type="classification"
        )

        # Create a simple plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        # Test saving
        try:
            explainer._save_fig(fig, "test_plot.png")
            self.assertTrue(Path("test_plot.png").exists())
            # Clean up
            Path("test_plot.png").unlink()
        except Exception as e:
            self.fail(f"_save_fig failed: {e}")
        finally:
            plt.close(fig)

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""

        # Test with model that has no feature_names_in_
        class CustomModel:
            def __init__(self):
                self.n_features_in_ = 5

            def predict(self, X):
                return np.random.random(X.shape[0])

        custom_model = CustomModel()
        custom_model.fit = lambda X, y: None

        # This should not raise an error
        try:
            explainer = SHAPExplainer(
                custom_model, self.X_train, problem_type="regression"
            )
            self.assertIsNotNone(explainer._feature_names)
        except Exception as e:
            self.fail(f"Custom model handling failed: {e}")

    def test_large_feature_count(self):
        """Test with large number of features."""
        # Create dataset with many features
        X_large, y_large = make_classification(
            n_samples=100, n_features=100, n_classes=2, random_state=42
        )
        X_large = pd.DataFrame(X_large, columns=[f"feature_{i}" for i in range(100)])

        # Train model
        model_large = RandomForestClassifier(random_state=42, n_estimators=10)
        model_large.fit(X_large, y_large)

        # Create explainer
        explainer = SHAPExplainer(model_large, X_large, problem_type="classification")

        # Test SHAP values generation
        shap_values = explainer.explain(X_large[:5])
        self.assertEqual(shap_values.shape, (5, 100, 2))  # (samples, features, classes)

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create large dataset
        X_large, y_large = make_classification(
            n_samples=1000, n_features=50, n_classes=2, random_state=42
        )
        X_large = pd.DataFrame(X_large, columns=[f"feature_{i}" for i in range(50)])

        # Train model
        model_large = RandomForestClassifier(random_state=42, n_estimators=10)
        model_large.fit(X_large, y_large)

        # Create explainer
        explainer = SHAPExplainer(model_large, X_large, problem_type="classification")

        # Test that we can handle large datasets without memory issues
        try:
            shap_values = explainer.explain(X_large[:100])  # Test with subset
            self.assertEqual(shap_values.shape, (100, 50, 2))
        except MemoryError:
            self.fail("Memory error with large dataset")
        except Exception as e:
            self.fail(f"Large dataset handling failed: {e}")


if __name__ == "__main__":
    unittest.main()
