"""Tests for the main SHAP explainer class."""

import unittest

import numpy as np
import pandas as pd
from mango_shap.explainer import SHAPExplainer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


class TestSHAPExplainer(unittest.TestCase):
    """Test cases for SHAPExplainer class."""

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


if __name__ == "__main__":
    unittest.main()
