"""Tests for the main SHAP explainer class."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from mango_shap.explainer import SHAPExplainer


class TestSHAPExplainer:
    """Test cases for SHAPExplainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42
        )
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_explainer_initialization(self, trained_model, sample_data):
        """Test SHAP explainer initialization."""
        X, _ = sample_data
        explainer = SHAPExplainer(
            model=trained_model,
            data=X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        assert explainer._model == trained_model
        assert explainer._x_transformed.shape == (50, 5)
        assert explainer._explainer is not None

    def test_explain_method(self, trained_model, sample_data):
        """Test SHAP values generation."""
        X, _ = sample_data
        explainer = SHAPExplainer(
            model=trained_model,
            data=X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        shap_values = explainer.explain(X[50:60])

        assert shap_values.shape == (10, 5)
        assert not np.isnan(shap_values).any()
        assert not np.isinf(shap_values).any()

    def test_model_type_detection(self, trained_model, sample_data):
        """Test automatic model type detection."""
        X, _ = sample_data
        explainer = SHAPExplainer(
            model=trained_model,
            data=X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        # Should detect tree-based model
        assert hasattr(explainer._explainer, "shap_values")

    def test_feature_names_handling(self, trained_model, sample_data):
        """Test feature names handling."""
        X, _ = sample_data
        explainer = SHAPExplainer(
            model=trained_model,
            data=X[:50],
            problem_type="binary_classification",
            model_name="test_model",
        )

        assert len(explainer._feature_names) == X.shape[1]

    def test_invalid_model(self, sample_data):
        """Test error handling for invalid model."""
        X, _ = sample_data

        with pytest.raises(ValueError):
            SHAPExplainer(
                model=None,
                data=X[:50],
                problem_type="binary_classification",
                model_name="test_model",
            )

    def test_invalid_data(self, trained_model):
        """Test error handling for invalid data."""
        with pytest.raises(ValueError):
            SHAPExplainer(
                model=trained_model,
                data=None,
                problem_type="binary_classification",
                model_name="test_model",
            )

    def test_empty_data(self, trained_model):
        """Test error handling for empty data."""
        empty_data = np.array([]).reshape(0, 5)

        with pytest.raises(ValueError):
            SHAPExplainer(
                model=trained_model,
                data=empty_data,
                problem_type="binary_classification",
                model_name="test_model",
            )

    def test_problem_type_validation(self, trained_model, sample_data):
        """Test problem type validation."""
        X, _ = sample_data

        with pytest.raises(ValueError):
            SHAPExplainer(
                model=trained_model,
                data=X[:50],
                problem_type="invalid_type",
                model_name="test_model",
            )

    def test_metadata_handling(self, trained_model, sample_data):
        """Test metadata handling."""
        X, _ = sample_data
        # Add metadata column
        X_with_metadata = pd.DataFrame(
            X, columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        X_with_metadata["id"] = range(len(X_with_metadata))

        explainer = SHAPExplainer(
            model=trained_model,
            data=X_with_metadata,
            problem_type="binary_classification",
            model_name="test_model",
            metadata=["id"],
        )

        # Should exclude metadata from SHAP calculations
        assert explainer._x_transformed.shape[1] == X.shape[1]
        assert "id" not in explainer._feature_names
