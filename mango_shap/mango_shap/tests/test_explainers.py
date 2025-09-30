"""Tests for explainer modules."""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from mango_shap.explainers import (
    TreeExplainer,
    LinearExplainer,
    DeepExplainer,
    KernelExplainer,
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestTreeExplainer(unittest.TestCase):
    """Test cases for TreeExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        # Train model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X, y)

        self.background_data = X.iloc[:20]
        self.test_data = X.iloc[20:30]

    def test_initialization(self):
        """Test TreeExplainer initialization."""
        explainer = TreeExplainer(self.model, self.background_data)
        self.assertIsNotNone(explainer.explainer)

    def test_shap_values(self):
        """Test SHAP values calculation."""
        explainer = TreeExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 5, 2))  # (samples, features, classes)

    def test_call_method(self):
        """Test __call__ method."""
        explainer = TreeExplainer(self.model, self.background_data)
        shap_values = explainer(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 5, 2))


class TestLinearExplainer(unittest.TestCase):
    """Test cases for LinearExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        # Train model
        self.model = LinearRegression()
        self.model.fit(X, y)

        self.background_data = X.iloc[:20]
        self.test_data = X.iloc[20:30]

    def test_initialization(self):
        """Test LinearExplainer initialization."""
        explainer = LinearExplainer(self.model, self.background_data)
        self.assertIsNotNone(explainer.explainer)

    def test_shap_values(self):
        """Test SHAP values calculation."""
        explainer = LinearExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 5))  # (samples, features)

    def test_call_method(self):
        """Test __call__ method."""
        explainer = LinearExplainer(self.model, self.background_data)
        shap_values = explainer(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 5))


class TestDeepExplainer(unittest.TestCase):
    """Test cases for DeepExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        # Create a simple neural network model
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense

            self.model = Sequential(
                [
                    Dense(10, activation="relu", input_shape=(5,)),
                    Dense(1, activation="sigmoid"),
                ]
            )
            self.model.compile(optimizer="adam", loss="binary_crossentropy")

            # Train model
            self.model.fit(X.values, y, epochs=1, verbose=0)

            self.background_data = X.iloc[:20]
            self.test_data = X.iloc[20:30]
            self.tensorflow_available = True

        except ImportError:
            # Skip tests if TensorFlow is not available
            self.tensorflow_available = False

    def test_initialization(self):
        """Test DeepExplainer initialization."""
        if not self.tensorflow_available:
            self.skipTest("TensorFlow not available")

        explainer = DeepExplainer(self.model, self.background_data)
        self.assertIsNotNone(explainer.explainer)

    def test_shap_values(self):
        """Test SHAP values calculation."""
        if not self.tensorflow_available:
            self.skipTest("TensorFlow not available")

        explainer = DeepExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 5))  # (samples, features)

    def test_call_method(self):
        """Test __call__ method."""
        if not self.tensorflow_available:
            self.skipTest("TensorFlow not available")

        explainer = DeepExplainer(self.model, self.background_data)
        shap_values = explainer(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 5))


class TestKernelExplainer(unittest.TestCase):
    """Test cases for KernelExplainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])

        # Train model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X, y)

        self.background_data = X.iloc[:20]
        self.test_data = X.iloc[20:30]

    def test_initialization(self):
        """Test KernelExplainer initialization."""
        explainer = KernelExplainer(self.model, self.background_data)
        self.assertIsNotNone(explainer.explainer)

    def test_shap_values(self):
        """Test SHAP values calculation."""
        explainer = KernelExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        # Shape depends on the model output

    def test_shap_values_with_max_evals(self):
        """Test SHAP values calculation with max_evals parameter."""
        explainer = KernelExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(self.test_data, max_evals=50)

        self.assertIsInstance(shap_values, np.ndarray)

    def test_call_method(self):
        """Test __call__ method."""
        explainer = KernelExplainer(self.model, self.background_data)
        shap_values = explainer(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)

    def test_call_method_with_max_evals(self):
        """Test __call__ method with max_evals parameter."""
        explainer = KernelExplainer(self.model, self.background_data)
        shap_values = explainer(self.test_data, max_evals=50)

        self.assertIsInstance(shap_values, np.ndarray)


class TestExplainerIntegration(unittest.TestCase):
    """Integration tests for explainers."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train different models
        self.tree_model = RandomForestClassifier(random_state=42)
        self.tree_model.fit(X_train, y_train)

        self.linear_model = LogisticRegression(random_state=42)
        self.linear_model.fit(X_train, y_train)

        self.background_data = X_train.iloc[:50]
        self.test_data = X_test.iloc[:10]

    def test_tree_explainer_integration(self):
        """Test TreeExplainer integration."""
        explainer = TreeExplainer(self.tree_model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 10, 2))

    def test_linear_explainer_integration(self):
        """Test LinearExplainer integration."""
        explainer = LinearExplainer(self.linear_model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)
        self.assertEqual(shap_values.shape, (10, 10, 2))

    def test_kernel_explainer_integration(self):
        """Test KernelExplainer integration."""
        explainer = KernelExplainer(self.tree_model, self.background_data)
        shap_values = explainer.shap_values(self.test_data)

        self.assertIsInstance(shap_values, np.ndarray)

    def test_explainer_consistency(self):
        """Test that different explainers produce consistent results."""
        # Test with same model and data
        tree_explainer = TreeExplainer(self.tree_model, self.background_data)
        kernel_explainer = KernelExplainer(self.tree_model, self.background_data)

        tree_shap = tree_explainer.shap_values(self.test_data[:5])
        kernel_shap = kernel_explainer.shap_values(self.test_data[:5])

        # Both should produce arrays (exact values may differ due to different algorithms)
        self.assertIsInstance(tree_shap, np.ndarray)
        self.assertIsInstance(kernel_shap, np.ndarray)

        # Shapes should be compatible
        self.assertEqual(len(tree_shap.shape), len(kernel_shap.shape))


if __name__ == "__main__":
    unittest.main()
