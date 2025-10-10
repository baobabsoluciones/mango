"""Tests for visualizer modules."""

import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from mango_shap.visualizers import (
    DependencePlot,
    ForcePlot,
    SummaryPlot,
    WaterfallPlot,
    BarPlot,
    BeeswarmPlot,
    DecisionPlot,
)


class TestDependencePlot(unittest.TestCase):
    """Test cases for DependencePlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = DependencePlot()

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

    def test_plot_basic(self):
        """Test basic dependence plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, feature_idx=0, show=False)
        except Exception as e:
            self.fail(f"Dependence plot creation failed: {e}")

    def test_plot_with_interaction(self):
        """Test dependence plot with interaction feature."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_idx=0,
                interaction_feature=1,
                show=False,
            )
        except Exception as e:
            self.fail(f"Dependence plot with interaction failed: {e}")

    def test_plot_with_feature_names(self):
        """Test dependence plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_idx=0,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Dependence plot with feature names failed: {e}")

    def test_plot_save(self):
        """Test dependence plot saving."""
        output_path = Path(self.temp_dir) / "dependence_plot.png"

        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_idx=0,
                save_path=str(output_path),
                show=False,
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Dependence plot saving failed: {e}")

    def test_plot_multi_class(self):
        """Test dependence plot with multi-class SHAP values."""
        # Create multi-class SHAP values with same number of features as data
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot(multi_class_shap, self.data, feature_idx=0, show=False)
        except Exception as e:
            self.fail(f"Multi-class dependence plot failed: {e}")


class TestForcePlot(unittest.TestCase):
    """Test cases for ForcePlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = ForcePlot()

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

    def test_plot_basic(self):
        """Test basic force plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, instance_idx=0, show=False)
        except Exception as e:
            self.fail(f"Force plot creation failed: {e}")

    def test_plot_different_instance(self):
        """Test force plot with different instance."""
        try:
            self.plotter.plot(self.shap_values, self.data, instance_idx=1, show=False)
        except Exception as e:
            self.fail(f"Force plot with different instance failed: {e}")

    def test_plot_with_feature_names(self):
        """Test force plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                instance_idx=0,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Force plot with feature names failed: {e}")

    def test_plot_save(self):
        """Test force plot saving."""
        output_path = Path(self.temp_dir) / "force_plot.png"

        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                instance_idx=0,
                save_path=str(output_path),
                show=False,
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Force plot saving failed: {e}")

    def test_plot_multi_class(self):
        """Test force plot with multi-class SHAP values."""
        # Create multi-class SHAP values with same number of features as data
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot(multi_class_shap, self.data, instance_idx=0, show=False)
        except Exception as e:
            self.fail(f"Multi-class force plot failed: {e}")


class TestSummaryPlot(unittest.TestCase):
    """Test cases for SummaryPlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = SummaryPlot()

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

    def test_plot_basic(self):
        """Test basic summary plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, show=False)
        except Exception as e:
            self.fail(f"Summary plot creation failed: {e}")

    def test_plot_with_feature_names(self):
        """Test summary plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Summary plot with feature names failed: {e}")

    def test_plot_with_max_display(self):
        """Test summary plot with max_display limit."""
        try:
            self.plotter.plot(self.shap_values, self.data, max_display=2, show=False)
        except Exception as e:
            self.fail(f"Summary plot with max_display failed: {e}")

    def test_plot_save(self):
        """Test summary plot saving."""
        output_path = Path(self.temp_dir) / "summary_plot.png"

        try:
            self.plotter.plot(
                self.shap_values, self.data, save_path=str(output_path), show=False
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Summary plot saving failed: {e}")

    def test_plot_multi_class(self):
        """Test summary plot with multi-class SHAP values."""
        # Create multi-class SHAP values with same number of features as data
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot(multi_class_shap, self.data, show=False)
        except Exception as e:
            self.fail(f"Multi-class summary plot failed: {e}")


class TestWaterfallPlot(unittest.TestCase):
    """Test cases for WaterfallPlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = WaterfallPlot()

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

    def test_plot_basic(self):
        """Test basic waterfall plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, instance_idx=0, show=False)
        except Exception as e:
            self.fail(f"Waterfall plot creation failed: {e}")

    def test_plot_different_instance(self):
        """Test waterfall plot with different instance."""
        try:
            self.plotter.plot(self.shap_values, self.data, instance_idx=1, show=False)
        except Exception as e:
            self.fail(f"Waterfall plot with different instance failed: {e}")

    def test_plot_with_feature_names(self):
        """Test waterfall plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                instance_idx=0,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Waterfall plot with feature names failed: {e}")

    def test_plot_save(self):
        """Test waterfall plot saving."""
        output_path = Path(self.temp_dir) / "waterfall_plot.png"

        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                instance_idx=0,
                save_path=str(output_path),
                show=False,
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Waterfall plot saving failed: {e}")

    def test_plot_multi_class(self):
        """Test waterfall plot with multi-class SHAP values."""
        # Create multi-class SHAP values with same number of features as data
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot(multi_class_shap, self.data, instance_idx=0, show=False)
        except Exception as e:
            self.fail(f"Multi-class waterfall plot failed: {e}")


class TestBarPlot(unittest.TestCase):
    """Test cases for BarPlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = BarPlot()

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

    def test_plot_basic(self):
        """Test basic bar plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, show=False)
        except Exception as e:
            self.fail(f"Bar plot creation failed: {e}")

    def test_plot_with_feature_names(self):
        """Test bar plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Bar plot with feature names failed: {e}")

    def test_plot_with_max_display(self):
        """Test bar plot with max_display limit."""
        try:
            self.plotter.plot(self.shap_values, self.data, max_display=2, show=False)
        except Exception as e:
            self.fail(f"Bar plot with max_display failed: {e}")

    def test_plot_save(self):
        """Test bar plot saving."""
        output_path = Path(self.temp_dir) / "bar_plot.png"

        try:
            self.plotter.plot(
                self.shap_values, self.data, save_path=str(output_path), show=False
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Bar plot saving failed: {e}")

    def test_plot_multi_class(self):
        """Test bar plot with multi-class SHAP values."""
        # Create multi-class SHAP values
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot_multi_class(
                multi_class_shap, self.data, ["class_0", "class_1"], show=False
            )
        except Exception as e:
            self.fail(f"Multi-class bar plot failed: {e}")


class TestBeeswarmPlot(unittest.TestCase):
    """Test cases for BeeswarmPlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = BeeswarmPlot()

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

    def test_plot_basic(self):
        """Test basic beeswarm plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, show=False)
        except Exception as e:
            self.fail(f"Beeswarm plot creation failed: {e}")

    def test_plot_with_feature_names(self):
        """Test beeswarm plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Beeswarm plot with feature names failed: {e}")

    def test_plot_with_max_display(self):
        """Test beeswarm plot with max_display limit."""
        try:
            self.plotter.plot(self.shap_values, self.data, max_display=2, show=False)
        except Exception as e:
            self.fail(f"Beeswarm plot with max_display failed: {e}")

    def test_plot_save(self):
        """Test beeswarm plot saving."""
        output_path = Path(self.temp_dir) / "beeswarm_plot.png"

        try:
            self.plotter.plot(
                self.shap_values, self.data, save_path=str(output_path), show=False
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Beeswarm plot saving failed: {e}")

    def test_plot_multi_class(self):
        """Test beeswarm plot with multi-class SHAP values."""
        # Create multi-class SHAP values
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot_multi_class(
                multi_class_shap, self.data, ["class_0", "class_1"], show=False
            )
        except Exception as e:
            self.fail(f"Multi-class beeswarm plot failed: {e}")

    def test_plot_with_feature_values(self):
        """Test beeswarm plot with feature coloring."""
        try:
            self.plotter.plot_with_feature_values(
                self.shap_values, self.data, color_feature=0, show=False
            )
        except Exception as e:
            self.fail(f"Beeswarm plot with feature coloring failed: {e}")


class TestDecisionPlot(unittest.TestCase):
    """Test cases for DecisionPlot class."""

    def setUp(self):
        """Set up test fixtures."""
        self.plotter = DecisionPlot()

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

    def test_plot_basic(self):
        """Test basic decision plot creation."""
        try:
            self.plotter.plot(self.shap_values, self.data, show=False)
        except Exception as e:
            self.fail(f"Decision plot creation failed: {e}")

    def test_plot_with_feature_names(self):
        """Test decision plot with feature names."""
        try:
            self.plotter.plot(
                self.shap_values,
                self.data,
                feature_names=self.feature_names,
                show=False,
            )
        except Exception as e:
            self.fail(f"Decision plot with feature names failed: {e}")

    def test_plot_with_max_display(self):
        """Test decision plot with max_display limit."""
        try:
            self.plotter.plot(self.shap_values, self.data, max_display=2, show=False)
        except Exception as e:
            self.fail(f"Decision plot with max_display failed: {e}")

    def test_plot_save(self):
        """Test decision plot saving."""
        output_path = Path(self.temp_dir) / "decision_plot.png"

        try:
            self.plotter.plot(
                self.shap_values, self.data, save_path=str(output_path), show=False
            )
            self.assertTrue(output_path.exists())
        except Exception as e:
            self.fail(f"Decision plot saving failed: {e}")

    def test_plot_single_sample(self):
        """Test decision plot for single sample."""
        try:
            self.plotter.plot_single_sample(
                self.shap_values, self.data, sample_idx=0, show=False
            )
        except Exception as e:
            self.fail(f"Single sample decision plot failed: {e}")

    def test_plot_highlighted_samples(self):
        """Test decision plot with highlighted samples."""
        try:
            self.plotter.plot_highlighted_samples(
                self.shap_values, self.data, [0, 1], show=False
            )
        except Exception as e:
            self.fail(f"Highlighted samples decision plot failed: {e}")

    def test_plot_multi_class(self):
        """Test decision plot with multi-class SHAP values."""
        # Create multi-class SHAP values
        multi_class_shap = np.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            ]
        )

        try:
            self.plotter.plot_multi_class(
                multi_class_shap, self.data, ["class_0", "class_1"], show=False
            )
        except Exception as e:
            self.fail(f"Multi-class decision plot failed: {e}")


if __name__ == "__main__":
    unittest.main()
