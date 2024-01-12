import os
import pickle
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor

from mango.models.experiment_tracking import (
    export_model, ProblemType,
)


class InvalidModel:
    """
    Dummy class to test errors
    """
    pass


class TestExperimentTracking(TestCase):
    """
    Tes suite for the experiment tracking module inside models.
    """
    folder_name = "test_experiment_tracking"

    @classmethod
    def setUpClass(cls):
        """
        Create data for the tests and needed folders.
        """
        os.makedirs(cls.folder_name, exist_ok=True)

        # Classification
        X_clf, y_clf = make_classification(
            n_samples=1000, n_features=10, random_state=42, n_classes=3, n_informative=5
        )
        X_clf = pd.DataFrame(X_clf, columns=[f"feature_{i}" for i in range(10)])
        y_clf = pd.Series(y_clf, name="target")

        # Shuffle
        X_clf = X_clf.sample(frac=1, random_state=42)
        y_clf = y_clf[X_clf.index]

        # Split
        cls.X_train_clf = X_clf[: int(len(X_clf) * 0.8)].reset_index(drop=True)
        cls.y_train_clf = y_clf[: int(len(y_clf) * 0.8)].reset_index(drop=True)
        cls.X_test_clf = X_clf[int(len(X_clf) * 0.8):].reset_index(drop=True)
        cls.y_test_clf = y_clf[int(len(y_clf) * 0.8):].reset_index(drop=True)

        # Regression
        X_reg, y_reg = make_regression(n_samples=1000, n_features=10, random_state=42)
        X_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])
        y_reg = pd.Series(y_reg, name="target")

        # Shuffle
        X_reg = X_reg.sample(frac=1, random_state=42)
        y_reg = y_reg[X_reg.index]

        # Split
        cls.X_train_reg = X_reg[: int(len(X_reg) * 0.8)].reset_index(drop=True)
        cls.y_train_reg = y_reg[: int(len(y_reg) * 0.8)].reset_index(drop=True)
        cls.X_test_reg = X_reg[int(len(X_reg) * 0.8):].reset_index(drop=True)
        cls.y_test_reg = y_reg[int(len(y_reg) * 0.8):].reset_index(drop=True)

    @classmethod
    def tearDownClass(cls):
        """
        Delete the folders created for the tests.
        """
        if os.path.exists(cls.folder_name):
            shutil.rmtree(cls.folder_name)

    def _check_model_with_zip(self, output_folder):
        """
        Helper function to check the model is saved correctly when zip_files is True.
        """
        # Assert zip files are saved
        self.assertTrue(os.path.exists(os.path.join(output_folder, "model.zip")))
        self.assertTrue(os.path.exists(os.path.join(output_folder, "datasets.zip")))

        # Assert files are saved correctly
        self.assertTrue(os.path.exists(os.path.join(output_folder, "summary.json")))

        # Assert files are not saved
        self.assertFalse(
            os.path.exists(os.path.join(output_folder, "model", "model.pkl"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(output_folder, "model", "hyperparameters.json"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(output_folder, "datasets", "X_train.csv"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(output_folder, "datasets", "y_train.csv"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(output_folder, "datasets", "X_test.csv"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(output_folder, "datasets", "y_test.csv"))
        )

        # Assert subfolder not saved
        self.assertFalse(os.path.exists(os.path.join(output_folder, "model")))
        self.assertFalse(os.path.exists(os.path.join(output_folder, "datasets")))

    def _check_model_without_zip(self, model, output_folder, problem_type):
        """
        Helper function to check the model is saved correctly when zip_files is False.
        """
        # Assert folders are saved correctly
        self.assertTrue(os.path.exists(os.path.join(output_folder, "model")))
        self.assertTrue(os.path.exists(os.path.join(output_folder, "datasets")))
        # Assert files are saved correctly
        self.assertTrue(os.path.exists(os.path.join(output_folder, "summary.json")))
        self.assertTrue(
            os.path.exists(os.path.join(output_folder, "model", "hyperparameters.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_folder, "model", "model.pkl"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_folder, "datasets", "X_train.csv"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_folder, "datasets", "y_train.csv"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_folder, "datasets", "X_test.csv"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(output_folder, "datasets", "y_test.csv"))
        )
        # Assert zip files are not saved
        self.assertFalse(os.path.exists(os.path.join(output_folder, "model.zip")))
        self.assertFalse(os.path.exists(os.path.join(output_folder, "datasets.zip")))
        # Assert files are valid for data folder
        X_train = pd.read_csv(os.path.join(output_folder, "datasets", "X_train.csv"))
        y_train = pd.read_csv(os.path.join(output_folder, "datasets", "y_train.csv")).values
        X_test = pd.read_csv(os.path.join(output_folder, "datasets", "X_test.csv"))
        y_test = pd.read_csv(os.path.join(output_folder, "datasets", "y_test.csv")).values
        if problem_type == ProblemType.CLASSIFICATION:
            assert_frame_equal(X_train, self.X_train_clf)
            self.assertListEqual(list([y for y in y_train.reshape(-1)]), list([y for y in self.y_train_clf.values]))
            assert_frame_equal(X_test, self.X_test_clf)
            self.assertListEqual(list([y for y in y_test.reshape(-1)]), list([y for y in self.y_test_clf.values]))
        elif problem_type == ProblemType.REGRESSION:
            assert_frame_equal(X_train, self.X_train_reg)
            self.assertListEqual(list([round(y, 4) for y in y_train.reshape(-1)]),
                                 list([round(y, 4) for y in self.y_train_reg.values]))
            assert_frame_equal(X_test, self.X_test_reg)
            self.assertListEqual(list([round(y, 4) for y in y_test.reshape(-1)]),
                                 list([round(y, 4) for y in self.y_test_reg.values]))
        else:
            raise ValueError("Problem type not supported")
        # Assert model is the same
        # Assert model is the same
        with open(os.path.join(output_folder, "model", "model.pkl"), "rb") as f:
            model_load = pickle.load(f)

        # Generate predictions from both models
        original_predictions = model.predict(self.X_test_reg)
        loaded_predictions = model_load.predict(self.X_test_reg)

        # Check if the predictions are almost the same
        self.assertTrue(np.allclose(original_predictions, loaded_predictions))

    def test_serialize_sklearn(self):
        """
        Test serialization of a sklearn model.
        """
        model = LinearRegression()
        model.fit(self.X_train_reg, self.y_train_reg)
        output_folder = export_model(
            model,
            self.X_train_reg,
            self.y_train_reg,
            self.X_test_reg,
            self.y_test_reg,
            self.folder_name,
            save_model=True,
            save_datasets=True,
            zip_files=False,
        )
        self._check_model_without_zip(output_folder=output_folder, model=model, problem_type=ProblemType.REGRESSION)
        # Assert works for classification with Zip
        model = LogisticRegression()
        model.fit(self.X_train_clf, self.y_train_clf)
        output_folder = export_model(
            model,
            self.X_train_clf,
            self.y_train_clf,
            self.X_test_clf,
            self.y_test_clf,
            self.folder_name,
            save_model=True,
            save_datasets=True,
            zip_files=True,
        )
        self._check_model_with_zip(output_folder=output_folder)

    def test_serialize_catboost(self):
        """
        Test serialization of a CatBoost model.
        """
        model = CatBoostClassifier(allow_writing_files=False, verbose=5, iterations=10)
        model.fit(self.X_train_clf, self.y_train_clf)
        output_folder = export_model(
            model,
            self.X_train_clf,
            self.y_train_clf,
            self.X_test_clf,
            self.y_test_clf,
            self.folder_name,
            save_model=True,
            save_datasets=True,
            zip_files=False,
        )
        self._check_model_without_zip(output_folder=output_folder, model=model, problem_type=ProblemType.CLASSIFICATION)

        # Assert works for regression with Zip
        model = CatBoostRegressor(allow_writing_files=False, verbose=5, iterations=10)
        model.fit(self.X_train_reg, self.y_train_reg)
        output_folder = export_model(
            model,
            self.X_train_reg,
            self.y_train_reg,
            self.X_test_reg,
            self.y_test_reg,
            self.folder_name,
            save_model=True,
            save_datasets=True,
            zip_files=True,
        )
        self._check_model_with_zip(output_folder=output_folder)

    def test_serialize_lightgbm(self):
        """
        Test serialization of a LightGBM model.
        """
        model = LGBMClassifier()
        model.fit(self.X_train_clf, self.y_train_clf)
        output_folder = export_model(
            model,
            self.X_train_clf,
            self.y_train_clf,
            self.X_test_clf,
            self.y_test_clf,
            self.folder_name,
            save_model=True,
            save_datasets=True,
            zip_files=False,
        )
        self._check_model_without_zip(output_folder=output_folder, model=model, problem_type=ProblemType.CLASSIFICATION)

        # Assert works for regression with Zip
        model = LGBMRegressor()
        model.fit(self.X_train_reg, self.y_train_reg)
        output_folder = export_model(
            model,
            self.X_train_reg,
            self.y_train_reg,
            self.X_test_reg,
            self.y_test_reg,
            self.folder_name,
            save_model=True,
            save_datasets=True,
            zip_files=True,
        )
        self._check_model_with_zip(output_folder=output_folder)

    def test_errors(self):
        """
        Test errors raised by the function.
        """
        # Not supported model
        model = InvalidModel()
        with self.assertRaises(ValueError):
            export_model(
                model,
                self.X_train_reg,
                self.y_train_reg,
                self.X_test_reg,
                self.y_test_reg,
                self.folder_name,
                save_model=True,
                save_datasets=True,
                zip_files=False,
            )

        # Invalid folder
        with self.assertRaises(FileNotFoundError):
            export_model(
                model,
                self.X_train_reg,
                self.y_train_reg,
                self.X_test_reg,
                self.y_test_reg,
                "invalid_folder",
                save_model=True,
                save_datasets=True,
                zip_files=False,
            )
