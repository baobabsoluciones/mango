import shutil
from unittest import TestCase

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from mango.shap_analysis import ShapAnalyzer
from mango.tests.const import normalize_path
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class ObjectTests(TestCase):
    # Init parameters
    X_train = None
    y_train = None
    X_train_reg = None
    y_train_reg = None
    _classification_models = None
    _regression_models = None
    _model_error = None

    @classmethod
    def setUpClass(cls) -> None:
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

        # Create a synthetic dataset
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
        cls._model_error.fit(cls.X_train, cls.y_train)

        # Regression models
        model_4 = Pipeline([("classifier", RandomForestRegressor(random_state=42))])
        model_5 = RandomForestRegressor(random_state=42)
        model_6 = LGBMRegressor(random_state=42)

        # Create a synthetic dataset
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
            model.fit(cls.X_train, cls.y_train)

    def tearDown(self):
        # Remove shap_folder and its content
        shutil.rmtree(normalize_path(f"shap_module/test_shap"))

    @staticmethod
    def _test_shap(
        problem_type,
        estimator,
        model_name,
        X_train,
        shap_folder,
        queries=None,
        metadata=None,
        pdp_tuples=None,
    ):
        # ShapAnalyzer
        instance = ShapAnalyzer(
            problem_type=problem_type,
            model_name=model_name,
            estimator=estimator,
            data=X_train,
            metadata=metadata,
            shap_folder=shap_folder,
        )

        # Make shap analysis
        instance.make_shap_analysis(queries=queries, pdp_tuples=pdp_tuples)

        return instance

    def test_binary_classification_report(self):
        for model in self._classification_models:
            instance = self._test_shap(
                problem_type="binary_classification",
                estimator=model,
                model_name=type(model).__name__,
                X_train=self.X_train,
                shap_folder=normalize_path(
                    f"shap_module/test_shap/binary_classification"
                ),
                queries=["feature_0 == 0.5036366361778611"],
            )

            # Check properties
            self.assertEqual(
                instance.problem_type,
                "binary_classification",
            )
            self.assertEqual(
                instance.model_name,
                type(model).__name__,
            )
            self.assertEqual(
                instance.shap_folder,
                normalize_path(f"shap_module/test_shap/binary_classification"),
            )
            self.assertEqual(
                instance.data.shape,
                self.X_train.shape,
            )
            self.assertEqual(
                instance.metadata,
                [],
            )

            # Check explainer property
            assert instance.shap_explainer

    def test_regression_report(self):
        for model in self._regression_models:
            instance = self._test_shap(
                problem_type="regression",
                estimator=model,
                model_name=type(model).__name__,
                X_train=self.X_train_reg,
                shap_folder=normalize_path(f"shap_module/test_shap/regression"),
                pdp_tuples=[("feature_0", "feature_1")],
                metadata="metadata_index",
                # queries=["metadata_index == 1"],
            )

            # Check properties
            self.assertEqual(
                instance.problem_type,
                "regression",
            )
            self.assertEqual(
                instance.model_name,
                type(model).__name__,
            )
            self.assertEqual(
                instance.shap_folder,
                normalize_path(f"shap_module/test_shap/regression"),
            )
            self.assertEqual(
                instance._data_with_metadata.shape,
                self.X_train_reg.shape,
            )
            self.assertEqual(
                instance.metadata,
                ["metadata_index"],
            )

    def test_get_sample_by_shap_values(self):
        model = self._classification_models[0]
        instance = self._test_shap(
            problem_type="binary_classification",
            estimator=model,
            model_name=type(model).__name__,
            X_train=self.X_train,
            shap_folder=normalize_path(f"shap_module/test_shap/binary_classification"),
            queries=["feature_0 == 0.5036366361778611"],
        )
        sample = instance.get_sample_by_shap_value(
            shap_value=10, feature_name="feature_5", class_name=1
        )

        # Assert sample columns are the same as the original dataset
        self.assertEqual(
            sample.columns.tolist(),
            self.X_train.columns.tolist(),
        )

        with self.assertRaises(ValueError):
            instance.get_sample_by_shap_value(
                shap_value=10, feature_name="feature_S", class_name=1
            )

        with self.assertRaises(ValueError):
            instance.get_sample_by_shap_value(
                shap_value=10, feature_name="feature_5", class_name=2
            )

    def test_get_sample_by_shap_values_regression(self):
        model = self._regression_models[0]
        instance = self._test_shap(
            problem_type="regression",
            estimator=model,
            model_name=type(model).__name__,
            X_train=self.X_train_reg,
            shap_folder=normalize_path(f"shap_module/test_shap/regression"),
            metadata=["metadata_index"],
        )
        sample = instance.get_sample_by_shap_value(
            shap_value=0.05, feature_name="feature_5"
        )

        # Assert sample columns are the same as the original dataset
        self.assertEqual(
            sample.columns.tolist(),
            self.X_train_reg.columns.tolist(),
        )

        with self.assertRaises(ValueError):
            instance.get_sample_by_shap_value(shap_value=10, feature_name="feature_S")

        with self.assertRaises(ValueError):
            instance.get_sample_by_shap_value(
                shap_value=10, feature_name="feature_1", operator=">>"
            )

    def test_values_errors(self):
        # Train
        model = self._classification_models[0]

        # Assert value error when problem_type is not supported
        with self.assertRaises(ValueError):
            ShapAnalyzer(
                problem_type="Fallo",
                model_name=type(model).__name__,
                estimator=model,
                data=self.X_train,
                metadata=None,
                shap_folder=normalize_path(f"shap_module/test_shap/regression"),
            )
        # Assert value error when model_name is not supported
        with self.assertRaises(ValueError):
            ShapAnalyzer(
                problem_type="binary_classification",
                model_name="model",
                estimator=self._model_error,
                data=self.X_train,
                shap_folder=normalize_path(
                    f"shap_module/test_shap/binary_classification"
                ),
            )

        # Assert value error when data type is not supported
        with self.assertRaises(ValueError):
            ShapAnalyzer(
                problem_type="binary_classification",
                model_name=type(model).__name__,
                estimator=model,
                data=[],
                shap_folder=normalize_path(
                    f"shap_module/test_shap/binary_classification"
                ),
            )

        # Assert value error when metadata type is not supported
        with self.assertRaises(ValueError):
            ShapAnalyzer(
                problem_type="binary_classification",
                model_name=type(model).__name__,
                estimator=model,
                data=self.X_train,
                metadata=1,
                shap_folder=normalize_path(
                    f"shap_module/test_shap/binary_classification"
                ),
            )

        # Assert value error when shap_folder type is not supported
        with self.assertRaises(ValueError):
            instance = ShapAnalyzer(
                problem_type="binary_classification",
                model_name=type(model).__name__,
                estimator=model,
                data=self.X_train,
                metadata=None,
                shap_folder=None,
            )
            instance.make_shap_analysis()

        # Assert value error when problem_type is not supported
        with self.assertRaises(ValueError):
            instance = ShapAnalyzer(
                problem_type="regression",
                model_name=type(model).__name__,
                estimator=self._regression_models[1],
                data=self.X_train_reg,
                metadata="metadata_index",
                shap_folder=normalize_path(f"shap_module/test_shap/regression"),
            )
            instance.make_shap_analysis(queries=["metadata_index == -1"])
