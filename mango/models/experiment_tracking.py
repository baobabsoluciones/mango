import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from typing import Any, Optional, Union, Tuple

import pandas as pd
from matplotlib import pyplot as plt

from .enums import ProblemType, ModelLibrary
from .metrics import (
    generate_metrics_regression,
    generate_metrics_classification,
)
from mango.config import BaseConfig

from pandas.testing import assert_frame_equal, assert_series_equal


class _DummyPipeline:
    pass


class _DummyLinearRegression:
    pass


class _DummyLogisticRegression:
    pass


def _json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def _clean_hyperparameters(hyperparameters: dict) -> dict:
    for key, value in hyperparameters.items():
        if isinstance(value, dict):
            _clean_hyperparameters(value)
        elif not _json_serializable(value):
            hyperparameters[key] = str(value)
    return hyperparameters


def export_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_path: str,
    custom_metrics: dict = None,
    description: str = None,
    base_folder_name: str = None,
    save_model: bool = True,
    save_datasets: bool = False,
    zip_files: bool = True,
) -> str:
    """
    Register model and metrics in a json file and save the model and datasets in a folder.

    :param model: A model from one of the supported libraries.
    :type model: :class:`Any`
    :param X_train: Training data as a pandas dataframe.
    :type X_train: :class:`pandas.DataFrame`
    :param y_train: Training target as a pandas series.
    :type y_train: :class:`pandas.Series`
    :param X_test: Test data as a pandas dataframe.
    :type X_test: :class:`pandas.DataFrame`
    :param y_test: Test target as a pandas series.
    :type y_test: :class:`pandas.Series`
    :param description: Description of the experiment.
    :type description: :class:`str`
    :param base_path: Path to the base folder where the model and datasets will be saved in a subfolder structure.
    :type base_path: :class:`str`
    :param base_folder_name: Custom name for the folder where the model and datasets will be saved.
    :type base_folder_name: :class:`str`
    :param zip_files: Whether to zip the files or not.
    :type zip_files: :class:`bool`
    :param save_datasets: Whether to save the datasets or not.
    :type save_datasets: :class:`bool`
    :param save_model: Whether to save the model or not.
    :type save_model: :class:`bool`
    :return: The path to the subfolder inside base_path where the model and datasets have been saved.
    :rtype: :class:`str`

    Usage
    -----
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> output_folder = export_model(model, X_train, y_train, X_test, y_test, "/my_experiments_folder")
    >>> print(output_folder) # /my_experiments_folder/experiment_LogisticRegression_YYYYMMDD-HHMMSS
    """
    _SUPPORTED_LIBRARIES_CLASSES = {}
    try:
        from sklearn.base import BaseEstimator
        from sklearn.pipeline import Pipeline

        pipeline_class = Pipeline

        _SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.SCIKIT_LEARN] = BaseEstimator
    except ImportError:
        pipeline_class = _DummyPipeline
    try:
        from catboost import CatBoost

        _SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.CATBOOST] = CatBoost
    except ImportError:
        pass
    try:
        from lightgbm import LGBMModel

        _SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.LIGHTGBM] = LGBMModel
    except ImportError:
        pass

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Folder {base_path} does not exist.")

    model_name = model.__class__.__name__

    if isinstance(model, pipeline_class):
        pipeline = model
        col_transformer = model[0]
        model = model[-1]
    else:
        pipeline = None
        col_transformer = None
    model_library = None
    for library, class_name in _SUPPORTED_LIBRARIES_CLASSES.items():
        if isinstance(model, class_name):
            model_library = library
    if model_library is None:
        raise ValueError(f"Model {model_name} is not supported.")

    # Detect if it is a classification or regression model
    if hasattr(model, "predict_proba"):
        problem_type = ProblemType.CLASSIFICATION
    else:
        problem_type = ProblemType.REGRESSION
    summary = {}
    extra_params = []
    # Fill structure
    summary["description"] = description
    summary["name"] = base_folder_name or model_name
    summary["training_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary["model"] = {}
    summary["model"]["name"] = model_name
    summary["model"]["problem_type"] = problem_type.value
    summary["model"]["target"] = y_train.name
    summary["model"]["library"] = model_library.value
    if model_library == ModelLibrary.CATBOOST:
        if pipeline is not None:
            summary["model"]["input"] = list(col_transformer.get_feature_names_out())
            summary["model"]["hyperparameters"] = pipeline.get_params(deep=True)
        else:
            summary["model"]["hyperparameters"] = model.get_all_params()
            summary["model"]["input"] = list(model.feature_names_)

    elif model_library == ModelLibrary.SCIKIT_LEARN:
        if pipeline is not None:
            summary["model"]["input"] = list(col_transformer.get_feature_names_out())
            summary["model"]["hyperparameters"] = pipeline.get_params(deep=True)
        else:
            summary["model"]["input"] = list(model.feature_names_in_)
            summary["model"]["hyperparameters"] = model.get_params(deep=True)
    elif model_library == ModelLibrary.LIGHTGBM:
        if pipeline is not None:
            summary["model"]["input"] = list(col_transformer.get_feature_names_out())
            summary["model"]["hyperparameters"] = pipeline.get_params(deep=True)
        else:
            summary["model"]["input"] = list(model.feature_name_)
            summary["model"]["hyperparameters"] = model.get_params(deep=True)

    # Clean hyperparameters for the sklearn pipeline or other non-serializable objects
    _clean_hyperparameters(summary["model"]["hyperparameters"])

    # Sort keys in summary["model"]
    if problem_type == ProblemType.CLASSIFICATION:
        summary["model"]["num_classes"] = len(y_train.unique())
        # Sort keys in summary["model"] to be: name, problem_type, num_classes, input, target, hyperparameters, library
        summary["model"] = {
            k: summary["model"][k]
            for k in [
                "name",
                "problem_type",
                "num_classes",
                "input",
                "target",
                "hyperparameters",
                "library",
            ]
        }
    else:
        # Sort keys in summary["model"] to be: name, problem_type, input, target, hyperparameters, library
        summary["model"] = {
            k: summary["model"][k]
            for k in [
                "name",
                "problem_type",
                "input",
                "target",
                "hyperparameters",
                "library",
            ]
        }

    # Restore pipeline to model variable
    if pipeline:
        model = pipeline

    # Generate metrics
    if model_library == ModelLibrary.CATBOOST:
        y_train_pred = pd.Series(model.predict(X_train).reshape(-1)).reset_index(
            drop=True
        )
        y_test_pred = pd.Series(model.predict(X_test).reshape(-1)).reset_index(
            drop=True
        )
    elif model_library in [ModelLibrary.SCIKIT_LEARN, ModelLibrary.LIGHTGBM]:
        y_train_pred = pd.Series(model.predict(X_train)).reset_index(drop=True)
        y_test_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)

    if problem_type == ProblemType.CLASSIFICATION:
        if not custom_metrics:
            summary["results"] = {
                "train_score": generate_metrics_classification(
                    y_train.reset_index(drop=True), y_train_pred
                ),
                "test_score": generate_metrics_classification(
                    y_test.reset_index(drop=True), y_test_pred
                ),
            }
        else:
            summary["results"] = custom_metrics
    elif problem_type == ProblemType.REGRESSION:
        summary["results"] = {
            "train_score": generate_metrics_regression(
                y_train.reset_index(drop=True), y_train_pred
            ),
            "test_score": generate_metrics_regression(
                y_test.reset_index(drop=True), y_test_pred
            ),
        }

    # Prepare environment to save files
    folder_name_default = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_experiment_{model_name}"
    )
    folder_name = base_folder_name or folder_name_default
    folder_name = os.path.join(
        base_path, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{folder_name}"
    )

    # Compress model and save
    if save_model:
        os.makedirs(os.path.join(folder_name, "model"))
        if not "files" in summary:
            summary["files"] = {}
        if not "model" in summary["files"]:
            summary["files"]["model"] = {}
        # Save hyperparameters
        hyperparameters_path = os.path.join(
            folder_name, "model", "hyperparameters.json"
        )
        summary["files"]["model"]["hyperparameters.json"] = os.path.abspath(
            hyperparameters_path
        )
        with open(hyperparameters_path, "w") as f:
            json.dump(summary["model"]["hyperparameters"], f, indent=4)
        # Save the model
        model_path = os.path.join(folder_name, "model", "model.pkl")
        summary["files"]["model"]["model.pkl"] = os.path.abspath(model_path)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        if zip_files:
            zip_path = os.path.join(folder_name, "model.zip")
            summary["files"]["model"]["zip"] = os.path.abspath(zip_path)
            shutil.make_archive(
                zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "model")
            )
            shutil.rmtree(os.path.join(folder_name, "model"))

    if save_datasets:
        os.makedirs(os.path.join(folder_name, "datasets"))
        if not "files" in summary:
            summary["files"] = {}
        if not "datasets" in summary["files"]:
            summary["files"]["datasets"] = {}
        X_train_path = os.path.join(folder_name, "datasets", "X_train.csv")
        summary["files"]["datasets"]["X_train"] = {}
        summary["files"]["datasets"]["X_train"]["path"] = os.path.abspath(X_train_path)
        summary["files"]["datasets"]["X_train"]["shape"] = X_train.shape
        X_train.to_csv(X_train_path, index=False)
        y_train_path = os.path.join(folder_name, "datasets", "y_train.csv")
        summary["files"]["datasets"]["y_train"] = {}
        summary["files"]["datasets"]["y_train"]["path"] = os.path.abspath(y_train_path)
        summary["files"]["datasets"]["y_train"]["shape"] = y_train.shape
        y_train.to_csv(y_train_path, index=False)
        X_test_path = os.path.join(folder_name, "datasets", "X_test.csv")
        summary["files"]["datasets"]["X_test"] = {}
        summary["files"]["datasets"]["X_test"]["path"] = os.path.abspath(X_test_path)
        summary["files"]["datasets"]["X_test"]["shape"] = X_test.shape
        X_test.to_csv(X_test_path, index=False)
        y_test_path = os.path.join(folder_name, "datasets", "y_test.csv")
        summary["files"]["datasets"]["y_test"] = {}
        summary["files"]["datasets"]["y_test"]["path"] = os.path.abspath(y_test_path)
        summary["files"]["datasets"]["y_test"]["shape"] = y_test.shape
        y_test.to_csv(y_test_path, index=False)
        if zip_files:
            # Compress data and save
            zip_path = os.path.join(folder_name, "datasets.zip")
            summary["files"]["datasets"]["zip"] = {}
            summary["files"]["datasets"]["zip"]["path"] = os.path.abspath(zip_path)
            shutil.make_archive(
                zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "datasets")
            )
            shutil.rmtree(os.path.join(folder_name, "datasets"))

    # Save json
    json_path = os.path.join(folder_name, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    return folder_name


class MLExperiment:
    """
    MLExperiment is a class that represents a machine learning experiment. It provides functionalities to initialize metrics,
    get feature importance, plot ROC curve, plot precision recall curve, plot feature importance, register an experiment,
    predict using the model, and load an experiment from a registered experiment.

    Currently, the following libraries are supported both for regression and classification problems:
        - scikit-learn
        - lightgbm
        - catboost

    Attributes:
        - **config:** Configuration for the experiment. (Not implemented yet)
        - **X_train:** Training data.
        - **y_train:** Training target.
        - **X_test:** Test data.
        - **y_test:** Test target.
        - **model:** A model from one of the supported libraries.
        - **problem_type:** Type of the problem (classification or regression).
        - **name:** Name of the experiment.
        - **description:** Description of the experiment.

    Methods:
        - **get_feature_importance():** Returns the feature importance of the model. If linear model, returns the coefficients.
        - **plot_roc_curve(show=False):** Plots the ROC curve of the experiment. If show is True, it displays the plot.
        - **plot_precision_recall_curve(show=False):** Plots the precision recall curve of the experiment. If show is True, it displays the plot.
        - **plot_feature_importance(show=False):** Plots the feature importance of the experiment. If show is True, it displays the plot.
        - **register_experiment(base_path, save_model=True, save_datasets=True, zip_files=True):** Registers the experiment and saves it as a zip file.
        - **from_registered_experiment(experiment_path):** Loads the experiment from a registered experiment.

    Usage
    -----
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> experiment = MLExperiment(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, problem_type=ProblemType.CLASSIFICATION, name="Logistic Regression Experiment", description="This is a logistic regression experiment.")
    >>> experiment.plot_roc_curve(show=True)
    >>> experiment.plot_precision_recall_curve(show=True)
    >>> experiment.plot_feature_importance(show=True)
    >>> experiment.register_experiment(base_path="/my_experiments_folder")
    >>> loaded_experiment = MLExperiment.from_registered_experiment(experiment_path="/my_experiments_folder/Logistic Regression Experiment")
    """

    def __init__(
        self,
        *,
        config: BaseConfig = None,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model: Any = None,
        problem_type: Union[str, ProblemType] = None,
        name: str = None,
        description: Optional[str] = None,
    ):
        """
        Initializes an instance of the MLExperiment class.

        :param config: Configuration for the experiment. Not implemented yet.
        :type config: :class:`BaseConfig`, optional
        :param X_train: Training data.
        :type X_train: :class:`pd.DataFrame`, optional
        :param y_train: Training target.
        :type y_train: :class:`pd.Series`, optional
        :param X_test: Test data.
        :type X_test: :class:`pd.DataFrame`, optional
        :param y_test: Test target.
        :type y_test: :class:`pd.Series`, optional
        :param model: A model from one of the supported libraries.
        :type model: Any, optional
        :param problem_type: Type of the problem (classification or regression).
        :type problem_type: Union[str, ProblemType], optional
        :param name: Name of the experiment.
        :type name: str, optional
        :param description: Description of the experiment.
        :type description: str, optional

        :raises NotImplementedError: If the config parameter is provided, as it's not implemented yet.
        """
        # For this version not implement config setup of the experiment
        if config:
            raise NotImplementedError("Config usage is not implemented yet.")

        # Search for supported libraries
        self._search_for_supported_libraries()

        # Public properties (Not defined in the if config block)
        self.name = name
        self.description = description
        self.problem_type = problem_type
        self.model = model
        self.base_model = None
        self.num_classes = None
        self.imbalance = None
        self.metrics = None
        self.best_threshold_roc_curve = self.best_threshold_pr_curve = 0.5
        self.base_model_library = None

        # Setup datasets
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        if self.problem_type == ProblemType.CLASSIFICATION:
            self.num_classes = len(self.y_test.unique())
            # Consider imbalance if for every 5 positive examples there are 25 negative examples.
            self.imbalance = (
                self.y_train.value_counts().values[1]
                / self.y_train.value_counts().values[0]
                < 0.2
            )

        # Private properties
        self._tpr_list = None
        self._fpr_list = None
        self._precision_list = None
        self._recall_list = None
        self._config = None
        self._is_pipeline = isinstance(self.model, self.pipeline_class)

        # Final Setup
        self._set_base_model_and_library()
        self._set_datasets_dtypes()
        self._init_metrics()

    def __eq__(self, other):
        assert isinstance(other, MLExperiment), "Can only compare with MLExperiment"
        assert_frame_equal(self.X_train, other.X_train, check_dtype=False)
        assert_series_equal(self.y_train, other.y_train, check_dtype=False)
        assert_frame_equal(self.X_test, other.X_test, check_dtype=False)
        assert_series_equal(self.y_test, other.y_test, check_dtype=False)
        return (
            self.name == other.name
            and self.description == other.description
            # and self.model == other.model # Cannot compare models
            and self.metrics == other.metrics
            and self.problem_type == other.problem_type
            and self.num_classes == other.num_classes
            # and self.base_model == other.base_model # Cannot compare models
            and self.base_model_library == other.base_model_library
            and self.imbalance == other.imbalance
        )

    # Properties
    @property
    def name(self) -> str:
        """
        Name of the experiment.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def description(self) -> str:
        """
        Description of the experiment.
        """
        return self._description

    @description.setter
    def description(self, value):
        if value is None:
            logging.warning("Description is empty.")
        self._description = value

    @property
    def model(self) -> Any:
        """
        The full model from the supported libraries.
        """
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def X_train(self) -> pd.DataFrame:
        """
        Training data.
        """
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        if value is None:
            raise ValueError("X_train cannot be None.")
        self._X_train = value

    @property
    def y_train(self) -> pd.Series:
        """
        Training target.
        """
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        if value is None:
            raise ValueError("y_train cannot be None.")
        if isinstance(value, pd.DataFrame):
            if value.shape[1] == 1:
                value = value.iloc[:, 0]
            else:
                raise ValueError("y_train must be a pandas Series.")
        if not isinstance(value, pd.Series):
            raise ValueError("y_train must be a pandas Series.")
        self._y_train = value

    @property
    def X_test(self) -> pd.DataFrame:
        """
        Test data.
        """
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        if value is None:
            raise ValueError("X_test cannot be None.")
        self._X_test = value

    @property
    def y_test(self) -> pd.Series:
        """
        Test target.
        """
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        if value is None:
            raise ValueError("y_test cannot be None.")
        if isinstance(value, pd.DataFrame):
            if value.shape[1] == 1:
                value = value.iloc[:, 0]
            else:
                raise ValueError("y_train must be a pandas Series.")
        if not isinstance(value, pd.Series):
            raise ValueError("y_train must be a pandas Series.")
        self._y_test = value

    @property
    def metrics(self) -> dict:
        """
        Dictionary with the metrics of the experiment.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    @property
    def problem_type(self) -> ProblemType:
        """
        Type of the problem (classification or regression).
        """
        return self._problem_type

    @problem_type.setter
    def problem_type(self, value):
        if value is None:
            raise ValueError("problem_type cannot be None.")
        # Check if is already an enum
        if isinstance(value, ProblemType):
            self._problem_type = value
        else:
            self._problem_type = ProblemType(value)

    @property
    def num_classes(self) -> Optional[int]:
        """
        Number of classes in the classification problem. If it's a regression problem, it's None.
        """
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

    @property
    def base_model(self) -> Any:
        """
        The base model from the supported libraries. If model is a pipeline, it's the last step of the pipeline,
        otherwise it's the model.
        """
        return self._base_model

    @base_model.setter
    def base_model(self, value):
        self._base_model = value

    @property
    def base_model_library(self) -> ModelLibrary:
        """
        The library of the base model.
        """
        return self._base_model_library

    @base_model_library.setter
    def base_model_library(self, value):
        self._base_model_library = value

    @property
    def imbalance(self) -> Optional[bool]:
        """
        Whether the problem is imbalanced or not. If it's a regression problem, it's None.
        """
        return self._imbalance

    @imbalance.setter
    def imbalance(self, value):
        self._imbalance = value

    # Utility methods
    def _search_for_supported_libraries(self):
        """
        Search if libraries are installed and lazy import them.
        """
        self._SUPPORTED_LIBRARIES_CLASSES = {}
        try:
            from sklearn.base import BaseEstimator
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression, LinearRegression

            self.pipeline_class = Pipeline
            self.sklearn_linear_regression_class = LinearRegression
            self.sklearn_logistic_regression_class = LogisticRegression

            self._SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.SCIKIT_LEARN] = BaseEstimator
        except ImportError:
            self.pipeline_class = _DummyPipeline
            self.sklearn_linear_regression_class = _DummyLinearRegression
            self.sklearn_logistic_regression_class = _DummyLogisticRegression
        try:
            from catboost import CatBoost

            self._SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.CATBOOST] = CatBoost
        except ImportError:
            pass
        try:
            from lightgbm import LGBMModel

            self._SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.LIGHTGBM] = LGBMModel
        except ImportError:
            pass

    def _set_datasets_dtypes(self):
        """
        Set the datasets dtypes to the correct ones so that CatBoost works.
        """
        # Set X_train dtypes
        if self.base_model_library == ModelLibrary.CATBOOST:
            for col_idx in self.base_model.get_param("cat_features") or []:
                self.X_train.iloc[:, col_idx] = self.X_train.iloc[:, col_idx].astype(
                    str
                )
                self.X_test.iloc[:, col_idx] = self.X_test.iloc[:, col_idx].astype(str)

    def _generate_classification_metrics_with_threshold(self):
        """
        Helper function to generate the classification metrics with different thresholds.
        """
        self.metrics = {"train_score": {}, "test_score": {}}
        if self.num_classes == 2:
            y_pred_train = self.model.predict_proba(self.X_train)[:, 1]
            y_pred_test = self.model.predict_proba(self.X_test)[:, 1]
            for threshold in [i / 100 for i in range(1, 101)]:
                self.metrics["train_score"][
                    threshold
                ] = generate_metrics_classification(
                    self.y_train, y_pred_train >= threshold
                )
                self.metrics["test_score"][threshold] = generate_metrics_classification(
                    self.y_test, y_pred_test >= threshold
                )
        else:
            self.metrics = {}
            y_pred_train = self.model.predict(self.X_train)
            y_pred_test = self.model.predict(self.X_test)
            self.metrics["train_score"] = generate_metrics_classification(
                self.y_train, y_pred_train
            )
            self.metrics["test_score"] = generate_metrics_classification(
                self.y_test, y_pred_test
            )

    @staticmethod
    def _find_saving_parameters_from_structure(experiment_folder: str) -> dict:
        """
        Find the paramrters used to export the experiment from the structure of the experiment folder.
        Walk around the folder and find the files.
        Returns a dictionary with the following keys:

            - save_datasets: Whether the datasets were saved or not.
            - save_model: Whether the model was saved or not.
            - zip_files: Whether the files were zipped or not.

        :param experiment_folder: Path to the experiment folder.
        :type experiment_folder: str
        :return: A dictionary with the saving parameters.
        """
        if not os.path.exists(experiment_folder):
            raise FileNotFoundError(f"The folder {experiment_folder} does not exist.")

        for root, dirs, files in os.walk(experiment_folder):
            assert (
                "summary.json" in files
            ), "The summary.json file is missing. Check if folder is a valid experiment folder."
            # Filter possible new files in new versions of experiments.
            files = [
                file
                for file in files
                if file in ["summary.json", "model.zip", "datasets.zip"]
            ]
            # Check if the files are in the root folder.
            if "model.zip" in files or "datasets.zip" in files:
                return {
                    "save_datasets": True if "datasets.zip" in files else False,
                    "save_model": True if "model.zip" in files else False,
                    "zip_files": True,
                }
            # Check if subfolders exist.
            if "model" in dirs or "data" in dirs:
                return {
                    "save_datasets": True if "datasets" in dirs else False,
                    "save_model": True if "model" in dirs else False,
                    "zip_files": False,
                }

    @staticmethod
    def _unzip_experiment_folder(experiment_path: str):
        """
        Unzip the experiment folder.
        :param experiment_path: Path to the experiment folder.
        :type experiment_path: str
        :return: None
        """
        files = [
            file
            for file in os.listdir(experiment_path)
            if file in ["model.zip", "datasets.zip"]
        ]
        for file in files:
            # Think of a better way to do this with shutil.
            shutil.unpack_archive(
                os.path.join(experiment_path, file),
                os.path.join(experiment_path, file.rstrip(".zip")),
            )
            os.remove(os.path.join(experiment_path, file))

    @staticmethod
    def _zip_experiment_folder(experiment_path: str):
        """
        Zip the experiment folder.
        :param experiment_path: Path to the experiment folder.
        :type experiment_path: str
        :return:
        """
        unzipped_folders = [
            folder
            for folder in os.listdir(experiment_path)
            if folder in ["model", "datasets"]
        ]
        for folder in unzipped_folders:
            shutil.make_archive(
                os.path.join(experiment_path, folder),
                "zip",
                os.path.join(experiment_path, folder),
            )
            shutil.rmtree(os.path.join(experiment_path, folder))

    def _load_model_from_config(self):
        """
        Load the model from the config.
        :return:
        """
        pass

    def _set_base_model_and_library(self):
        """
        Get the model library from the model or pipeline.
        Sets the following attributes:

            - base_model
            - base_model_library

        :return: None
        """
        # Detect if pipeline or model
        if self._is_pipeline:
            # Get the last step
            model = self.model[-1]
            self.base_model = model
        else:
            model = self.model
            self.base_model = model

        # Get the library
        matching_libraries = []
        for library, class_name in self._SUPPORTED_LIBRARIES_CLASSES.items():
            if isinstance(model, class_name):
                matching_libraries.append(library)
            # Some models inherit from sklearn hence if len(matching_libraries) > 1 and sklearn is one of them pop it
        if len(matching_libraries) == 1:
            pass
        elif (
            len(matching_libraries) == 2
            and ModelLibrary.SCIKIT_LEARN in matching_libraries
        ):
            matching_libraries.remove(ModelLibrary.SCIKIT_LEARN)
        else:
            raise ValueError(
                f"Could not detect library or is not installed. Model name {model.__class__.__name__}"
            )
        self.base_model_library = matching_libraries[0]

    def _calc_precision_recall_curve_data(self):
        """
        Get the data to plot the precision recall curve.
        Sets the following attributes:

            - _precision_list
            - _recall_list
            - best_threshold_pr_curve

        :return:
        """
        if self.num_classes is not None and self.num_classes > 2:
            raise NotImplementedError(
                "Precision recall curve is only supported for binary classification."
            )
        elif self.num_classes is None:
            raise ValueError(
                "Precision recall curve is only for classification problems"
            )
        precision_list = []
        recall_list = []
        best_distance = 9999
        best_threshold = None
        for threshold, metric in self.metrics["test_score"].items():
            precision = metric["precision"]
            recall = metric["recall"]
            precision_list.append(precision)
            recall_list.append(recall)
            distance = (precision - 1) ** 2 + (recall - 1) ** 2
            if distance <= best_distance:
                best_distance = distance
                best_threshold = threshold
        self.best_threshold_pr_curve = best_threshold
        self._precision_list = precision_list
        self._recall_list = recall_list

    def _calc_roc_curve_data(self):
        """
        Get the data to plot the roc curve.
        Sets the following attributes:

            - _tpr_list
            - _fpr_list
            - best_threshold_roc_curve

        :return:
        """
        if self.num_classes is not None and self.num_classes > 2:
            raise NotImplementedError(
                "ROC curve is only supported for binary classification."
            )
        elif self.num_classes is None:
            raise ValueError("ROC curve is only for classification problems")
        tpr_list = []
        fpr_list = []
        best_distance = 9999
        best_threshold = None
        for threshold, metric in self.metrics["test_score"].items():
            (tn, fp), (fn, tp) = metric["confusion_matrix"]
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            distance = (tpr - 1) ** 2 + (fpr - 0) ** 2
            if distance <= best_distance:
                best_distance = distance
                best_threshold = threshold
        self.best_threshold_roc_curve = best_threshold
        self._tpr_list = tpr_list
        self._fpr_list = fpr_list

    # Public methods
    def _init_metrics(self):
        """
        Initialize the metrics for the experiment.
        Sets the following attributes:

            - metrics

        :return:
        """
        if self.problem_type == ProblemType.REGRESSION:
            self.metrics = generate_metrics_regression(
                self.y_test, self.model.predict(self.X_test)
            )
        elif self.problem_type == ProblemType.CLASSIFICATION:
            self._generate_classification_metrics_with_threshold()
            if self.num_classes == 2:
                self._calc_precision_recall_curve_data()
                self._calc_roc_curve_data()

    def get_feature_importance(self) -> pd.Series:
        """
        Get the feature importance of the model. In case of a linear model, it returns the coefficients.
        :return: A pandas Series with the feature importance.
        :rtype: :class:`pd.Series`
        :raises NotImplementedError: If the model does not support feature importance.
        """

        is_linear_model = isinstance(
            self.base_model,
            (
                self.sklearn_linear_regression_class,
                self.sklearn_logistic_regression_class,
            ),
        )

        if self._is_pipeline:
            # Assume first step is the column transformer
            feature_names = self.model[0].get_feature_names_out()
        else:
            feature_names = self.X_train.columns

        if is_linear_model:
            # Linear model from sklearn
            feature_importance = self.base_model.coef_[0]
            return pd.Series(feature_importance, index=feature_names).sort_values(
                ascending=False
            )

        if hasattr(self.base_model, "feature_importances_"):
            # Feature importance from model
            feature_importance = self.base_model.feature_importances_
            return pd.Series(feature_importance, index=feature_names).sort_values(
                ascending=False
            )
        raise NotImplementedError(
            f"Feature importance is not supported for model {self.base_model.__class__.__name__}"
        )

    def plot_roc_curve(
        self, show: bool = False
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plot the ROC curve. If show is True, it displays the plot.
        :param show: Whether to display the plot or not.
        :type show: bool, optional
        :return: A tuple with the matplotlib Figure and Axes.
        :rtype: Tuple[plt.Figure, plt.Axes]
        :raises ValueError: If the problem is not classification.
        :raises NotImplementedError: If the problem is not binary classification.
        """
        if self.num_classes is None:
            raise ValueError("ROC curve is only for classification problems")
        elif self.num_classes > 2:
            raise NotImplementedError(
                "ROC curve is only supported for binary classification."
            )
        fig, ax = plt.subplots(figsize=(15, 10))
        # Scatter and show cmap legend
        thresholds = list(self.metrics["test_score"].keys())
        ax.scatter(self._fpr_list, self._tpr_list, c=thresholds, cmap="viridis")
        ax.set_title(f"ROC Curve, best threshold {self.best_threshold_roc_curve:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        # Add circle around best threshold
        best_threshold_idx = int(
            self.best_threshold_roc_curve * 100 - 1
        )  # Due to how the thresholds are generated
        ax.scatter(
            self._fpr_list[best_threshold_idx],
            self._tpr_list[best_threshold_idx],
            s=100,
            facecolors="none",
            edgecolors="r",
        )
        fig.add_axes(ax)
        fig.colorbar(ax.collections[0], ax=ax)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig, ax

    def plot_precision_recall_curve(
        self, show=False
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plot the precision recall curve. If show is True, it displays the plot.
        :param show: Whether to display the plot or not.
        :type show: bool, optional
        :return: A tuple with the matplotlib Figure and Axes.
        :rtype: Tuple[plt.Figure, plt.Axes]
        :raises ValueError: If the problem is not classification.
        :raises NotImplementedError: If the problem is not binary classification.
        """
        if self.num_classes is None:
            raise ValueError(
                "Precision recall curve is only for classification problems"
            )
        elif self.num_classes > 2:
            raise NotImplementedError(
                "Precision recall curve is only supported for binary classification."
            )
        fig, ax = plt.subplots(figsize=(15, 10))
        # Scatter and show cmap legend
        thresholds = list(self.metrics["test_score"].keys())
        ax.scatter(
            self._recall_list, self._precision_list, c=thresholds, cmap="viridis"
        )
        ax.set_title(
            f"Precision Recall Curve, best threshold {self.best_threshold_pr_curve:.2f}"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        # Add circle around best threshold
        best_threshold_idx = int(self.best_threshold_pr_curve * 100 - 1)
        ax.scatter(
            self._recall_list[best_threshold_idx],
            self._precision_list[best_threshold_idx],
            s=100,
            facecolors="none",
            edgecolors="r",
        )
        fig.add_axes(ax)
        fig.colorbar(ax.collections[0], ax=ax)
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig, ax

    def plot_feature_importance(
        self, show=False
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plot the feature importance. If show is True, it displays the plot.
        :param show: Whether to display the plot or not.
        :type show: bool, optional
        :return: A tuple with the matplotlib Figure and Axes.
        :rtype: Tuple[plt.Figure, plt.Axes]
        """
        importance = self.get_feature_importance()
        fig, ax = plt.subplots(figsize=(20, 40))
        # Sort importance
        importance = importance.sort_values(ascending=True)
        ax.barh(importance.index, importance.values)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        # Tight layout
        fig.tight_layout()
        if show:
            plt.show()
        else:
            return fig, ax

    def register_experiment(
        self,
        base_path,
        zip_files: bool = True,
    ) -> str:
        """
        Register the experiment and save it
        :param base_path: Path to the folder where the experiment will be saved.
        :type base_path: str
        :param zip_files: Whether to zip the files or not.
        :type zip_files: bool, optional
        :return: The path to the experiment folder.
        :rtype: str
        """
        custom_metrics = self.metrics
        if self.num_classes == 2:
            # Make sure is inserted at the beginning of the dictionary.
            threshold = (
                self.best_threshold_pr_curve
                if self.imbalance
                else self.best_threshold_roc_curve
            )
            custom_metrics = {
                "best_threshold": {
                    "value": threshold,
                    "train_score": self.metrics["train_score"][threshold],
                    "test_score": self.metrics["test_score"][threshold],
                },
                **custom_metrics,
            }
        return export_model(
            self.model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            description=self.description,
            custom_metrics=custom_metrics,
            base_path=base_path,
            base_folder_name=self.name,
            save_model=True,
            save_datasets=True,
            zip_files=zip_files,
        )

    @classmethod
    def from_registered_experiment(cls, experiment_path: str):
        """
        Load the experiment from a registered experiment.
        :param experiment_path: Path to the experiment folder.
        :return: An instance of MLExperiment.
        """
        saving_params = cls._find_saving_parameters_from_structure(experiment_path)

        # Try-except-finally to make sure we zip the folder again if it was unzipped and an exception is raised.
        try:
            if saving_params["zip_files"]:
                cls._unzip_experiment_folder(experiment_path)
            with open(os.path.join(experiment_path, "summary.json"), "r") as f:
                summary = json.load(f)
            # Set params
            with open(os.path.join(experiment_path, "model", "model.pkl"), "rb") as f:
                model = pickle.load(f)

            X_train = pd.read_csv(
                os.path.join(experiment_path, "datasets", "X_train.csv"),
                low_memory=False,
            )
            y_train = pd.read_csv(
                os.path.join(experiment_path, "datasets", "y_train.csv")
            )
            X_test = pd.read_csv(
                os.path.join(experiment_path, "datasets", "X_test.csv"),
                low_memory=False,
            )
            y_test = pd.read_csv(
                os.path.join(experiment_path, "datasets", "y_test.csv")
            )
            # Make sure is a pd.Series
            y_train = y_train.iloc[:, 0]
            y_test = y_test.iloc[:, 0]

            experiment = cls(
                name=summary.get("name", experiment_path.split("-", 1)[1].rstrip("/")),
                description=summary.get("description", ""),
                problem_type=ProblemType(summary["model"]["problem_type"]),
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        except Exception as e:
            raise e
        finally:
            # Raise exception but make sure we zip the folder again if it was unzipped.
            if saving_params["zip_files"]:
                # Zip the folder again.
                cls._zip_experiment_folder(experiment_path)
        return experiment


class MLTracker:
    """
    MLTracker is a class that manages multiple machine learning experiments. It provides functionalities to scan for
    existing experiments, add new experiments, compare experiments, update experiment metrics, and generate comparison
    dataframes and hyperparameters json.

    Attributes:
        **experiment_folder:** The folder where the experiments are stored.
        **experiments:** A dictionary of the experiments.

    Methods:
        **scan_for_experiments():** Scans the experiment folder for existing experiments.
        **add_experiment(exp: MLExperiment):** Adds a new experiment to the tracker.
        **compare_experiments(experiments=None, show_plots=False):** Compares the experiments.
        **update_experiments_metrics():** Only use to update old versions of experiments from MLExperiment.
        **create_comparison_df(save=True):** Creates a comparison dataframe of the experiments.
        **create_hyperparameters_json(save=True):** Creates a json file of the hyperparameters of the experiments.

    Usage
    -----
    >>> from mango.models.experiment_tracking import MLExperiment, MLTracker
    >>> tracker = MLTracker(experiment_folder="/path/to/experiments")
    >>> tracker.scan_for_experiments()
    >>> experiment = MLExperiment.from_registered_experiment(experiment_path="/path/to/experiment")
    >>> tracker.add_experiment(experiment)
    >>> tracker.compare_experiments()
    >>> tracker.create_comparison_df(save=True)
    >>> tracker.create_hyperparameters_json(save=True)
    """

    def __init__(self, experiment_folder):
        """
        Initializes an instance of the MLTracker class.
        :param experiment_folder: The folder where the experiments are stored.
        :type experiment_folder: str
        """
        self.experiment_folder = experiment_folder
        self.experiments = {}

    @property
    def experiment_folder(self) -> str:
        return self._experiment_folder

    @experiment_folder.setter
    def experiment_folder(self, value):
        self._experiment_folder = value

    @property
    def experiments(self) -> dict:
        return self._experiments

    @experiments.setter
    def experiments(self, value):
        self._experiments = value

    def scan_for_experiments(self):
        """
        Scan the experiment folder for experiments and load them.
        :return: None
        """
        for experiments_folders in os.listdir(self.experiment_folder):
            if os.path.isdir(os.path.join(self.experiment_folder, experiments_folders)):
                try:
                    exp = MLExperiment.from_registered_experiment(
                        os.path.join(self.experiment_folder, experiments_folders)
                    )
                    if not experiments_folders in self._experiments:
                        self._experiments[experiments_folders] = exp
                    else:
                        logging.warning(
                            f"Experiment {experiments_folders} already exists in the tracker. Skipping."
                        )
                except Exception as e:
                    logging.error(f"Could not load experiment {experiments_folders}.")
                    logging.error(e, exc_info=True)
        logging.info(f"Found {len(self._experiments)} experiments.")

    def add_experiment(self, experiment: MLExperiment):
        """
        Add an experiment to the tracker.
        :param experiment: An instance of MLExperiment.
        :type experiment: :class:`mango.models.experiment_tracking.MLExperiment`
        :return: None
        """

        # Make sure exp.name is not in self._experiments.
        exp_folder_name = experiment.register_experiment(self.experiment_folder)
        self._experiments[os.path.basename(exp_folder_name)] = experiment
        logging.info(
            f"Added experiment {exp_folder_name} to the tracker. Current experiments: {len(self._experiments)}."
        )

    def create_plots(
        self, show_plots: bool = False
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Create plots for the experiments. In classification problems, it creates the ROC curve, precision recall curve,
        and feature importance plots.

        In regression problems, it creates the feature importance plot only.

        :param show_plots: If True, it displays the plots.
        :return: figures and axes of the plots if show_plots is False.
        """
        for experiment_name, experiment in self._experiments.items():
            if experiment.problem_type == ProblemType.CLASSIFICATION:
                fig, ax = experiment.plot_roc_curve()
                ax.set_title(experiment_name + "_" + ax.get_title())
                fig.savefig(
                    os.path.join(
                        self.experiment_folder, experiment_name, "roc_curve.png"
                    )
                )
                fig, ax = experiment.plot_precision_recall_curve()
                ax.set_title(experiment_name + "_" + ax.get_title())
                fig.savefig(
                    os.path.join(
                        self.experiment_folder, experiment_name, "precision_recall.png"
                    )
                )
                if show_plots:
                    fig.show()
                fig.close()
            fig, ax = experiment.plot_feature_importance()
            ax.set_title(experiment_name + "_" + ax.get_title())
            fig.savefig(
                os.path.join(
                    self.experiment_folder, experiment_name, "feature_importance.png"
                )
            )
            if show_plots:
                fig.show()
            fig.close()

        return None

    def update_experiments_metrics(self):
        """
        Update the metrics of the experiments. Only use to update old versions of experiments.
        """
        for experiment_name, experiment in self._experiments.items():
            # Make sure metrics are updated.
            json_path = os.path.join(
                self.experiment_folder, experiment_name, "summary.json"
            )
            with open(json_path, "r") as f:
                summary = json.load(f)
            custom_metrics = experiment.metrics
            if experiment.num_classes == 2:
                # Make sure is inserted at the beginning of the dictionary.
                threshold = (
                    experiment.best_threshold_pr_curve
                    if experiment.imbalance
                    else experiment.best_threshold_roc_curve
                )
                custom_metrics = {
                    "best_threshold": {
                        "value": threshold,
                        "train_score": experiment.metrics["train_score"][threshold],
                        "test_score": experiment.metrics["test_score"][threshold],
                    },
                    **custom_metrics,
                }
            summary["results"] = custom_metrics
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            logging.info(f"Updated experiment {experiment_name}.")

    def create_comparison_df(self, save: bool = True) -> pd.DataFrame:
        """
        Create a comparison dataframe.
        :param save: If True, it saves the dataframe to an excel file or csv file if openpyxl is not installed.
        :type save: bool, optional
        :return: A pandas dataframe.
        """
        row_index = []
        metrics_row = []
        for experiment_name, experiment in self.experiments.items():
            metrics = experiment.metrics
            metadata = {
                "experiment_name": experiment_name,
                "description": experiment.description,
                "date": pd.to_datetime(experiment_name.split("_")[0]),
            }
            if experiment.problem_type == ProblemType.CLASSIFICATION:
                metrics = {
                    "train_score": metrics["train_score"][
                        experiment.best_threshold_pr_curve
                    ],
                    "test_score": metrics["test_score"][
                        experiment.best_threshold_pr_curve
                    ],
                }
                metadata["best_threshold"] = (
                    experiment.best_threshold_pr_curve
                    if experiment.imbalance
                    else experiment.best_threshold_roc_curve
                )
            else:
                metrics = {
                    "train_score": metrics["train_score"],
                    "test_score": metrics["test_score"],
                }
            row_index.append(
                {**metadata, **metrics},
            )
            metrics_row.append(metrics)
        # Make a dataframe with multilevel column for the train and test scores which are dictionaries.
        df = pd.DataFrame(row_index).drop(columns=["train_score", "test_score"])
        metrics_train = pd.DataFrame([row["train_score"] for row in metrics_row])
        metrics_test = pd.DataFrame([row["test_score"] for row in metrics_row])
        # Concatenate the dataframes in a way that one from train next from test and so on.
        metrics = pd.DataFrame()
        for col in metrics_train.columns:
            metrics = pd.concat(
                [metrics, metrics_train[col], metrics_test[col]], axis=1
            ).copy()

        metrics.columns = pd.MultiIndex.from_product(
            [metrics_train.columns, ["train", "test"]]
        )
        df = pd.concat([df, metrics], axis=1)
        # Set multilevel index
        df = df.set_index(["experiment_name", "description", "date", "best_threshold"])
        # df = df.reset_index()
        # level_3 must be a subindex of train_score and test_score
        if save:
            try:
                import openpyxl

                df.to_excel(
                    os.path.join(self.experiment_folder, "comparison.xlsx"),
                    index=True,
                )
            except ImportError:
                logging.warning(
                    "Could not import openpyxl. Saving to excel will not work. Will save to csv."
                )
                df.to_csv(
                    os.path.join(self.experiment_folder, "comparison.csv"), index=True
                )
        return df

    def create_hyperparameters_json(self, save: bool = True) -> dict:
        """
        Create a json with the hyperparameters of the experiments.
        :param save: If True, it saves the json to a file.
        :type save: bool, optional
        :return:
        """
        hyperparameters = {}
        for experiment_name, experiment in self.experiments.items():
            with open(
                os.path.join(self.experiment_folder, experiment_name, "summary.json"),
                "r",
            ) as f:
                summary = json.load(f)
                hyperparameters[experiment_name] = summary["model"]["hyperparameters"]
        if save:
            with open(
                os.path.join(self.experiment_folder, "hyperparameters_summary.json"),
                "w",
            ) as f:
                json.dump(hyperparameters, f, indent=4, ensure_ascii=False)
        return hyperparameters
