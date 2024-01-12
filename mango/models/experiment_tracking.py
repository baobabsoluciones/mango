import json
import logging

import os
import pickle
import shutil
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .metrics import (
    generate_metrics_regression,
    generate_metrics_classification,
)


class ProblemType(Enum):
    """
    Basic enum to represent the problem type.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    # When creating a new one convert to lowercase
    @classmethod
    def _missing_(cls, value: str):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


class ModelLibrary(Enum):
    """
    Basic enum to represent the model library.
    """

    SCIKIT_LEARN = "scikit-learn"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"

def _json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def _clean_hyperparameters(hyperparameters):
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
    description: str = None,
    custom_folder_name: str = None,
    save_model: bool = True,
    save_datasets: bool = False,
    zip_files: bool = True,
):
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
    :param custom_folder_name: Custom name for the folder where the model and datasets will be saved.
    :type custom_folder_name: :class:`str`
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

        _SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.SCIKIT_LEARN] = BaseEstimator
    except ImportError:
        pass
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
    summary["model"] = {}
    summary["model"]["name"] = model_name
    summary["model"]["problem_type"] = problem_type.value
    summary["model"]["target"] = y_train.name
    summary["model"]["library"] = model_library.value
    if model_library == ModelLibrary.CATBOOST:
        summary["model"]["input"] = list(model.feature_names_)
        summary["model"]["hyperparameters"] = model.get_all_params()
    elif model_library == ModelLibrary.SCIKIT_LEARN:
        summary["model"]["input"] = list(model.feature_names_in_)
        summary["model"]["hyperparameters"] = model.get_params(deep=True)
    elif model_library == ModelLibrary.LIGHTGBM:
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
        summary["results"] = {
            "train_score": generate_metrics_classification(
                y_train.reset_index(drop=True), y_train_pred
            ),
            "test_score": generate_metrics_classification(
                y_test.reset_index(drop=True), y_test_pred
            ),
        }
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
    folder_name_default = f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    folder_name = custom_folder_name or folder_name_default
    folder_name = os.path.join(base_path, folder_name)

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


def _find_saving_parameters_from_structure(experiment_folder):
    """
    Find the saving parameters from the structure of the experiment folder.
    :param experiment_folder:
    :return:
    """
    # Walk around the folder and find the files.
    # Should return the following dictionary:
    # {
    #     "save_datasets": True, if inside folder datasets.zip or data/ exist.
    #     "save_model": True, if inside folder model.zip or model/ exist.
    #     "zip_files": True, if inside folder model.zip or data.zip exist.
    # }
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


def _unzip_experiment_folder(experiment_path):
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


def _zip_experiment_folder(experiment_path):
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


class MLExperiment:
    __VALID_MODELS = {
        "sklearn": {
            "regression": "sklearn.linear_model.LinearRegression",
            "classification": "sklearn.linear_model.LogisticRegression",
        },
        "lightgbm": {
            "regression": "lightgbm.LGBMRegressor",
            "classification": "lightgbm.LGBMClassifier",
        },
        "catboost": {
            "regression": "catboost.CatBoostRegressor",
            "classification": "catboost.CatBoostClassifier",
        },
    }

    def __init__(
        self,
        config=None,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        model=None,
        problem_type=None,
        name=None,
        description=None,
    ):
        if config:
            self._config = config
            self._name = self._config("name")
            self._description = self._config("description")
            self._problem_type = self._config("problem_type")
            self._model = self._load_model_from_config(self._config)
            self._X_train = pd.read_csv(self._config("X_train"))
            self._y_train = pd.read_csv(self._config("y_train"))
            self._X_test = pd.read_csv(self._config("X_test"))
            self._y_test = pd.read_csv(self._config("y_test"))
        else:
            self._config = None
            self._name = name
            self._description = description
            self._problem_type = ProblemType(problem_type)
            self._model = model
            assert X_train is not None, "X_train cannot be None."
            self._X_train = X_train
            assert y_train is not None, "y_train cannot be None."
            self._y_train = y_train
            assert X_test is not None, "X_test cannot be None."
            self._X_test = X_test
            assert y_test is not None, "y_test cannot be None."
            self._y_test = y_test

        self._metrics = None
        self._column_transformer = None
        self._base_model_library = None
        self._base_model = None
        self._num_preprocessors = None
        self._num_features = None
        self._cat_preprocessors = None
        self._cat_features = None
        self._model_params = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    @property
    def metrics(self):
        if self._metrics is None:
            logging.warning("Metrics have not been calculated yet. Calculating now.")
            if self._problem_type == ProblemType.REGRESSION:
                self._metrics = generate_metrics_regression(
                    self._y_test, self._model.predict(self._X_test)
                )
            elif self._problem_type == ProblemType.CLASSIFICATION:
                self._metrics = generate_metrics_classification(
                    self._y_test, self._model.predict(self._X_test)
                )
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def _check_model_is_fitted(self):
        """
        Check if the model is fitted.
        :return:
        """
        pass

    def _get_model_from_string(self, model_string):
        """
        Get the model from the string.
        :param model_string:
        :return:
        """
        pass

    def _load_model_from_config(self, config):
        """
        Load the model from the config.
        :param config:
        :return:
        """
        self.base_model_library = ModelLibrary(
            config("model_library")
        )  # This should be a string.
        self.base_model = config(
            "model"
        )  # This should be a class string equal to the class name.
        self.base_model = None

        # This would be strings and we need to somehow convert them to the actual objects.
        self.num_preprocessors = config(
            "numeric_preprocessors"
        )  # This should be a dictionary of classes.
        self.num_features = config("numeric_features")  # This should be a list.
        self.cat_preprocessors = config(
            "categorical_preprocessors"
        )  # This should be a list.
        self.cat_features = config("categorical_features")  # This should be a list.
        self.model_params = config("model_params")  # This should be a dictionary.

        # Create Pipeline from sklearn.
        # Create the numeric pipeline.
        if self.base_model_library == ModelLibrary.SCIKIT_LEARN:
            self._column_transformer = ColumnTransformer(
                transformers=[
                    (
                        "numeric_pipeline",
                        Pipeline(steps=[]),
                        self.num_features,
                    ),
                    (
                        "categorical_pipeline",
                        Pipeline(steps=[]),
                        self.cat_features,
                    ),
                ]
            )

            return Pipeline(
                steps=[
                    ("column_transformer", self._column_transformer),
                    ("model", self.base_model(**self.model_params)),
                ]
            )
        elif self.base_model_library == ModelLibrary.LIGHTGBM:
            pass
        elif self.base_model_library == ModelLibrary.CATBOOST:
            pass
        else:
            raise ValueError(f"{self.base_model_library} is not a valid model library.")

    def register_experiment(
        self, base_path, save_model=True, save_datasets=True, zip_files=True
    ):
        """
        Register the experiment and save it as a zip file.
        :param base_path:
        :param save_model:
        :param save_datasets:
        :param zip_files:
        :return:
        """
        return export_model(
            self.model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            base_path=base_path,
            custom_folder_name=self.name,
            save_model=save_model,
            save_datasets=save_datasets,
            zip_files=zip_files,
        )

    @classmethod
    def from_registered_experiment(cls, experiment_path):
        """
        Load the experiment from a registered experiment.
        :param experiment_path:
        :return:
        """
        # Read files in the folder and load them.
        # Get saving params
        saving_params = _find_saving_parameters_from_structure(experiment_path)
        try:
            if saving_params["zip_files"]:
                _unzip_experiment_folder(experiment_path)
            with open(os.path.join(experiment_path, "summary.json"), "r") as f:
                summary = json.load(f)
            # Set params
            with open(os.path.join(experiment_path, "model", "model.pkl"), "rb") as f:
                model = pickle.load(f)
            if saving_params["save_datasets"]:
                X_train = pd.read_csv(
                    os.path.join(experiment_path, "datasets", "X_train.csv")
                )
                y_train = pd.read_csv(
                    os.path.join(experiment_path, "datasets", "y_train.csv")
                )
                X_test = pd.read_csv(
                    os.path.join(experiment_path, "datasets", "X_test.csv")
                )
                y_test = pd.read_csv(
                    os.path.join(experiment_path, "datasets", "y_test.csv")
                )
            else:
                X_train = None
                y_train = None
                X_test = None
                y_test = None

            experiment = cls(
                name=summary["model"]["name"],
                description=summary["model"].get("description", ""),
                problem_type=ProblemType(summary["model"]["problem_type"]),
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            experiment.metrics = summary["results"]
        except Exception as e:
            raise e
        finally:
            # Raise exception but make sure we zip the folder again if it was unzipped.
            if saving_params["zip_files"]:
                # Zip the folder again.
                _zip_experiment_folder(experiment_path)
        return experiment


class MLTracker:
    def __init__(self, experiment_folder):
        self._experiment_folder = experiment_folder
        self._experiments = {}

    @property
    def experiment_folder(self):
        return self._experiment_folder

    @property
    def experiments(self):
        return self._experiments

    def scan_for_experiments(self):
        """
        Scan the experiment folder for experiments.
        :return:
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

    def add_experiment(self, exp, register=True):
        """
        Add an experiment to the tracker.
        :param exp:
        :return:
        """
        # Make sure exp.name is not in self._experiments.
        if exp.name in self._experiments or exp.name in os.listdir(self.experiment_folder):
            logging.warning("Experiment name already exists. Creating with suffix.")
            for i in range(1, 1000):
                if f"{exp.name} ({i})" not in self._experiments and f"{exp.name} ({i})" not in os.listdir(self.experiment_folder):
                    exp.name = f"{exp.name} ({i})"
                    break
        self._experiments[exp.name] = exp
        logging.info(f"Added experiment {exp.name} to the tracker. Current experiments: {len(self._experiments)}.")
        if register:
            exp.register_experiment(self.experiment_folder)
