import json
import os
import pickle
import shutil
from datetime import datetime
from enum import Enum
from typing import Dict, Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    confusion_matrix,
    precision_score,
    f1_score,
    recall_score,
)


class ProblemType(Enum):
    """
    Basic enum to represent the problem type.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class ModelLibrary(Enum):
    """
    Basic enum to represent the model library.
    """

    SCIKIT_LEARN = "scikit-learn"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"


def generate_metrics_regression(
    y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, float]:
    """
    Generate common metrics for regression and return them in a dictionary. The metrics are:
        - R2 score
        - Mean absolute error
        - Mean squared error
        - Root mean squared error
        - Median absolute error

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :return: A dictionary of metrics.
    :rtype: :class:`dict`

    Usage
    -----
    >>> y_true = pd.Series([3, -0.5, 2, 7])
    >>> y_pred = pd.Series([2.5, 0.0, 2, 8])
    >>> metrics = generate_metrics_regression(y_true, y_pred)
    >>> print(metrics)
    {'r2_score': 0.9486, 'mean_absolute_error': 0.5, 'mean_squared_error': 0.375, 'root_mean_squared_error': 0.6124, 'median_absolute_error': 0.5}
    """
    return {
        "r2_score": round(r2_score(y_true, y_pred), 4),
        "mean_absolute_error": round(mean_absolute_error(y_true, y_pred), 4),
        "mean_squared_error": round(mean_squared_error(y_true, y_pred), 4),
        "root_mean_squared_error": round(
            mean_squared_error(y_true, y_pred, squared=False), 4
        ),
        "median_absolute_error": round(median_absolute_error(y_true, y_pred), 4),
    }


def generate_metrics_classification(
    y_true: pd.Series, y_pred: pd.Series
) -> Dict[str, float]:
    """
    Generate common metrics for classification and return them in a dictionary. The metrics for binary classification
    are:
        - Confusion matrix
        - Accuracy
        - Precision
        - Recall
        - F1 score

    In case It is a multiclass classification, the metrics are:
        - Confusion matrix
        - Accuracy
        - Precision macro
        - Recall macro
        - F1 score macro

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :return: A dictionary of metrics.
    :rtype: :class:`dict`

    Usage
    -----
    >>> y_true = pd.Series([0, 1, 1, 0])
    >>> y_pred = pd.Series([0, 0, 1, 1])
    >>> metrics = generate_metrics_classification(y_true, y_pred)
    >>> print(metrics)
    {'confusion_matrix': [[1, 1], [1, 1]], 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5}
    """
    if len(y_true.unique()) == 2:
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "accuracy": round((y_true == y_pred).sum() / len(y_true), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
        }
    else:
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "accuracy": round((y_true == y_pred).sum() / len(y_true), 4),
            "precision_macro": round(
                precision_score(y_true, y_pred, average="macro"), 4
            ),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4),
            "f1_score_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        }


def export_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_path: str,
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
    :param base_path: Path to the base folder where the model and datasets will be saved in a subfolder structure.
    :type base_path: :class:`str`
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
    _SUPPORTED_LIBRARIES_CLASSES[ModelLibrary.SCIKIT_LEARN] = BaseEstimator
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
    folder_name = os.path.join(
        base_path,
        f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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
        os.makedirs(os.path.join(folder_name, "data"))
        if not "files" in summary:
            summary["files"] = {}
        if not "data" in summary["files"]:
            summary["files"]["data"] = {}
        X_train_path = os.path.join(folder_name, "data", "X_train.csv")
        summary["files"]["data"]["X_train"] = {}
        summary["files"]["data"]["X_train"]["path"] = os.path.abspath(X_train_path)
        summary["files"]["data"]["X_train"]["shape"] = X_train.shape
        X_train.to_csv(X_train_path, index=False)
        y_train_path = os.path.join(folder_name, "data", "y_train.csv")
        summary["files"]["data"]["y_train"] = {}
        summary["files"]["data"]["y_train"]["path"] = os.path.abspath(y_train_path)
        summary["files"]["data"]["y_train"]["shape"] = y_train.shape
        y_train.to_csv(y_train_path, index=False)
        X_test_path = os.path.join(folder_name, "data", "X_test.csv")
        summary["files"]["data"]["X_test"] = {}
        summary["files"]["data"]["X_test"]["path"] = os.path.abspath(X_test_path)
        summary["files"]["data"]["X_test"]["shape"] = X_test.shape
        X_test.to_csv(X_test_path, index=False)
        y_test_path = os.path.join(folder_name, "data", "y_test.csv")
        summary["files"]["data"]["y_test"] = {}
        summary["files"]["data"]["y_test"]["path"] = os.path.abspath(y_test_path)
        summary["files"]["data"]["y_test"]["shape"] = y_test.shape
        y_test.to_csv(y_test_path, index=False)
        if zip_files:
            # Compress data and save
            zip_path = os.path.join(folder_name, "data.zip")
            summary["files"]["data"]["zip"] = {}
            summary["files"]["data"]["zip"]["path"] = os.path.abspath(zip_path)
            shutil.make_archive(
                zip_path.rstrip(".zip"), "zip", os.path.join(folder_name, "data")
            )
            shutil.rmtree(os.path.join(folder_name, "data"))

    # Save json
    json_path = os.path.join(folder_name, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    return folder_name
