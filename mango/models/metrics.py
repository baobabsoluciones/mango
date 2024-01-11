from typing import Dict

import pandas as pd


# Define the metrics without sklearn
def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the R2 score for regression.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :return: The R2 score.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([3, -0.5, 2, 7])
    >>> y_pred = pd.Series([2.5, 0.0, 2, 8])
    >>> r2_score(y_true, y_pred)
    0.9486
    """
    mean_y_true = y_true.mean()
    ss_tot = ((y_true - mean_y_true) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return 1 - ss_res / ss_tot


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the mean absolute error for regression.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :return: The mean absolute error.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([3, -0.5, 2, 7])
    >>> y_pred = pd.Series([2.5, 0.0, 2, 8])
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    """
    return (y_true - y_pred).abs().mean()


def mean_squared_error(
    y_true: pd.Series, y_pred: pd.Series, squared: bool = True
) -> float:
    """
    Calculate the mean squared error for regression.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :param squared: Whether to return the squared error or not.
    :type squared: :class:`bool`
    :return: The mean squared error.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([3, -0.5, 2, 7])
    >>> y_pred = pd.Series([2.5, 0.0, 2, 8])
    >>> mean_squared_error(y_true, y_pred)
    0.375
    """
    mse = ((y_true - y_pred) ** 2).mean()
    if squared:
        return mse
    else:
        return mse**0.5


def median_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the median absolute error for regression.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :return: The median absolute error.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([3, -0.5, 2, 7])
    >>> y_pred = pd.Series([2.5, 0.0, 2, 8])
    >>> median_absolute_error(y_true, y_pred)
    0.5
    """
    return (y_true - y_pred).abs().median()


def confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Calculate the confusion matrix for classification.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :return: The confusion matrix.
    :rtype: :class:`pandas.DataFrame`

    Usage
    -----
    >>> y_true = pd.Series([0, 1, 1, 0])
    >>> y_pred = pd.Series([0, 0, 1, 1])
    >>> confusion_matrix(y_true, y_pred)
    array([[1, 1],
           [1, 1]])
    """
    return pd.crosstab(y_true, y_pred)


def precision_score(
    y_true: pd.Series, y_pred: pd.Series, average: str = "binary"
) -> float:
    """
    Calculate the precision score for classification.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :param average: The type of averaging performed.
    :type average: :class:`str`
    :return: The precision score.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([0, 1, 1, 0])
    >>> y_pred = pd.Series([0, 0, 1, 1])
    >>> precision_score(y_true, y_pred)
    0.5
    """
    if average == "binary":
        return ((y_true == 1) & (y_pred == 1)).sum() / (y_pred == 1).sum()
    elif average == "macro":
        return (
            ((y_true == 1) & (y_pred == 1)).sum()
            + ((y_true == 0) & (y_pred == 0)).sum()
        ) / len(y_true)
    else:
        raise ValueError(f"{average} is not a valid value for average.")


def recall_score(
    y_true: pd.Series, y_pred: pd.Series, average: str = "binary"
) -> float:
    """
    Calculate the recall score for classification.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :param average: The type of averaging performed.
    :type average: :class:`str`
    :return: The recall score.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([0, 1, 1, 0])
    >>> y_pred = pd.Series([0, 0, 1, 1])
    >>> recall_score(y_true, y_pred)
    0.5
    """
    if average == "binary":
        return ((y_true == 1) & (y_pred == 1)).sum() / (y_true == 1).sum()
    elif average == "macro":
        return (
            ((y_true == 1) & (y_pred == 1)).sum()
            + ((y_true == 0) & (y_pred == 0)).sum()
        ) / len(y_true)
    else:
        raise ValueError(f"{average} is not a valid value for average.")


def f1_score(y_true: pd.Series, y_pred: pd.Series, average: str = "binary") -> float:
    """
    Calculate the F1 score for classification.

    :param y_true: The true values.
    :type y_true: :class:`pandas.Series`
    :param y_pred: The predicted values.
    :type y_pred: :class:`pandas.Series`
    :param average: The type of averaging performed.
    :type average: :class:`str`
    :return: The F1 score.
    :rtype: :class:`float`

    Usage
    -----
    >>> y_true = pd.Series([0, 1, 1, 0])
    >>> y_pred = pd.Series([0, 0, 1, 1])
    >>> f1_score(y_true, y_pred)
    0.5
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return 2 * (precision * recall) / (precision + recall)


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
    Generate common metrics for classification and return them in a dictionary. The metrics for binary classification are:
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
