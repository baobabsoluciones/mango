import logging
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd


def get_corr_matrix(
    df: pd.DataFrame,
    n_top: int,
    date_col: str = None,
    years_corr: List = None,
    subset=None,
):
    """
    The get_corr_matrix function takes a dataframe and returns the correlation matrix of the top n_top correlated
    columns. The function also allows to specify a date column, which will be used to compute correlations for each year.
    The user can also specify which years should be considered in the computation of correlations.

    :param df: pd.DataFrame: Pass the dataframe to be used in the function
    :param n_top: int: Select the top n correlated variables
    :param date_col: str: Specify the column name of the date
    :param years_corr: List: Specify the years to use for calculating correlations
    :param subset: Select a subset of columns to calculate the correlation matrix
    :param : Get the correlation matrix of a subset of columns
    :return: A dataframe with the correlation matrix
    :doc-author: baobab soluciones
    """
    if not date_col:
        date_col, as_index = get_date_col_candidate(df)
    else:
        as_index = False
    raise_if_inconsistency(df, date_col, as_index)  # Raises error if problems
    if not as_index:
        df = df.set_index(date_col)
    return get_corr_matrix_aux(df, years_corr, n_top, subset)


def get_date_col_candidate(df: pd.DataFrame):
    """
    The get_date_col_candidate function takes a dataframe as an input and returns the name of the column that is
    a datetime type. If there are no columns with datetime types, it will return None. It also returns a boolean value
    that indicates whether or not the index is a datetime type.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :return: A list of columns that have datetime dtypes
    :doc-author: baobab soluciones
    """
    date_column = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    if len(date_column) == 0:
        if isinstance(df.index, pd.DatetimeIndex):
            as_index = True
            date_column = None
            return date_column, as_index
        else:
            as_index = False
            date_column = None
            return date_column, as_index
    else:
        as_index = False
    return date_column, as_index


def raise_if_inconsistency(df: pd.DataFrame, date_col: str, as_index: bool):
    """
    The raise_if_inconsistency function raises a ValueError if the input dataframe is not in the correct format.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :param date_col: str: Specify the name of the column that contains the date
    :param as_index: bool: Check if the dataframe is pivoted or not
    :return: A valueerror if the dataframe is not in the correct format
    :doc-author: baobab soluciones
    """
    if date_col is None and as_index is False:
        raise ValueError("Dataframe must contain one datetime column")
    elif date_col is None and as_index:
        dupli = df.index.duplicated().sum()
        if dupli > 0:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )
            if columns_num == len(df.columns):
                raise ValueError("There are duplicates in the index")
            else:
                data = {
                    "fecha": pd.date_range("2023-01-01", "2023-01-06"),
                    "ventas_loc1": [30, 50, 10, 25, 32, 45],
                    "ventas_loc2": [60, 31, 46, 43, 60, 20],
                }
                example = pd.DataFrame(data)
                raise ValueError(
                    f"Dataframe must be pivot:{print(example.to_markdown())}"
                )
        else:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )

            if columns_num != len(df.columns):
                raise ValueError("Not all columns in Dataframe are numerics")
    elif type(date_col) == list and len(date_col) > 1:
        raise ValueError("Dataframe must contain one datetime column")
    elif type(date_col) == list or type(date_col) == str:
        if type(date_col) == list:
            date_col = date_col[0]
        dupli = df[date_col].duplicated().sum()
        if dupli > 0:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )
            if columns_num == len(df.columns) - 1:
                raise ValueError("There are duplicates in the index")
            else:
                data = {
                    "fecha": pd.date_range("2023-01-01", "2023-01-06"),
                    "ventas_loc1": [30, 50, 10, 25, 32, 45],
                    "ventas_loc2": [60, 31, 46, 43, 60, 20],
                }
                example = pd.DataFrame(data)
                raise ValueError(
                    f"Dataframe must be pivot:{print(example.to_markdown())}"
                )
        else:
            columns_num = sum(
                pd.api.types.is_numeric_dtype(df[col]) for col in df.columns
            )

            if columns_num != len(df.columns) - 1:
                raise ValueError("Not all columns in Dataframe are numerics")


def get_corr_matrix_aux(
    df: pd.DataFrame, years_corr, n_top, subset=None
) -> Dict[str, List[str]]:
    """
    The get_corr_matrix_aux function computes the correlation matrix for a given dataframe.
    It then filters by years and subset, if specified. It returns a dictionary with the top n correlations for each time series in the dataframe.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :param years_corr: Filter the dataframe by year
    :param n_top: Specify the number of top correlations to return for each time series
    :param subset: Filter the dataframe by a subset of time series
    :return: A dictionary with the top n correlations for each time series
    :doc-author: baobab soluciones
    """
    if n_top >= df.shape[1]:
        warnings.warn(
            "Number of n_top is bigger than number of columns of the dataframe"
        )

    if years_corr is not None:
        # Filter by years
        df = df[df.index.year.isin(years_corr)]

    if subset is None:
        logging.debug(
            f"Getting {n_top} top correlations for each time series in the dataframe"
        )
    else:
        logging.debug(
            f"Getting {n_top} top correlations for each time series in the dataframe with respect to the subset"
        )

    # Compute correlation matrix
    correlation_matrix = df.corr(method="pearson")
    # Make sure the correlation with itself is not the highest
    np.fill_diagonal(correlation_matrix.values, -100)

    # Filter by subset
    if subset:
        # Keep only as columns the time series in the subset
        correlation_matrix = correlation_matrix[
            correlation_matrix.columns.intersection(subset)
        ]
        # Drop rows in the subset to avoid comparing with the subset time series
        correlation_matrix = correlation_matrix.drop(index=subset, errors="ignore")

    # Get top n correlations for each time series
    top_correlations = {}
    for column in correlation_matrix.columns:
        top_correlations[column] = (
            correlation_matrix[column].nlargest(n_top).index.tolist()
        )
    return top_correlations
