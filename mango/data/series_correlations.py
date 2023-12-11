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
    if not date_col:
        date_col, as_index = get_date_col_candidate(df)
    else:
        as_index = False
    raise_if_inconsistency(df, date_col, as_index)  # Raises error if problems
    if not as_index:
        df = df.set_index(date_col)
    return get_corr_matrix_aux(df, years_corr, n_top, subset)


def get_date_col_candidate(df: pd.DataFrame):
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
