import logging
import warnings

import numpy as np
from typing import List, Dict

try:
    import pandas as pd
except ImportError:
    pd = None


def create_dense_data(
    df: pd.DataFrame,
    id_cols,
    freq: str,
    min_max_by_id: bool = None,
    date_init=None,
    date_end=None,
    time_col: str = "timeslot",
) -> pd.DataFrame:
    """
    Create a dense dataframe with a frequency of freq, given range of dates or inherited from the dataframe,
     using the id_cols as keys.
    :param df: dataframe to be expanded
    :param id_cols: list of columns to be used as keys
    :param freq: frequency of the new dataframe
    :param min_max_by_id: boolean to indicate if the range of dates is the min and max of the dataframe by id
    :param date_init: if it has a value, all initial dates will be set to this value
    :param date_end: if it has a value, all final dates will be set to this value
    :param time_col: string with name of the column with the time information
    :return: dataframe with all the dates using the frequency freq
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas need to be installed to use this function")
    df_w = df.copy()

    # get cols id_cols from df and drop duplicates
    df_id = df_w[id_cols].drop_duplicates()

    # if min_max_by_id is True, get the min and max of the time_col by id_cols
    if min_max_by_id:
        df_min_max = (
            df_w.groupby(id_cols, dropna=False)
            .agg({time_col: ["min", "max"]})
            .reset_index()
        )
        df_min_max.columns = id_cols + ["min_date", "max_date"]

        if date_init is not None:
            df_min_max["min_date"] = date_init

        if date_end is not None:
            df_min_max["max_date"] = date_end

        grid_min_date = df_min_max["min_date"].min()
        grid_max_date = df_min_max["max_date"].max()

    else:
        if date_init is not None:
            grid_min_date = date_init
        else:
            grid_min_date = df_w[time_col].min()

        if date_end is not None:
            grid_max_date = date_end
        else:
            grid_max_date = df_w[time_col].max()

    # create dataframe with column timeslot from grid_min_date to grid_max_date
    df_timeslots = pd.DataFrame(
        {time_col: pd.date_range(grid_min_date, grid_max_date, freq=freq)}
    )
    df_timeslots["key"] = 1

    # create dataframe with all possible combinations of id_cols
    df_id["key"] = 1
    df_grid = df_timeslots.merge(df_id, on="key", how="outer").drop("key", axis=1)

    # filter registers in df_grid using the min_date and max_date by id_cols
    if min_max_by_id:
        df_grid = df_grid.merge(df_min_max, on=id_cols, how="left")
        df_grid = df_grid[
            (df_grid[time_col] >= df_grid["min_date"])
            & (df_grid[time_col] <= df_grid["max_date"])
        ]
        df_grid = df_grid.drop(["min_date", "max_date"], axis=1)

    # merge df_grid with df_w
    df_w = df_grid.merge(df_w, on=id_cols + [time_col], how="left")

    return df_w


def create_lags_col(
    df: pd.DataFrame, col: str, lags: List[int], check_col: List[str] = None
) -> pd.DataFrame:
    """
    The create_lags_col function creates lagged columns for a given dataframe.
    The function takes three arguments: df, col, and lags. The df argument is the
    dataframe to which we want to add lagged columns. The col argument is the name of
    the column in the dataframe that we want to create lag variables for (e.g., 'sales').
    The lags argument should be a list of integers representing how far back in time we
    want to shift our new lag features (e.g., [3, 6] would create two new lag features).

    :param pd.DataFrame df: Pass the dataframe to be manipulated
    :param str col: Specify which column to create lags for
    :param list[int] lags: Specify the lags that should be created
    :param list[str] check_col: Check if the value of a column is equal to the previous or next value
    :return: A dataframe with the lagged columns
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas need to be installed to use this function")

    df_c = df.copy()

    for i in lags:
        if i > 0:
            name_col = col + "_lag" + str(abs(i))
            df_c[name_col] = df_c[col].shift(i)
            if check_col is None:
                continue
            if type(check_col) is list:
                for c in check_col:
                    df_c.loc[df_c[c] != df_c[c].shift(i), name_col] = np.nan
            else:
                df_c.loc[df_c[check_col] != df_c[check_col].shift(i), name_col] = np.nan

        if i < 0:
            name_col = col + "_lead" + str(abs(i))
            df_c[name_col] = df_c[col].shift(i)
            if check_col is None:
                continue
            if type(check_col) is list:
                for c in check_col:
                    df_c.loc[df_c[c] != df_c[c].shift(i), name_col] = np.nan
            else:
                df_c.loc[df_c[check_col] != df_c[check_col].shift(i), name_col] = np.nan

    return df_c


def create_recurrent_dataset(
    data: np.array,
    look_back: int,
    include_output_lags: bool = False,
    lags: List[int] = None,
    output_last: bool = True,
):
    """
    The create_recurrent_dataset function creates a dataset for recurrent neural networks.
    The function takes in an array of data, and returns two arrays: one containing the input data,
    and another containing the output labels. The input is a 2D array with shape (num_samples, num_features).
    The input data output is a 3D array with shape (num_samples, look_back, num_features), while the labels output
    have a 1D array with shape (num_samples, ).

    The function allows to include the output lags in the input output data. If include_output_lags is True,
    the function will create the lags indicated on the lags' argument.

    The function allows for the label to be the first "column" of the input data, or the last "column" of the input data
    by setting the output_last argument to False or True, respectively.

    :param :class:`np.array` data: pass the data to be used for training
    :param int look_back: define the number of previous time steps to use as input variables
    to predict the next time period
    :param bool include_output_lags: decide whether the output lags should be included in the input data
    :param lags:sSpecify which lags should be included in the input data
    :param output_last: indicate if the label column is the first or last one in the original data
    :return: A tuple of numpy arrays: (input_data, labels)
    :rtype: tuple
    :doc-author: baobab soluciones
    """
    x, y = [], []
    if output_last:
        x_in = data[:, :-1]
        y_in = data[:, -1:]
    else:
        x_in = data[:, 1:]
        y_in = data[:, :1]

    if lags is None or not include_output_lags:
        max_lag = 0
    else:
        max_lag = max(lags)

    for i in range(max_lag, data.shape[0] - look_back):
        a = x_in[i : (i + look_back), :]

        if include_output_lags:
            lagged = np.empty((look_back, 1))
            for lag in lags:
                lagged = np.concatenate(
                    (
                        lagged,
                        y_in[i - lag : (i + look_back - lag)].reshape((look_back, 1)),
                    ),
                    axis=1,
                )
            lagged = lagged[:, 1:]

            x.append(np.concatenate((a, lagged), axis=1))
        else:
            x.append(a)

        if output_last:
            y.append(y_in[i + look_back])
        else:
            y.append(y_in[i + look_back])

    return np.array(x), np.array(y).reshape((data.shape[0] - look_back - max_lag,))


def get_corr_matrix(
    df: pd.DataFrame,
    n_top: int = None,
    threshold: int = None,
    date_col: str = None,
    years_corr: List = None,
    subset: List = None,
):
    """
    The get_corr_matrix function takes a dataframe and returns the correlation matrix of the columns.

    :param df: pd.DataFrame: Pass in the dataframe that we want to get the correlation matrix for
    :param n_top: int: Select the top n correlated variables
    :param threshold: int: Filter the correlation matrix by a threshold value
    :param date_col: str: Specify the name of the column that contains dates
    :param years_corr: List: Specify the years for which we want to calculate the correlation matrix
    :param subset: List: Specify a subset of columns to be used in the correlation matrix
    :param : Specify the number of top correlated variables to be returned
    :return: A correlation matrix of the dataframe
    :doc-author: baobab soluciones
    """
    if not date_col:
        date_col, as_index = get_date_col_candidate(df)
    else:
        as_index = False
    raise_if_inconsistency(df, date_col, as_index)  # Raises error if problems
    if not as_index:
        df = df.set_index(date_col)
    return get_corr_matrix_aux(df, years_corr, n_top, threshold, subset)


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
                example = pd.DataFrame(data).set_index("fecha")
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
                example = pd.DataFrame(data).set_index("fecha")
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
    df: pd.DataFrame,
    years_corr,
    n_top,
    threshold,
    subset: List = None,
) -> Dict[str, Dict[str, float]]:
    """
    The get_corr_matrix_aux function computes the correlation matrix of a dataframe and returns
    a dictionary with the top n correlations for each time series.
    The function can also filter by years, subset and threshold.

    :param df: pd.DataFrame: Pass the dataframe to the function
    :param years_corr: Filter the dataframe by year
    :param n_top: Get the top n correlations for each time series
    :param threshold: Filter the correlation matrix by a threshold value
    :param subset: List: Specify a subset of time series to compare the correlations with
    :param : Filter the dataframe by years
    :return: A dictionary with the names of the time series as keys and a list of tuples (name, correlation) as values
    :doc-author: baobab soluciones
    """
    if n_top is not None and n_top >= df.shape[1]:
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
    if threshold is not None:
        for column in correlation_matrix.columns:
            name_correlations = correlation_matrix[column][
                correlation_matrix[column] > threshold
            ]
            if name_correlations.empty:
                warnings.warn(
                    "There are no rows that have a value greater than threshold, so it returns all rows"
                )
                top_correlations[column] = dict(correlation_matrix[column])

            else:
                name_correlations = dict(name_correlations)
                top_correlations[column] = name_correlations

    elif n_top is not None:
        for column in correlation_matrix.columns:
            name_correlations = dict(correlation_matrix[column].nlargest(n_top))
            top_correlations[column] = {
                index: value for index, value in name_correlations.items()
            }
    else:
        for column in correlation_matrix.columns:
            name_correlations = dict(correlation_matrix[column])
            top_correlations[column] = name_correlations
        warnings.warn(
            "n_top and threshold are None so top_correlations return all the correlations"
        )

    return top_correlations
