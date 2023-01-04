import numpy as np
import pandas as pd
from typing import List


def create_dense_data(
    df,
    id_cols,
    freq: str,
    min_max_by_id: bool = None,
    date_init=None,
    date_end=None,
    time_col: str = "timeslot",
):
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
