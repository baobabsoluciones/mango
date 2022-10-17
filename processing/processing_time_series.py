import pandas as pd


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
