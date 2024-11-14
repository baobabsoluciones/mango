import pandas as pd

from mango.logging.logger import get_basic_logger

logger = get_basic_logger()


def create_recent_folds(df: pd.DataFrame, horizon, SERIES_CONF, recent_folds: int = 3):
    """
    Create the recent folds for the time series cross validation

    :param df: pd.DataFrame
    :param horizon: int
    :param SERIES_CONF: dict
    :param recent_folds: int
    :return: pd.DataFrame
    """
    logger.info(
        f"Creating time series recent folds. Horizon: {horizon}. Folds: {recent_folds}"
    )

    if horizon < 28:
        test_len = 28
    else:
        test_len = horizon

    last_date = df["datetime"].max()

    splits = []

    # conditions: first fold is the most recent one. from last_date - test_len to last_date, then last_date - test_len*2 to last_date - test_len
    for i in range(1, recent_folds + 1):
        # test set
        df_temp = df.copy()
        lower_window = last_date - pd.to_timedelta(
            test_len * i, unit=SERIES_CONF["TIME_PERIOD_DESCR"]
        )
        upper_window = last_date - pd.to_timedelta(
            test_len * (i - 1), unit=SERIES_CONF["TIME_PERIOD_DESCR"]
        )

        df_temp_tr = df_temp[(df_temp["datetime"] <= lower_window)]
        df_temp_te = df_temp[
            (df_temp["datetime"] > lower_window) & (df_temp["datetime"] <= upper_window)
        ]

        train_indices = df_temp_tr.index.tolist()
        test_indices = df_temp_te.index.tolist()

        logger.info(
            f"Recent fold: {i}. Train: {df_temp_tr['datetime'].min().strftime('%Y-%m-%d')} > {df_temp_tr['datetime'].max().strftime('%Y-%m-%d')}. Test: {df_temp_te['datetime'].min().strftime('%Y-%m-%d')} > {df_temp_te['datetime'].max().strftime('%Y-%m-%d')}"
        )

        # Store the indices as a tuple in the splits list
        splits.append((train_indices, test_indices))

    return splits


def create_recent_seasonal_folds(
    df: pd.DataFrame, horizon, SERIES_CONF, season_folds: int = 1
):
    """
    Create the seasonal folds for the time series cross validation

    :param df: pd.DataFrame
    :param horizon: int
    :param SERIES_CONF: dict
    :param season_folds: int
    :return: pd.DataFrame
    """
    logger.info(
        f"Creating time series seasonal folds. Horizon: {horizon}. Folds: {season_folds}"
    )

    if horizon < 28:
        test_len = 28
    else:
        test_len = horizon

    last_date = df["datetime"].max()

    splits = []

    # conditions: first fold is the most recent one. from last_date - test_len to last_date, then last_date - test_len*2 to last_date - test_len
    for i in range(1, season_folds + 1):
        # test set

        df_temp = df.copy()
        # lower is last_date - seasonal_offset
        lower_window = last_date - pd.to_timedelta(
            (SERIES_CONF["TS_PARAMETERS"]["seasonal_fold_offset"] * season_folds),
            unit=SERIES_CONF["TIME_PERIOD_DESCR"],
        )
        upper_window = lower_window + pd.to_timedelta(
            test_len, unit=SERIES_CONF["TIME_PERIOD_DESCR"]
        )

        df_temp_tr = df_temp[(df_temp["datetime"] <= lower_window)]
        df_temp_te = df_temp[
            (df_temp["datetime"] > lower_window) & (df_temp["datetime"] <= upper_window)
        ]

        train_indices = df_temp_tr.index.tolist()
        test_indices = df_temp_te.index.tolist()

        logger.info(
            f"Seasonal fold: {i}. Train: {df_temp_tr['datetime'].min().strftime('%Y-%m-%d')} > {df_temp_tr['datetime'].max().strftime('%Y-%m-%d')}. Test: {df_temp_te['datetime'].min().strftime('%Y-%m-%d')} > {df_temp_te['datetime'].max().strftime('%Y-%m-%d')}"
        )

        # Store the indices as a tuple in the splits list
        splits.append((train_indices, test_indices))

    return splits


def get_ts_folds_ids(df, horizon, SERIES_CONF):
    """
    Create the folds for the time series cross validation
    :param df: pd.DataFrame
    :param SERIES_CONF: dict
    :return: pd.DataFrame
    """

    logger.info("Creating time series folds")

    # Create a copy of the dataframe
    df = df.copy()

    ids_recent = create_recent_folds(df, horizon, SERIES_CONF)
    ids_seasonal = create_recent_seasonal_folds(df, horizon, SERIES_CONF)

    # join the two lists
    ids_folds = ids_recent + ids_seasonal

    return ids_folds
