from typing import List, Dict, Any

from mango_time_series.utils import timedelta_equivalences, PARAMETERS_BASE


class SeriesConfiguration:
    """
    Configuration class for time series
    :param key_cols: list of key columns
    :param time_period_descr: time period description
    :param recent_folds: number of recent folds
    :param seasonal_folds: number of seasonal folds
    :param agg_operations: aggregation operations
    :param time_col: time column
    :param value_col: value column
    """

    def __init__(
        self,
        key_cols: List[str],
        time_period_descr: str,
        recent_folds: int,
        seasonal_folds: int,
        agg_operations: Dict[str, Any],
        time_col: str,
        value_col: str,
    ):
        self.KEY_COLS: List[str] = key_cols
        self.TIME_PERIOD_DESCR: str = time_period_descr
        self.TIME_PERIOD_PD: str = timedelta_equivalences[time_period_descr]
        self.TS_PARAMETERS: Dict[str, Any] = PARAMETERS_BASE[time_period_descr]
        self.RECENT_FOLDS: int = recent_folds
        self.SEASONAL_FOLDS: int = seasonal_folds
        self.AGG_OPERATIONS: Dict[str, Any] = agg_operations
        self.TIME_COL: str = time_col
        self.VALUE_COL: str = value_col

    def __repr__(self):
        return (
            f"SeriesConfiguration(KEY_COLS={self.KEY_COLS}, TIME_PERIOD_DESCR={self.TIME_PERIOD_DESCR}, "
            f"TIME_PERIOD_PD={self.TIME_PERIOD_PD}, TS_PARAMETERS={self.TS_PARAMETERS}, "
            f"RECENT_FOLDS={self.RECENT_FOLDS}, SEASONAL_FOLDS={self.SEASONAL_FOLDS}, "
            f"AGG_OPERATIONS={self.AGG_OPERATIONS}, TIME_COL={self.TIME_COL}, VALUE_COL={self.VALUE_COL})"
        )
