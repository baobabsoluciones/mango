# SERIES_CONFIGURATION = {
#     "KEY_COLS": ["store", "product"],
#     "TIME_COL": "date",
#     "VALUE_COL": "sales",
#     "AGG_OPERATIONS": {"y": "sum"},
#     "TIME_PERIOD_DESCR": "day",
# }

SERIES_CONFIGURATION = {
    "KEY_COLS": ["airport"],
    "TIME_COL": "Fecha_Vuelo",
    "VALUE_COL": "FIN",
    "AGG_OPERATIONS": {"y": "sum"},
    "TIME_PERIOD_DESCR": "day",
}

timedelta_equivalences = {
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "week": "1w",
    "month": "1mo",
    "quarter": "1q",
    "year": "1y",
}


PARAMETERS_BASE = {
    "quarter": {
        "season_period": 4,
        "trend_window": 5,
        "window_size": "12M",
        "agg": "QS",
        "seasonal_lags": [4],
        "non_seasonal_lags": [1, 2],
        "rolling_window": [4],
        "order": 6,
    },
    "month": {
        "season_period": 12,
        "trend_window": 13,
        "window_size": "12mo",
        "agg": "mo",
        "seasonal_lags": [12],
        "non_seasonal_lags": [1, 2],
        "rolling_window": [6, 12],
        "order": 5,
    },
    "week": {
        "season_period": 52,
        "trend_window": 53,
        "window_size": "4w",
        "agg": "w",
        "seasonal_lags": [4],
        "non_seasonal_lags": [4, 12],
        "rolling_window": [4, 12],
        "order": 4,
    },
    "day": {
        "season_period": [7, 365],
        "trend_window": [7, 7],
        "window_size": "30d",
        "agg": "d",
        "seasonal_lags": [7, 14, 21],
        "non_seasonal_lags": [7, 14, 21],
        "order": 3,
    },
    "hour": {
        "season_period": 24,
        "trend_window": 25,
        "window_size": "48h",
        "agg": "h",
        "seasonal_lags": [24],
        "non_seasonal_lags": [24],
        "order": 2,
    },
    "minute": {
        "season_period": 60,
        "trend_window": 61,
        "window_size": "120m",
        "agg": "m",
        "seasonal_lags": [60],
        "non_seasonal_lags": [60],
        "order": 1,
    },
    "second": {
        "season_period": 60,
        "trend_window": 61,
        "window_size": "300s",
        "agg": "s",
        "seasonal_lags": [60],
        "non_seasonal_lags": [60],
        "order": 0,
    },
}


SERIES_CONFIGURATION["TIME_PERIOD"] = timedelta_equivalences[
    SERIES_CONFIGURATION["TIME_PERIOD_DESCR"]
]

PARAMETERS = PARAMETERS_BASE[SERIES_CONFIGURATION["TIME_PERIOD_DESCR"]]
