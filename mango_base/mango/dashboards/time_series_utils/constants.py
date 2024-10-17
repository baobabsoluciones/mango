from typing import Dict, Union

from statsforecast.models import (
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    MSTL,
    AutoTheta,
)

ModelType = Union[HistoricAverage, Naive, RandomWalkWithDrift, MSTL]


# Constants for time aggregation
SELECT_AGR_TMP_DICT = {
    "hourly": "h",
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "QE",
    "yearly": "YE",
}

default_models: Dict[str, ModelType] = {
    "HistoricAverage": HistoricAverage(),
    "Naive": Naive(),
    "RWD": RandomWalkWithDrift(),
    # TODO: Add model parameters based on frequency/granularity
    "MSTL": MSTL(season_length=[7, 365], trend_forecaster=AutoTheta()),
}


# Remove these dictionaries from here and add them to the UI_TEXT files
# DAY_NAME_DICT = {...}
# MONTH_NAME_DICT = {...}
# ALL_DICT = {...}
