import inspect
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
    "MSTL": MSTL(season_length=[7, 365], trend_forecaster=AutoTheta()),
}

model_context = []
all_imports = set()

for name, model_instance in default_models.items():
    parameters = {}
    for param_name, param_value in model_instance.__dict__.items():
        if (
            inspect.isclass(type(param_value))
            and param_value.__class__.__module__ == "statsforecast.models"
        ):
            all_imports.add(param_value.__class__.__name__)
        parameters[param_name] = param_value

    model_context.append(
        {
            "name": name,
            "class_name": model_instance.__class__.__name__,
            "parameters": parameters,
        }
    )
    all_imports.add(model_instance.__class__.__name__)

all_imports = list(all_imports)
