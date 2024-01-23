from enum import Enum


class ProblemType(Enum):
    """
    Enum to represent the problem type.
    """

    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    # When creating a new one convert to lowercase
    @classmethod
    def _missing_(cls, value: str):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


class ModelLibrary(Enum):
    """
    Enum to represent the model library.
    """

    SCIKIT_LEARN = "scikit-learn"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
