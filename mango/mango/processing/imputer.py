from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import polars as pl
from pandas.core.generic import InterpolateOptions
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression


class Imputer:
    def __init__(
        self,
        strategy="mean",
        k_neighbors: Optional[int] = None,
        arbitrary_value: Optional[float] = None,
        column_strategies: Optional[Dict[str, str]] = None,
        regression_model: Optional[str] = "ridge",
        regression_params: Optional[Dict] = None,
        knn_params: Optional[Dict] = None,
        simple_params: Optional[Dict] = None,
        iterative_params: Optional[Dict] = None,
        time_series_strategy: Optional[InterpolateOptions] = "linear",
    ):
        """
        Imputer class to fill missing values in a dataset.
        :param strategy: strategy to fill missing values. It can be one of "mean", "median", "mode" or "arbitrary".
        :type strategy: str
        :param k_neighbors: number of neighbors to use for KNN imputation. Required if strategy is "knn".
        :type k_neighbors: Optional[int]
        :param arbitrary_value: value to fill missing values if strategy is "arbitrary"
        :type arbitrary_value: Optional[float]
        :param column_strategies: dictionary containing column names and their respective strategies. If None, global strategy will be used.
        :type column_strategies: Optional[Dict[str,str]]
        :param regression_model: regression model to use for regression imputation. It can be "ridge" or "linear".
        :type regression_model: Optional[str]
        :param regression_params: parameters for the regression model.
        :type regression_params: Optional[Dict]
        :param knn_params: parameters for the KNN imputer.
        :type knn_params: Optional[Dict]
        :param simple_params: parameters for the SimpleImputer.
        :type simple_params: Optional[Dict]
        :param time_series_strategy: strategy to use for time series interpolation. It can be "linear" or "polynomial".
        :type time_series_strategy: Optional[InterpolateOptions]
        """
        self.strategy = strategy
        self.k_neighbors = k_neighbors
        self.arbitrary_value = arbitrary_value
        self.column_strategies = column_strategies
        self.regression_model = regression_model
        self.regression_params = regression_params or {}
        self.knn_params = knn_params or {}
        self.simple_params = simple_params or {}
        self.iterative_params = iterative_params or {}
        self.time_series_strategy = time_series_strategy

        strategy_methods = {
            "mean": self._simple_impute,
            "median": self._simple_impute,
            "mode": self._mode_impute,
            "most_frequent": self._simple_impute,
            "knn": self._knn_impute,
            "mice": self._mice_impute,
            "arbitrary": self._arbitrary_impute,
            "forward": self._fill_impute,
            "backward": self._fill_impute,
            "regression": self._regression_impute,
            "interpolate": self._interpolate_impute,
        }

        self.strategy_methods = strategy_methods

    @staticmethod
    def _convert_to_numpy(data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Convert pandas or polars DataFrame to numpy array and detect data type.

        :param data: Input data (pandas or polars DataFrame)
        :return: Tuple containing (numpy array, data type as string, column names)
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(), "pandas", data.columns
        elif isinstance(data, pl.DataFrame):
            return data.to_numpy(), "polars", data.columns
        else:
            raise ValueError("Input data must be a pandas or polars DataFrame.")

    @staticmethod
    def _convert_back(data, data_type, columns):
        """
        Convert numpy array back to pandas or polars DataFrame.
        :param data: Input data
        :type data: np.ndarray
        :param data_type: Data type as string
        :type data_type: str
        :param columns: Column names
        :type columns: list
        :return: pandas or polars DataFrame
        """
        if data_type == "pandas":
            return pd.DataFrame(data, columns=columns)
        elif data_type == "polars":
            return pl.DataFrame(data, schema=columns)

    def fit_transform(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Fit and transform the data to fill missing values.

        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        """
        if self.column_strategies:
            return self._apply_column_wise_imputation(data)
        else:
            return self._apply_global_imputation(data)

    def _apply_global_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply a global imputation strategy to the entire dataset.
        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        """
        data_array, data_type, columns = self._convert_to_numpy(data)

        if self.strategy not in self.strategy_methods:
            raise ValueError(f"Invalid strategy '{self.strategy}'.")

        imputed_data = self.strategy_methods[self.strategy](data_array)

        return self._convert_back(imputed_data, data_type, columns)

    def _apply_column_wise_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply different imputation strategies to different columns.
        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        """
        data_type = "pandas" if isinstance(data, pd.DataFrame) else "polars"

        if data_type == "pandas":
            df = data.copy()
        else:
            df = data.to_pandas().copy()  # Convert polars to pandas for processing

        for column, strategy in self.column_strategies.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset.")

            if strategy not in self.strategy_methods:
                raise ValueError(
                    f"Invalid strategy '{strategy}' for column '{column}'."
                )

            df[[column]] = self.strategy_methods[strategy](df[[column]].to_numpy())

        return df if data_type == "pandas" else pl.from_pandas(df)

    def _simple_impute(self, data_array: np.ndarray):
        """
        Impute missing values using SimpleImputer.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        """
        imputer = SimpleImputer(strategy=self.strategy, **self.simple_params)
        return imputer.fit_transform(data_array)

    def _knn_impute(self, data_array: np.ndarray):
        """
        Impute missing values using KNN.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        """
        if self.k_neighbors is None:
            raise ValueError("k_neighbors must be specified for KNN imputation.")

        imputer = KNNImputer(n_neighbors=self.k_neighbors, **self.knn_params)
        return imputer.fit_transform(data_array)

    def _mice_impute(self, data_array: np.ndarray):
        """
        Impute missing values using MICE (IterativeImputer).

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        """
        imputer = IterativeImputer(**self.iterative_params)
        return imputer.fit_transform(data_array)

    def _arbitrary_impute(self, data_array: np.ndarray):
        """
        Impute missing values with an arbitrary value.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        """
        if self.arbitrary_value is None:
            raise ValueError(
                "An arbitrary_value must be provided for arbitrary imputation."
            )

        imputer = SimpleImputer(
            strategy="constant", fill_value=self.arbitrary_value, **self.simple_params
        )
        return imputer.fit_transform(data_array)

    def _fill_impute(self, data_array: np.ndarray):
        """
        Fill missing values using forward or backward fill.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        """
        df = pd.DataFrame(data_array)
        if self.strategy == "forward":
            return df.ffill().to_numpy()
        elif self.strategy == "backward":
            return df.bfill().to_numpy()
        else:
            raise ValueError("Invalid fill strategy. Use 'forward' or 'backward'.")

    def _regression_impute(self, data_array: np.ndarray):
        df = pd.DataFrame(data_array)
        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue

            train_data = df.dropna()
            test_data = df[df[column].isnull()].drop(columns=[column])

            if train_data.empty or test_data.empty:
                continue

            X_train = train_data.drop(columns=[column])
            y_train = train_data[column]

            model_mapping = {"ridge": Ridge, "lasso": Lasso, "linear": LinearRegression}

            if self.regression_model not in model_mapping:
                raise ValueError(
                    "Invalid regression model. Choose from 'ridge', 'lasso', 'linear'."
                )

            model = model_mapping[self.regression_model](**self.regression_params)
            model.fit(X_train, y_train)
            df.loc[df[column].isnull(), column] = model.predict(test_data)

        return df.to_numpy()

    def _interpolate_impute(self, data_array: np.ndarray):
        df = pd.DataFrame(data_array)
        return df.interpolate(method=self.time_series_strategy).to_numpy()

    @staticmethod
    def _mode_impute(data_array):
        df = pd.DataFrame(data_array)
        return df.fillna(df.mode().iloc[0]).to_numpy()
