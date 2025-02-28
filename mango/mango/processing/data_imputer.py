from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import polars as pl
from pandas.core.generic import InterpolateOptions
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression


class DataImputer:
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

        This class offers various imputation strategies through different libraries:

        - Statistical imputation (mean, median, mode): Uses ``sklearn.impute.SimpleImputer`` and pandas
        - KNN imputation: Uses ``sklearn.impute.KNNImputer``
        - MICE imputation: Uses ``sklearn.impute.IterativeImputer``
        - Regression imputation: Uses ``sklearn.linear_model`` (Ridge, Lasso, LinearRegression)
        - Time series imputation (forward fill, backward fill, interpolation): Uses pandas methods
        - Arbitrary value imputation: Uses ``sklearn.impute.SimpleImputer`` with constant strategy

        :param strategy: strategy to fill missing values. It can be one of "mean", "median", "mode" or "arbitrary"
        :type strategy: str
        :param k_neighbors: number of neighbors to use for KNN imputation. Required if strategy is "knn"
        :type k_neighbors: Optional[int]
        :param arbitrary_value: value to fill missing values if strategy is "arbitrary"
        :type arbitrary_value: Optional[float]
        :param column_strategies: dictionary containing column names and their respective strategies. If None, global strategy will be used
        :type column_strategies: Optional[Dict[str,str]]
        :param regression_model: regression model to use for regression imputation. It can be "ridge" or "linear"
        :type regression_model: Optional[str]
        :param regression_params: parameters for the regression model
        :type regression_params: Optional[Dict]
        :param knn_params: parameters for the KNN imputer
        :type knn_params: Optional[Dict]
        :param simple_params: parameters for the SimpleImputer
        :type simple_params: Optional[Dict]
        :param iterative_params: parameters for the IterativeImputer
        :type iterative_params: Optional[Dict]
        :param time_series_strategy: strategy to use for time series interpolation. It can be "linear" or "polynomial"
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

        self.strategy_methods = {
            "mean": self._simple_impute,
            "median": self._simple_impute,
            "most_frequent": self._simple_impute,
            "knn": self._knn_impute,
            "mice": self._mice_impute,
            "arbitrary": self._arbitrary_impute,
            "forward": self._fill_impute,
            "backward": self._fill_impute,
            "regression": self._regression_impute,
            "interpolate": self._interpolate_impute,
        }

    @staticmethod
    def _convert_to_numpy(data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Convert pandas or polars DataFrame to numpy array and detect data type.

        Uses ``pandas`` or ``polars`` depending on input type, and converts to ``numpy`` array.

        :param data: Input data (pandas or polars DataFrame)
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Tuple containing (numpy array, data type as string, column names)
        :rtype: tuple
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

        Uses ``pandas.DataFrame`` or ``polars.DataFrame`` depending on the original data type.

        :param data: Input data
        :type data: np.ndarray
        :param data_type: Data type as string
        :type data_type: str
        :param columns: Column names
        :type columns: list
        :return: pandas or polars DataFrame
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        columns = list(columns)

        if data_type == "pandas":
            return pd.DataFrame(data, columns=columns)
        elif data_type == "polars":
            return pl.DataFrame(data, schema=columns)

    def apply_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Fit and transform the data to fill missing values.

        This is the main method to apply imputation to a dataset, supporting both pandas and polars
        DataFrames as input and output.

        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        if self.column_strategies:
            return self._apply_column_wise_imputation(data)
        else:
            return self._apply_global_imputation(data)

    def _apply_global_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply a global imputation strategy to the entire dataset.

        Handles conversion between pandas/polars and numpy formats.

        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        data_array, data_type, columns = self._convert_to_numpy(data)

        if self.strategy not in self.strategy_methods:
            raise ValueError(f"Invalid strategy '{self.strategy}'.")

        imputed_data = self.strategy_methods[self.strategy](data_array)

        return self._convert_back(imputed_data, data_type, columns)

    def _apply_column_wise_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply different imputation strategies to different columns.

        This method allows for fine-grained control of imputation by column.
        Converts polars DataFrame to pandas for processing if needed.

        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        data_type = "pandas" if isinstance(data, pd.DataFrame) else "polars"

        if data_type == "pandas":
            df = data.copy()
        else:
            df = data.to_pandas().copy()

        for column, strategy in self.column_strategies.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset.")

            if strategy not in self.strategy_methods:
                raise ValueError(
                    f"Invalid strategy '{strategy}' for column '{column}'."
                )
            self.strategy = strategy
            df[[column]] = self.strategy_methods[self.strategy](df[[column]].to_numpy())

        return self._convert_back(df.to_numpy(), data_type, df.columns)

    def _simple_impute(self, data_array: np.ndarray):
        """
        Impute missing values using SimpleImputer.

        Uses ``sklearn.impute.SimpleImputer`` with the specified strategy (mean, median, or most_frequent).

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        imputer = SimpleImputer(strategy=self.strategy, **self.simple_params)
        return imputer.fit_transform(data_array)

    def _knn_impute(self, data_array: np.ndarray):
        """
        Impute missing values using KNN.

        Uses ``sklearn.impute.KNNImputer`` to impute missing values based on k-nearest neighbors.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        if self.k_neighbors is None or self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be specified for KNN imputation.")

        imputer = KNNImputer(n_neighbors=self.k_neighbors, **self.knn_params)
        return imputer.fit_transform(data_array)

    def _mice_impute(self, data_array: np.ndarray):
        """
        Impute missing values using MICE (IterativeImputer).

        Uses ``sklearn.impute.IterativeImputer`` which requires importing
        ``sklearn.experimental.enable_iterative_imputer`` first.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        imputer = IterativeImputer(**self.iterative_params)
        return imputer.fit_transform(data_array)

    def _arbitrary_impute(self, data_array: np.ndarray):
        """
        Impute missing values with an arbitrary value.

        Uses ``sklearn.impute.SimpleImputer`` with strategy='constant' and the specified fill_value.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
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

        Uses ``pandas.DataFrame.ffill()`` or ``pandas.DataFrame.bfill()`` methods.

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        df = pd.DataFrame(data_array)
        if self.strategy == "forward":
            return df.ffill().to_numpy()
        elif self.strategy == "backward":
            return df.bfill().to_numpy()

    def _regression_impute(self, data_array: np.ndarray):
        """
        Impute missing values using regression models.

        Uses models from ``sklearn.linear_model``:
        - ``Ridge`` for ridge regression
        - ``Lasso`` for lasso regression
        - ``LinearRegression`` for standard linear regression

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        df = pd.DataFrame(data_array)
        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue

            train_data = df.dropna()
            test_data = df[df[column].isnull()].drop(columns=[column])

            if train_data.empty or test_data.empty:
                raise ValueError("Insufficient data for regression imputation.")

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
        """
        Impute missing values using interpolation methods.

        Uses ``pandas.DataFrame.interpolate()`` method with the specified method (linear, polynomial, etc.).

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        df = pd.DataFrame(data_array)
        return df.interpolate(method=self.time_series_strategy).to_numpy()
