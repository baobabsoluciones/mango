from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression


class DataImputer:
    def __init__(
        self,
        strategy: str = "mean",
        column_strategies: Optional[Dict[str, str]] = None,
        regression_model: Optional[str] = "ridge",
        id_columns: Optional[str] = None,
        **kwargs,
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
        :param column_strategies: dictionary containing column names and their respective strategies. If None, global strategy will be used
        :type column_strategies: Optional[Dict[str,str]]
        :param regression_model: regression model to use for regression imputation. It can be "ridge" or "linear"
        :type regression_model: Optional[str]
        :param id_columns: columns to be used as identifiers for regression imputation
        :type id_columns: Optional[str]
        :param kwargs: additional keyword arguments for the imputation strategy
        :type kwargs: dict
        """

        self.strategy = strategy
        self.column_strategies = column_strategies
        self.regression_model = regression_model
        self.id_columns = id_columns or []
        self.kwargs = kwargs

        self.model_mapping = {
            "ridge": Ridge,
            "lasso": Lasso,
            "linear": LinearRegression,
        }

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

        self._validate_params()

    def _validate_strategy(self, strategy: str):
        """
        Validate the imputation strategy.
        :param strategy: Imputation strategy
        :type strategy: str
        """
        if strategy not in self.strategy_methods:
            raise ValueError(f"Invalid strategy '{strategy}'.")

        if strategy == "knn":
            n_neighbors = self.kwargs.get("n_neighbors")
            if n_neighbors is None:
                self.kwargs["n_neighbors"] = 5
            elif n_neighbors <= 0:
                raise ValueError("n_neighbors must be greater than 0.")

        if strategy == "arbitrary" and self.kwargs.get("fill_value") is None:
            raise ValueError(
                "An arbitrary value must be provided for arbitrary imputation."
            )

        if strategy == "regression":
            regression_model = self.kwargs.get(
                "regression_model", self.regression_model
            )
            if regression_model not in self.model_mapping:
                raise ValueError(
                    "Invalid regression model. Choose from 'ridge', 'lasso', 'linear'."
                )

    def _validate_params(self):
        """
        Validate the imputer parameters.
        """
        if self.column_strategies is None:
            self._validate_strategy(self.strategy)

        else:
            if not isinstance(self.column_strategies, dict):
                raise ValueError("Column strategies must be a dictionary.")

            # Validate that ID columns are not in column_strategies
            invalid_id_columns = [
                col for col in self.id_columns if col in self.column_strategies
            ]
            if invalid_id_columns:
                raise ValueError(
                    f"ID columns {invalid_id_columns} should not be included in column strategies."
                )

            # The different strategies are validated
            invalid_strategies = [
                (col, strat)
                for col, strat in self.column_strategies.items()
                if strat not in self.strategy_methods
            ]

            if invalid_strategies:
                errors = ", ".join(
                    [f"'{col}': '{strat}'" for col, strat in invalid_strategies]
                )
                raise ValueError(f"Invalid strategies found for columns: {errors}")

            for col, strat in self.column_strategies.items():
                self._validate_strategy(strat)

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

    def _apply_imputation_by_id(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply imputation by group of identifiers.
        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()

        # Check if the id column is in the dataset
        if not all(col in data.columns for col in self.id_columns):
            raise ValueError(f"Columns {self.id_columns} not found in dataset.")

        grouped = data.groupby(self.id_columns)
        imputed_groups = []

        for _, group in grouped:
            group = group.copy()
            if self.column_strategies:
                imputed_group = self._apply_column_wise_imputation(group)
            else:
                imputed_group = self._apply_global_imputation(group)
            imputed_groups.append(imputed_group)

        return pd.concat(imputed_groups).reset_index(drop=True)

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
        if self.id_columns:
            return self._apply_imputation_by_id(data)
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

        imputed_data = self.strategy_methods[self.strategy](data_array)

        return self._convert_back(imputed_data, data_type, columns)

    def _apply_column_wise_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply column-wise imputation strategies to the dataset.

        :param data: Input data
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Imputed data
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        # Convert polars or pandas DataFrame to numpy array to obtain data type and column names
        data_array, data_type, columns = self._convert_to_numpy(data)

        # The columns that are specified in the column_strategies must be present in the dataset
        invalid_columns = [col for col in self.column_strategies if col not in columns]
        if invalid_columns:
            raise ValueError(f"Columns {invalid_columns} not found in dataset.")

        imputed_array = data_array.copy()

        for i, column in enumerate(columns):
            if column in self.column_strategies:
                self.strategy = self.column_strategies[column]
                impute_func = self.strategy_methods[self.strategy]

                # Extract column, reshape to (n,1) for sklearn compatibility
                column_data = imputed_array[:, i].reshape(-1, 1)
                imputed_array[:, i] = impute_func(column_data).flatten()

        return self._convert_back(imputed_array, data_type, columns)

    def _simple_impute(self, data_array: np.ndarray):
        """
        Impute missing values using SimpleImputer.

        Uses ``sklearn.impute.SimpleImputer`` with the specified strategy (mean, median, or most_frequent).

        :param data_array: Input data
        :type data_array: np.ndarray
        :return: Imputed data
        :rtype: np.ndarray
        """
        imputer = SimpleImputer(strategy=self.strategy, **self.kwargs)
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
        imputer = KNNImputer(**self.kwargs)
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
        imputer = IterativeImputer(**self.kwargs)
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
        imputer = SimpleImputer(strategy="constant", **self.kwargs)
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
        if df.shape[1] == 1:
            raise ValueError(
                "Insufficient columns for regression imputation. It is necessary to have at least two columns."
            )

        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue

            train_data = df.dropna()
            test_data = df[df[column].isnull()].drop(columns=[column])

            if train_data.empty or test_data.empty:
                raise ValueError("Insufficient data for regression imputation.")

            X_train = train_data.drop(columns=[column])
            y_train = train_data[column]

            model = self.model_mapping[self.regression_model](**self.kwargs)
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
        return df.interpolate(**self.kwargs).to_numpy()
