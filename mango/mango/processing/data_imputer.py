from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import polars as pl
from mango.logging import get_configured_logger
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression

log = get_configured_logger(__name__)


class DataImputer:
    """
    Comprehensive data imputation class supporting multiple strategies and libraries.

    This class provides a unified interface for filling missing values in datasets using
    various imputation strategies. It supports both pandas and polars DataFrames and
    offers flexible configuration options for different imputation approaches.

    The class supports the following imputation strategies:

    - **Statistical imputation**: mean, median, most_frequent using sklearn
    - **KNN imputation**: k-nearest neighbors based imputation
    - **MICE imputation**: Multiple Imputation by Chained Equations
    - **Regression imputation**: Ridge, Lasso, or Linear regression models
    - **Time series imputation**: forward fill, backward fill, interpolation
    - **Arbitrary value imputation**: fill with specified constant values

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create sample data with missing values
        >>> data = pd.DataFrame({
        ...     'A': [1, 2, np.nan, 4, 5],
        ...     'B': [np.nan, 2, 3, 4, np.nan],
        ...     'C': [1, np.nan, 3, 4, 5]
        ... })
        >>>
        >>> # Mean imputation
        >>> imputer = DataImputer(strategy="mean")
        >>> imputed_data = imputer.apply_imputation(data)
        >>>
        >>> # Column-specific strategies
        >>> strategies = {'A': 'mean', 'B': 'median', 'C': 'knn'}
        >>> imputer = DataImputer(column_strategies=strategies)
        >>> imputed_data = imputer.apply_imputation(data)
    """

    def __init__(
        self,
        strategy: str = "mean",
        column_strategies: Optional[Dict[str, str]] = None,
        regression_model: Optional[str] = "ridge",
        id_columns: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the DataImputer with specified imputation strategy.

        Sets up the imputer with the chosen strategy and validates all parameters.
        The imputer can use either a global strategy for all columns or different
        strategies for specific columns.

        :param strategy: Global imputation strategy to use for all columns
        :type strategy: str
        :param column_strategies: Dictionary mapping column names to specific strategies
        :type column_strategies: Optional[Dict[str, str]]
        :param regression_model: Regression model for regression imputation ("ridge", "lasso", "linear")
        :type regression_model: Optional[str]
        :param id_columns: Column names to use as identifiers for group-based imputation
        :type id_columns: Optional[str]
        :param kwargs: Additional keyword arguments passed to the imputation methods
        :type kwargs: dict
        :raises ValueError: If strategy or parameters are invalid
        :raises TypeError: If parameter types are incorrect

        Example:
            >>> # Simple mean imputation
            >>> imputer = DataImputer(strategy="mean")
            >>>
            >>> # Column-specific strategies with KNN parameters
            >>> strategies = {'age': 'mean', 'income': 'knn'}
            >>> imputer = DataImputer(
            ...     column_strategies=strategies,
            ...     n_neighbors=3
            ... )
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

        log.info(f"Initializing DataImputer with strategy: {strategy}")
        self._validate_params()
        log.info("DataImputer initialization completed successfully")

    def _validate_strategy(self, strategy: str):
        """
        Validate the imputation strategy and its parameters.

        Checks if the strategy is supported and validates strategy-specific
        parameters such as n_neighbors for KNN or fill_value for arbitrary imputation.

        :param strategy: Imputation strategy to validate
        :type strategy: str
        :raises ValueError: If strategy is invalid or parameters are incorrect

        Example:
            >>> imputer = DataImputer()
            >>> imputer._validate_strategy("mean")  # Valid
            >>> imputer._validate_strategy("invalid")  # Raises ValueError
        """
        if strategy not in self.strategy_methods:
            log.error(f"Invalid strategy '{strategy}' provided")
            raise ValueError(f"Invalid strategy '{strategy}'.")

        if strategy == "knn":
            n_neighbors = self.kwargs.get("n_neighbors")
            if n_neighbors is None:
                self.kwargs["n_neighbors"] = 5
                log.info("Setting default n_neighbors=5 for KNN imputation")
            elif n_neighbors <= 0:
                log.error(f"Invalid n_neighbors value: {n_neighbors}")
                raise ValueError("n_neighbors must be greater than 0.")

        if strategy == "arbitrary" and self.kwargs.get("fill_value") is None:
            log.error("Arbitrary imputation requires fill_value parameter")
            raise ValueError(
                "An arbitrary value must be provided for arbitrary imputation."
            )

        if strategy == "regression":
            regression_model = self.kwargs.get(
                "regression_model", self.regression_model
            )
            if regression_model not in self.model_mapping:
                log.error(f"Invalid regression model: {regression_model}")
                raise ValueError(
                    "Invalid regression model. Choose from 'ridge', 'lasso', 'linear'."
                )

    def _validate_params(self):
        """
        Validate all imputer parameters and configuration.

        Performs comprehensive validation of the imputer configuration including
        strategy validation, column strategy validation, and parameter consistency
        checks. Ensures that ID columns are not included in column strategies.

        :raises ValueError: If any parameter validation fails
        :raises TypeError: If parameter types are incorrect

        Example:
            >>> imputer = DataImputer()
            >>> imputer._validate_params()  # Validates current configuration
        """
        if self.column_strategies is None:
            log.debug(f"Validating global strategy: {self.strategy}")
            self._validate_strategy(self.strategy)

        else:
            if not isinstance(self.column_strategies, dict):
                log.error("Column strategies must be a dictionary")
                raise ValueError("Column strategies must be a dictionary.")

            # Validate that ID columns are not in column_strategies
            invalid_id_columns = [
                col for col in self.id_columns if col in self.column_strategies
            ]
            if invalid_id_columns:
                log.error(f"ID columns {invalid_id_columns} found in column strategies")
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
                log.error(f"Invalid strategies found: {errors}")
                raise ValueError(f"Invalid strategies found for columns: {errors}")

            for col, strat in self.column_strategies.items():
                log.debug(f"Validating strategy '{strat}' for column '{col}'")
                self._validate_strategy(strat)

    @staticmethod
    def _convert_to_numpy(data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Convert pandas or polars DataFrame to numpy array and detect data type.

        Converts the input DataFrame to a numpy array while preserving column
        information and detecting the original data type for later conversion back.
        This method is used internally to standardize data processing across
        different DataFrame types.

        :param data: Input data as pandas or polars DataFrame
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Tuple containing (numpy array, data type string, column names)
        :rtype: tuple[np.ndarray, str, list]
        :raises ValueError: If input data is not a supported DataFrame type

        Example:
            >>> import pandas as pd
            >>> import polars as pl
            >>>
            >>> # Pandas DataFrame
            >>> df_pd = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> array, dtype, cols = DataImputer._convert_to_numpy(df_pd)
            >>> print(dtype)  # 'pandas'
            >>>
            >>> # Polars DataFrame
            >>> df_pl = pl.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> array, dtype, cols = DataImputer._convert_to_numpy(df_pl)
            >>> print(dtype)  # 'polars'
        """
        if isinstance(data, pd.DataFrame):
            log.debug("Converting pandas DataFrame to numpy array")
            return data.to_numpy(), "pandas", data.columns
        elif isinstance(data, pl.DataFrame):
            log.debug("Converting polars DataFrame to numpy array")
            return data.to_numpy(), "polars", data.columns
        else:
            log.error(f"Unsupported data type: {type(data)}")
            raise ValueError("Input data must be a pandas or polars DataFrame.")

    @staticmethod
    def _convert_back(data, data_type, columns):
        """
        Convert numpy array back to pandas or polars DataFrame.

        Reconstructs the original DataFrame type from a numpy array using the
        preserved column names and data type information. This method ensures
        that the output format matches the input format.

        :param data: Numpy array containing the processed data
        :type data: np.ndarray
        :param data_type: Original data type string ("pandas" or "polars")
        :type data_type: str
        :param columns: List of column names to assign to the DataFrame
        :type columns: list
        :return: Reconstructed DataFrame in the original format
        :rtype: Union[pd.DataFrame, pl.DataFrame]

        Example:
            >>> import numpy as np
            >>>
            >>> # Convert back to pandas
            >>> array = np.array([[1, 2], [3, 4]])
            >>> df = DataImputer._convert_back(array, "pandas", ['A', 'B'])
            >>> print(type(df))  # <class 'pandas.core.frame.DataFrame'>
            >>>
            >>> # Convert back to polars
            >>> df = DataImputer._convert_back(array, "polars", ['A', 'B'])
            >>> print(type(df))  # <class 'polars.dataframe.frame.DataFrame'>
        """
        columns = list(columns)

        if data_type == "pandas":
            log.debug("Converting numpy array back to pandas DataFrame")
            return pd.DataFrame(data, columns=columns)
        elif data_type == "polars":
            log.debug("Converting numpy array back to polars DataFrame")
            return pl.DataFrame(data, schema=columns)

    def _apply_imputation_by_id(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply imputation grouped by identifier columns.

        Performs imputation within groups defined by the specified ID columns.
        This is useful when data has a hierarchical structure where missing
        values should be imputed within each group separately.

        :param data: Input data containing the ID columns and data to impute
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Data with missing values imputed within each ID group
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        :raises ValueError: If ID columns are not found in the dataset

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> # Data with groups
            >>> data = pd.DataFrame({
            ...     'group': ['A', 'A', 'B', 'B'],
            ...     'value': [1, np.nan, 3, np.nan]
            ... })
            >>>
            >>> # Impute within each group
            >>> imputer = DataImputer(strategy="mean", id_columns=['group'])
            >>> result = imputer._apply_imputation_by_id(data)
        """
        if isinstance(data, pl.DataFrame):
            log.debug("Converting polars DataFrame to pandas for group operations")
            data = data.to_pandas()

        # Check if the id column is in the dataset
        if not all(col in data.columns for col in self.id_columns):
            log.error(f"ID columns {self.id_columns} not found in dataset")
            raise ValueError(f"Columns {self.id_columns} not found in dataset.")

        log.info(f"Applying imputation by ID columns: {self.id_columns}")
        grouped = data.groupby(self.id_columns)
        imputed_groups = []

        for group_id, group in grouped:
            log.debug(f"Processing group: {group_id}")
            group = group.copy()
            if self.column_strategies:
                imputed_group = self._apply_column_wise_imputation(group)
            else:
                imputed_group = self._apply_global_imputation(group)
            imputed_groups.append(imputed_group)

        log.info(f"Completed imputation for {len(imputed_groups)} groups")
        return pd.concat(imputed_groups).reset_index(drop=True)

    def apply_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply imputation to fill missing values in the dataset.

        This is the main public method for applying imputation to a dataset.
        It automatically determines the appropriate imputation approach based
        on the configuration and applies it to the input data.

        :param data: Input data containing missing values to be imputed
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Dataset with missing values filled according to the strategy
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        :raises ValueError: If data validation fails or imputation cannot be applied

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> # Create data with missing values
            >>> data = pd.DataFrame({
            ...     'A': [1, 2, np.nan, 4],
            ...     'B': [np.nan, 2, 3, np.nan]
            ... })
            >>>
            >>> # Apply mean imputation
            >>> imputer = DataImputer(strategy="mean")
            >>> result = imputer.apply_imputation(data)
            >>> print(result.isnull().sum())
        """
        log.info(f"Starting imputation on dataset with shape: {data.shape}")

        if self.id_columns:
            log.info("Using ID-based imputation")
            return self._apply_imputation_by_id(data)
        if self.column_strategies:
            log.info("Using column-wise imputation")
            return self._apply_column_wise_imputation(data)
        else:
            log.info("Using global imputation")
            return self._apply_global_imputation(data)

    def _apply_global_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply a global imputation strategy to the entire dataset.

        Applies the same imputation strategy to all columns in the dataset.
        This method handles the conversion between pandas/polars and numpy
        formats internally and applies the configured strategy uniformly.

        :param data: Input data to be imputed
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Dataset with missing values filled using the global strategy
        :rtype: Union[pd.DataFrame, pl.DataFrame]

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> data = pd.DataFrame({
            ...     'A': [1, np.nan, 3],
            ...     'B': [np.nan, 2, np.nan]
            ... })
            >>>
            >>> imputer = DataImputer(strategy="mean")
            >>> result = imputer._apply_global_imputation(data)
            >>> # Both columns will use mean imputation
        """
        log.debug(f"Applying global strategy '{self.strategy}' to all columns")
        data_array, data_type, columns = self._convert_to_numpy(data)

        imputed_data = self.strategy_methods[self.strategy](data_array)
        log.debug(f"Global imputation completed. Shape: {imputed_data.shape}")

        return self._convert_back(imputed_data, data_type, columns)

    def _apply_column_wise_imputation(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Apply column-specific imputation strategies to the dataset.

        Applies different imputation strategies to different columns based on
        the column_strategies configuration. Each column can have its own
        imputation method while maintaining the original data structure.

        :param data: Input data to be imputed
        :type data: Union[pd.DataFrame, pl.DataFrame]
        :return: Dataset with missing values filled using column-specific strategies
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        :raises ValueError: If specified columns are not found in the dataset

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>>
            >>> data = pd.DataFrame({
            ...     'age': [25, np.nan, 30],
            ...     'income': [50000, 60000, np.nan],
            ...     'category': ['A', 'B', np.nan]
            ... })
            >>>
            >>> strategies = {
            ...     'age': 'mean',
            ...     'income': 'median',
            ...     'category': 'most_frequent'
            ... }
            >>>
            >>> imputer = DataImputer(column_strategies=strategies)
            >>> result = imputer._apply_column_wise_imputation(data)
        """
        # Convert polars or pandas DataFrame to numpy array to obtain data type and column names
        data_array, data_type, columns = self._convert_to_numpy(data)
        log.debug(
            f"Column-wise imputation for columns: {list(self.column_strategies.keys())}"
        )

        # The columns that are specified in the column_strategies must be present in the dataset
        invalid_columns = [col for col in self.column_strategies if col not in columns]
        if invalid_columns:
            log.error(f"Columns {invalid_columns} not found in dataset")
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
        Impute missing values using sklearn SimpleImputer.

        Uses sklearn's SimpleImputer with statistical strategies like mean,
        median, or most_frequent. This is suitable for numerical and
        categorical data with simple missing value patterns.

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled using statistical methods
        :rtype: np.ndarray

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
            >>> imputer = DataImputer(strategy="mean")
            >>> result = imputer._simple_impute(data)
            >>> # Missing values replaced with column means
        """
        log.debug(f"Applying SimpleImputer with strategy: {self.strategy}")
        imputer = SimpleImputer(strategy=self.strategy, **self.kwargs)
        result = imputer.fit_transform(data_array)
        log.debug(f"SimpleImputer completed. Missing values: {np.isnan(result).sum()}")
        return result

    def _knn_impute(self, data_array: np.ndarray):
        """
        Impute missing values using K-Nearest Neighbors.

        Uses sklearn's KNNImputer to fill missing values based on the k-nearest
        neighbors. This method is particularly useful when there are relationships
        between features that should be preserved during imputation.

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled using KNN approach
        :rtype: np.ndarray
        :raises ValueError: If insufficient data for KNN imputation

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
            >>> imputer = DataImputer(strategy="knn", n_neighbors=2)
            >>> result = imputer._knn_impute(data)
            >>> # Missing value filled based on nearest neighbors
        """
        n_neighbors = self.kwargs.get("n_neighbors", 5)
        log.debug(f"Applying KNNImputer with n_neighbors: {n_neighbors}")
        imputer = KNNImputer(**self.kwargs)
        result = imputer.fit_transform(data_array)
        log.debug(f"KNNImputer completed. Missing values: {np.isnan(result).sum()}")
        return result

    def _mice_impute(self, data_array: np.ndarray):
        """
        Impute missing values using MICE (Multiple Imputation by Chained Equations).

        Uses sklearn's IterativeImputer which implements the MICE algorithm.
        This method iteratively imputes missing values by modeling each feature
        as a function of other features. It's particularly effective for
        complex missing data patterns.

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled using MICE algorithm
        :rtype: np.ndarray
        :raises ValueError: If insufficient data for MICE imputation

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan], [np.nan, 5, 6], [7, np.nan, 9]])
            >>> imputer = DataImputer(strategy="mice")
            >>> result = imputer._mice_impute(data)
            >>> # Missing values filled using iterative modeling
        """
        log.debug("Applying MICE imputation using IterativeImputer")
        imputer = IterativeImputer(**self.kwargs)
        result = imputer.fit_transform(data_array)
        log.debug(
            f"MICE imputation completed. Missing values: {np.isnan(result).sum()}"
        )
        return result

    def _arbitrary_impute(self, data_array: np.ndarray):
        """
        Impute missing values with a specified constant value.

        Uses sklearn's SimpleImputer with strategy='constant' to fill all
        missing values with a user-specified value. This is useful when
        missing values have a specific meaning or when a default value is known.

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled with the specified constant
        :rtype: np.ndarray
        :raises ValueError: If fill_value is not specified in kwargs

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
            >>> imputer = DataImputer(strategy="arbitrary", fill_value=0)
            >>> result = imputer._arbitrary_impute(data)
            >>> # All missing values replaced with 0
        """
        fill_value = self.kwargs.get("fill_value")
        log.debug(f"Applying arbitrary imputation with fill_value: {fill_value}")
        imputer = SimpleImputer(strategy="constant", **self.kwargs)
        result = imputer.fit_transform(data_array)
        log.debug(
            f"Arbitrary imputation completed. Missing values: {np.isnan(result).sum()}"
        )
        return result

    def _fill_impute(self, data_array: np.ndarray):
        """
        Fill missing values using forward or backward fill methods.

        Uses pandas forward fill (ffill) or backward fill (bfill) methods to
        propagate the last valid observation forward or the next valid
        observation backward. This is particularly useful for time series data.

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled using forward/backward fill
        :rtype: np.ndarray

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
            >>> imputer = DataImputer(strategy="forward")
            >>> result = imputer._fill_impute(data)
            >>> # Missing values filled with previous values
        """
        df = pd.DataFrame(data_array)
        result = data_array.copy()
        if self.strategy == "forward":
            log.debug("Applying forward fill imputation")
            result = df.ffill().to_numpy()
        elif self.strategy == "backward":
            log.debug("Applying backward fill imputation")
            result = df.bfill().to_numpy()

        log.debug(
            f"Fill imputation completed. Missing values: {np.isnan(result).sum()}"
        )
        return result

    def _regression_impute(self, data_array: np.ndarray):
        """
        Impute missing values using regression models.

        Uses sklearn linear regression models to predict missing values based
        on other features. The method trains a regression model on complete
        cases and uses it to predict missing values.

        Supported models:
        - Ridge regression: Good for multicollinearity
        - Lasso regression: Includes feature selection
        - Linear regression: Standard linear model

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled using regression predictions
        :rtype: np.ndarray
        :raises ValueError: If insufficient data for regression imputation

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
            >>> imputer = DataImputer(strategy="regression", regression_model="ridge")
            >>> result = imputer._regression_impute(data)
            >>> # Missing value predicted using ridge regression
        """
        df = pd.DataFrame(data_array)
        if df.shape[1] == 1:
            log.error("Regression imputation requires at least 2 columns")
            raise ValueError(
                "Insufficient columns for regression imputation. It is necessary to have at least two columns."
            )

        log.debug(f"Applying regression imputation with model: {self.regression_model}")

        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue

            log.debug(f"Imputing column '{column}' using regression")
            train_data = df.dropna()
            test_data = df[df[column].isnull()].drop(columns=[column])

            if train_data.empty or test_data.empty:
                log.error(
                    f"Insufficient data for regression imputation of column '{column}'"
                )
                raise ValueError("Insufficient data for regression imputation.")

            X_train = train_data.drop(columns=[column])
            y_train = train_data[column]

            model = self.model_mapping[self.regression_model](**self.kwargs)
            model.fit(X_train, y_train)
            predictions = model.predict(test_data)
            df.loc[df[column].isnull(), column] = predictions
            log.debug(f"Imputed {len(predictions)} missing values in column '{column}'")

        log.debug(
            f"Regression imputation completed. Missing values: {df.isnull().sum().sum()}"
        )
        return df.to_numpy()

    def _interpolate_impute(self, data_array: np.ndarray):
        """
        Impute missing values using interpolation methods.

        Uses pandas interpolation methods to estimate missing values based on
        surrounding data points. This is particularly useful for time series
        data where values change smoothly over time.

        :param data_array: Input data as numpy array
        :type data_array: np.ndarray
        :return: Data with missing values filled using interpolation
        :rtype: np.ndarray

        Example:
            >>> import numpy as np
            >>>
            >>> data = np.array([[1, 2, np.nan, 4], [5, np.nan, 7, 8]])
            >>> imputer = DataImputer(strategy="interpolate", method="linear")
            >>> result = imputer._interpolate_impute(data)
            >>> # Missing values interpolated linearly
        """
        method = self.kwargs.get("method", "linear")
        log.debug(f"Applying interpolation imputation with method: {method}")
        df = pd.DataFrame(data_array)
        result = df.interpolate(**self.kwargs).to_numpy()
        log.debug(
            f"Interpolation imputation completed. Missing values: {np.isnan(result).sum()}"
        )
        return result
