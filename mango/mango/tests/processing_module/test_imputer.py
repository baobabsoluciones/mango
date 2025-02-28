import unittest
import pandas as pd
import polars as pl
import numpy as np
from mango.processing import DataImputer


class ImputerTests(unittest.TestCase):
    def setUp(self):
        self.imputer = DataImputer()
        self.df_pandas = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
        self.df_polars = pl.DataFrame({"A": [1, 2, None, 4], "B": [None, 2, 3, 4]})

    def tearDown(self):
        pass

    def test_convert_to_numpy_pandas(self):
        array, dtype, columns = self.imputer._convert_to_numpy(self.df_pandas)
        self.assertEqual(dtype, "pandas")
        self.assertEqual(len(columns), 2)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_data_type_convert_to_numpy(self):
        with self.assertRaises(ValueError):
            self.imputer._convert_to_numpy([1, 2, 3])

    def test_convert_to_numpy_polars(self):
        array, dtype, columns = self.imputer._convert_to_numpy(self.df_polars)
        self.assertEqual(dtype, "polars")
        self.assertEqual(len(columns), 2)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_apply_global_imputation(self):
        imputed_df = self.imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_apply_column_wise_imputation_pandas(self):
        imputer = DataImputer(column_strategies={"A": "mean", "B": "median"})
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_apply_column_wise_imputation_polars(self):
        imputer = DataImputer(column_strategies={"A": "mean", "B": "median"})
        imputed_df = imputer.apply_imputation(self.df_polars)
        self.assertFalse(imputed_df.null_count().to_series().sum() > 0)

    def test_apply_column_global_imputation_with_polars(self):
        imputer = DataImputer(strategy="mean")
        result = imputer.apply_imputation(self.df_polars)
        self.assertIsInstance(result, pl.DataFrame)

    def test_invalid_column_strategy(self):
        imputer = DataImputer(column_strategies={"A": "invalid_strategy", "B": "mean"})
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_invalid_column_dataframe(self):
        imputer = DataImputer(column_strategies={"C": "mean", "B": "mean"})
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_most_frequent_imputation(self):
        imputer = DataImputer(strategy="most_frequent")
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_knn_imputer_without_k_neighbors(self):
        imputer = DataImputer(strategy="knn")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_knn_imputer_without_k_neighbors_polars(self):
        imputer = DataImputer(strategy="knn")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_polars)

    def test_knn_imputer_with_k_neighbors_negative(self):
        imputer = DataImputer(strategy="knn", k_neighbors=-1)
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_knn_impute(self):
        imputer = DataImputer(strategy="knn", k_neighbors=2)
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_mice_impute(self):
        imputer = DataImputer(strategy="mice")
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_arbitrary_impute(self):
        imputer = DataImputer(strategy="arbitrary", arbitrary_value=10)
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertTrue((imputed_df == 10).any().any())

    def test_arbitrary_imputer_without_value(self):
        imputer = DataImputer(strategy="arbitrary")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_fill_impute_forward_backward(self):
        imputer = DataImputer(column_strategies={"A": "forward", "B": "backward"})
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_invalid_fill_strategy(self):
        imputer = DataImputer(strategy="invalid_fill")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_regression_impute(self):
        imputer = DataImputer(strategy="regression")
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_invalid_regression_model(self):
        imputer = DataImputer(strategy="regression", regression_model="invalid")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_interpolate_impute(self):
        imputer = DataImputer(column_strategies={"A": "interpolate", "B": "mean"})
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_invalid_strategy(self):
        imputer = DataImputer(strategy="invalid")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_pandas)

    def test_skip_non_missing_column_in_regression(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]})
        imputer = DataImputer(strategy="regression")
        result = imputer.apply_imputation(df)
        self.assertTrue(result.equals(df))

    def test_insufficient_data_for_regression_imputation(self):
        df = pd.DataFrame({"A": [np.nan, np.nan, np.nan, np.nan], "B": [1, 2, 3, 4]})
        imputer = DataImputer(strategy="regression")
        with self.assertRaises(ValueError):
            imputer.apply_imputation(df)


if __name__ == "__main__":
    unittest.main()
