import unittest
import pandas as pd
import polars as pl
import numpy as np
from mango.processing.


class ImputerTests(unittest.TestCase):
    def setUp(self):
        self.imputer = Imputer()
        self.df_pandas = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
        self.df_polars = pl.DataFrame({"A": [1, 2, None, 4], "B": [None, 2, 3, 4]})

    def tearDown(self):
        pass

    def test_convert_to_numpy_pandas(self):
        array, dtype, columns = self.imputer._convert_to_numpy(self.df_pandas)
        self.assertEqual(dtype, "pandas")
        self.assertEqual(len(columns), 2)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_convert_to_numpy_polars(self):
        array, dtype, columns = self.imputer._convert_to_numpy(self.df_polars)
        self.assertEqual(dtype, "polars")
        self.assertEqual(len(columns), 2)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_apply_global_imputation(self):
        imputed_df = self.imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_apply_column_wise_imputation(self):
        imputer = Imputer(column_strategies={"A": "mean", "B": "median"})
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_knn_impute(self):
        imputer = Imputer(strategy="knn", k_neighbors=2)
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_mice_impute(self):
        imputer = Imputer(strategy="mice")
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_arbitrary_impute(self):
        imputer = Imputer(strategy="arbitrary", arbitrary_value=10)
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertTrue((imputed_df == 10).any().any())

    def test_fill_impute_forward(self):
        imputer = Imputer(strategy="forward")
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_fill_impute_backward(self):
        imputer = Imputer(strategy="backward")
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_regression_impute(self):
        imputer = Imputer(strategy="regression")
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_interpolate_impute(self):
        imputer = Imputer(strategy="interpolate")
        imputed_df = imputer.fit_transform(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_invalid_strategy(self):
        imputer = Imputer(strategy="invalid")
        with self.assertRaises(ValueError):
            imputer.fit_transform(self.df_pandas)


if __name__ == "__main__":
    unittest.main()
