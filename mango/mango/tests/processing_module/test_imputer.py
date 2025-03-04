import unittest

import numpy as np
import pandas as pd
import polars as pl
from mango.processing import DataImputer


class ImputerTests(unittest.TestCase):
    def setUp(self):
        self.df_pandas = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
        self.df_polars = pl.DataFrame({"A": [1, 2, None, 4], "B": [None, 2, 3, 4]})

        self.df_id = pd.DataFrame(
            {
                "store_id": [1, 1, 1, 2, 2, 2, 3, 3],
                "A": [1, np.nan, 3, 4, np.nan, 6, 7, np.nan],
                "B": [np.nan, 2, 3, 4, 5, np.nan, 7, 8],
            }
        )

        self.df_multiple_id = pd.DataFrame(
            {
                "store_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
                "store_id2": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                "A": [1, np.nan, 3, 4, 5, np.nan, 7, 8, 9, np.nan],
                "B": [np.nan, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

    def tearDown(self):
        pass

    def test_convert_to_numpy_pandas(self):
        imputer = DataImputer(strategy="mean")
        array, dtype, columns = imputer._convert_to_numpy(self.df_pandas)
        self.assertEqual(dtype, "pandas")
        self.assertEqual(len(columns), 2)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_invalid_data_type_convert_to_numpy(self):
        imputer = DataImputer(strategy="mean")
        with self.assertRaises(ValueError):
            imputer._convert_to_numpy([1, 2, 3])

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            DataImputer(strategy="invalid")

    def test_column_strategie_not_dict(self):
        with self.assertRaises(ValueError):
            DataImputer(column_strategies="mean")

    def test_convert_to_numpy_polars(self):
        imputer = DataImputer(strategy="mean")
        array, dtype, columns = imputer._convert_to_numpy(self.df_polars)
        self.assertEqual(dtype, "polars")
        self.assertEqual(len(columns), 2)
        self.assertTrue(isinstance(array, np.ndarray))

    def test_apply_global_imputation(self):
        imputer = DataImputer(strategy="mean")
        imputed_df = imputer.apply_imputation(self.df_pandas)
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
        with self.assertRaises(ValueError):
            DataImputer(column_strategies={"A": "invalid_strategy", "B": "mean"})

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
        # Check in n_neighbors is 5 (default value)
        self.assertEqual(imputer.kwargs.get("n_neighbors"), 5)

    def test_knn_imputer_with_k_neighbors_negative(self):
        with self.assertRaises(ValueError):
            DataImputer(strategy="knn", n_neighbors=-1)

    def test_knn_impute(self):
        imputer = DataImputer(strategy="knn", n_neighbors=2)
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_mice_impute(self):
        imputer = DataImputer(strategy="mice")
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_arbitrary_impute(self):
        imputer = DataImputer(strategy="arbitrary", fill_value=10)
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertTrue((imputed_df == 10).any().any())

    def test_arbitrary_imputer_without_value(self):
        with self.assertRaises(ValueError):
            DataImputer(strategy="arbitrary")

    def test_fill_impute_forward_backward(self):
        imputer = DataImputer(column_strategies={"A": "forward", "B": "backward"})
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_regression_impute_default(self):
        imputer = DataImputer(strategy="regression")
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_regression_impute(self):
        imputer = DataImputer(strategy="regression", regression_model="lasso")
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_invalid_regression_model(self):
        with self.assertRaises(ValueError):
            DataImputer(strategy="regression", regression_model="invalid")

    def test_interpolate_impute(self):
        imputer = DataImputer(column_strategies={"A": "interpolate", "B": "mean"})
        imputed_df = imputer.apply_imputation(self.df_pandas)
        self.assertFalse(imputed_df.isnull().values.any())

    def test_regression_impute_column(self):
        imputer = DataImputer(
            column_strategies={"A": "regression", "B": "mean"}, regression_model="lasso"
        )
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

    def test_imputation_global_with_id(self):
        imputer = DataImputer(strategy="mean", id_columns=["store_id"])
        imputed_df = imputer.apply_imputation(self.df_id)
        self.assertFalse(imputed_df.isnull().values.any())
        group_means = self.df_id.groupby("store_id").mean()
        for store_id in self.df_id["store_id"].unique():
            imputed_group = imputed_df[imputed_df["store_id"] == store_id]
            for col in ["A", "B"]:
                expected_value = group_means.loc[store_id, col]
                self.assertTrue(
                    (
                        imputed_group[col].fillna(expected_value) == imputed_group[col]
                    ).all()
                )

    def test_column_wise_imputation_with_id(self):
        imputer = DataImputer(
            column_strategies={"A": "mean", "B": "median"}, id_columns=["store_id"]
        )
        imputed_df = imputer.apply_imputation(self.df_id)
        self.assertFalse(imputed_df.isnull().values.any())
        for store_id in self.df_id["store_id"].unique():
            original_group = self.df_id[self.df_id["store_id"] == store_id]
            imputed_group = imputed_df[imputed_df["store_id"] == store_id]

            for col in ["A", "B"]:
                if col in ["A"]:
                    expected_value = original_group["A"].mean()
                else:
                    expected_value = original_group["B"].median()

                self.assertTrue(
                    (
                        imputed_group[col].fillna(expected_value) == imputed_group[col]
                    ).all()
                )

    def test_imputation_fails_for_missing_id_column(self):
        imputer = DataImputer(strategy="mean", id_columns=["non_existing_column"])
        with self.assertRaises(ValueError):
            imputer.apply_imputation(self.df_id)

    def test_id_in_column_strategies(self):
        with self.assertRaises(ValueError):
            DataImputer(
                column_strategies={"store_id": "mean", "B": "median"},
                id_columns=["store_id"],
            )

    def test_multiple_id_global_imputation(self):
        imputer = DataImputer(strategy="mean", id_columns=["store_id", "store_id2"])
        imputed_df = imputer.apply_imputation(self.df_multiple_id)
        self.assertFalse(imputed_df.isnull().values.any())
        group_means = self.df_multiple_id.groupby(["store_id", "store_id2"]).mean()
        for store_id, store_id2 in (
            self.df_multiple_id[["store_id", "store_id2"]].drop_duplicates().values
        ):
            imputed_group = imputed_df[
                (imputed_df["store_id"] == store_id)
                & (imputed_df["store_id2"] == store_id2)
            ]
            for col in ["A", "B"]:
                expected_value = group_means.loc[(store_id, store_id2), col]
                self.assertTrue(
                    (
                        imputed_group[col].fillna(expected_value) == imputed_group[col]
                    ).all()
                )

    def test_multiple_id_column_wise_imputation(self):
        imputer = DataImputer(
            column_strategies={"A": "mean", "B": "median"},
            id_columns=["store_id", "store_id2"],
        )
        imputed_df = imputer.apply_imputation(self.df_multiple_id)
        self.assertFalse(imputed_df.isnull().values.any())
        for store_id, store_id2 in (
            self.df_multiple_id[["store_id", "store_id2"]].drop_duplicates().values
        ):
            original_group = self.df_multiple_id[
                (self.df_multiple_id["store_id"] == store_id)
                & (self.df_multiple_id["store_id2"] == store_id2)
            ]
            imputed_group = imputed_df[
                (imputed_df["store_id"] == store_id)
                & (imputed_df["store_id2"] == store_id2)
            ]
            for col in ["A", "B"]:
                if col in ["A"]:
                    expected_value = original_group["A"].mean()
                else:
                    expected_value = original_group["B"].median()

                self.assertTrue(
                    (
                        imputed_group[col].fillna(expected_value) == imputed_group[col]
                    ).all()
                )


if __name__ == "__main__":
    unittest.main()
