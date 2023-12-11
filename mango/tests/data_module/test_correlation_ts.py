import random
from unittest import TestCase

import pandas as pd

from mango.data import get_corr_matrix


class CorrelationMatrix(TestCase):
    valid_df1 = None
    expected_result_valid_df1 = None
    valid_df2 = None
    expected_result_valid_df2 = None
    valid_df3 = None
    expected_result_valid_df3 = None
    invalid_df4 = None
    expected_result_invalid_df4 = None
    valid_df5 = None
    expected_result_valid_df5 = None
    invalid_df6 = None
    expected_result_invalid_df6 = None
    invalid_df7 = None
    expected_result_invalid_df7 = None

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls) -> None:
        data_1 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
        }
        cls.valid_df1 = pd.DataFrame(data_1)
        cls.expected_result_valid_df1 = dict()

        data_2 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
        }
        cls.valid_df2 = pd.DataFrame(data_2)
        cls.expected_result_valid_df2 = dict()

        data_3 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.valid_df3 = pd.DataFrame(data_3)
        cls.expected_result_valid_df3 = dict()

        data_4 = {
            "fecha": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
                "2023-01-08",
                "2023-01-09",
                "2023-01-10",
                "2023-01-11",
                "2023-01-12",
                "2023-01-13",
            ],
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.invalid_df4 = pd.DataFrame(data_4)
        cls.expected_result_invalid_df4 = dict()

        data_5 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.valid_df5 = pd.DataFrame(data_5).set_index("fecha")
        cls.expected_result_valid_df5 = dict()

        data_6 = {
            "fecha": list(range(0, 14, 1)),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.invalid_df6 = pd.DataFrame(data_6).set_index("fecha")
        cls.expected_result_invalid_df6 = dict()

        data_7 = {
            "fecha": list(range(0, 14, 1)),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.invalid_df7 = pd.DataFrame(data_7)
        cls.expected_result_invalid_df7 = dict()

    def test_get_corr_matrix(self):
        # Call function get_corr_matrix with all the parameters: df, date_column, years_corr, n_top
        top_corr1 = get_corr_matrix(
            self.valid_df1, date_col="fecha", years_corr=[2023], n_top=5
        )
        # self.assertEqual(top_corr1,self.expected_result_valid_df1)
        # self.assertRaises(ValueError, get_corr_matrix,self.valid_df1,date_col="fecha", years_corr=[2023], n_top=5)

        top_corr2 = get_corr_matrix(
            self.valid_df2, date_col="fecha", years_corr=[2023], n_top=5
        )

        top_corr3 = get_corr_matrix(
            self.valid_df3, date_col="fecha", years_corr=[2023], n_top=5
        )
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df4,
            date_col="fecha",
            years_corr=[2023],
            n_top=5,
        )
        # top_corr4 = get_corr_matrix(
        #     self.invalid_df4, date_col="fecha", years_corr=[2023], n_top=5
        # )

        top_corr5 = get_corr_matrix(self.valid_df5, n_top=5)

        # top_corr6 = get_corr_matrix(self.invalid_df6, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df6,
            n_top=5,
        )

        # top_corr7 = get_corr_matrix(self.invalid_df7, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df7,
            n_top=5,
        )
