import random
from unittest import TestCase, mock

from mango.data import get_ts_dataset
from mango_time_series.utils.processing_time_series import (
    create_lags_col,
    create_recurrent_dataset,
    get_corr_matrix,
)

try:
    import pandas as pd
except ImportError:
    pd = None


class ProcessingTests(TestCase):
    def setUp(self):
        # Read data to test the functions
        self.df = get_ts_dataset()

    def tearDown(self):
        pass

    def test_create_lags_col(self):
        lags = [-1, -2, 0, 1, 2, 3, 4]
        df_result = create_lags_col(self.df, col="target", lags=lags)
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(
            df_result.shape,
            (
                len(self.df),
                len(self.df.columns) + len(set([lag for lag in lags if lag != 0])),
            ),
        )
        for lag in lags:
            if lag > 0:
                self.assertIn(f"target_lag{lag}", df_result.columns)
            elif lag < 0:
                self.assertIn(f"target_lead{abs(lag)}", df_result.columns)

        # Test pandas ImportError
        with mock.patch.dict("sys.modules", {"pandas": None}):
            with self.assertRaises(
                ImportError,
            ) as context:
                create_lags_col(self.df, col="target", lags=lags)

            self.assertEqual(
                str(context.exception),
                "pandas and numpy need to be installed to use this function",
            )

    def test_create_for_recurrent_network(self):
        random.seed(42)
        data = self.df
        data["new_feature"] = [random.randrange(1, 5) for _ in range(len(data))]
        data["other_new_feature"] = [random.randrange(10, 50) for _ in range(len(data))]
        data = data[["new_feature", "other_new_feature", "date", "target"]]
        data = data.to_numpy()
        look_back = 4

        x, y = create_recurrent_dataset(
            data, look_back, include_output_lags=False, output_last=True
        )

        self.assertEqual(
            x.shape, (data.shape[0] - look_back, look_back, data.shape[1] - 1)
        )
        self.assertEqual(y.shape, (data.shape[0] - look_back,))

    def test_create_for_recurrent_with_lags(self):
        random.seed(42)
        data = self.df
        data["new_feature"] = [random.randrange(1, 5) for _ in range(len(data))]
        data["other_new_feature"] = [random.randrange(10, 50) for _ in range(len(data))]
        data = data[["new_feature", "other_new_feature", "date", "target"]]
        data = data.to_numpy()
        look_back = 4

        x, y = create_recurrent_dataset(
            data, look_back, include_output_lags=True, lags=[1, 7], output_last=True
        )

        self.assertEqual(
            x.shape, (data.shape[0] - look_back - 7, look_back, data.shape[1] + 1)
        )
        self.assertEqual(y.shape, (data.shape[0] - look_back - 7,))

    def test_create_for_recurrent_y_first(self):
        random.seed(42)
        data = self.df
        data["new_feature"] = [random.randrange(1, 5) for _ in range(len(data))]

        data = data[["target", "new_feature", "date"]]
        data = data.to_numpy()
        look_back = 4

        x, y = create_recurrent_dataset(
            data, look_back, include_output_lags=False, output_last=False
        )

        self.assertEqual(
            x.shape, (data.shape[0] - look_back, look_back, data.shape[1] - 1)
        )
        self.assertEqual(y.shape, (data.shape[0] - look_back,))


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
    invalid_df8 = None
    expected_result_invalid_df8 = None
    invalid_df9 = None
    expected_result_invalid_df9 = None
    invalid_df10 = None
    expected_result_invalid_df10 = None
    invalid_df11 = None
    expected_result_invalid_df11 = None
    invalid_df12 = None
    expected_result_invalid_df12 = None
    invalid_df13 = None
    expected_result_invalid_df13 = None
    invalid_df14 = None
    expected_result_invalid_df14 = None

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

        data_8 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "fecha2": pd.date_range("2022-01-01", "2022-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.invalid_df8 = pd.DataFrame(data_8)
        cls.expected_result_invalid_df8 = dict()

        data_9 = {
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
        data_9 = pd.DataFrame(data_9)
        data_9["fecha"] = pd.to_datetime(data_9["fecha"])
        cls.invalid_df9 = data_9.set_index("fecha")
        cls.expected_result_invalid_df9 = dict()

        data_10 = {
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
            "var_str": [
                "a",
                "b",
                "c",
                "c",
                "a",
                "a",
                "b",
                "c",
                "c",
                "a",
                "b",
                "c",
                "c",
                "a",
            ],
        }
        data_10 = pd.DataFrame(data_10)
        data_10["fecha"] = pd.to_datetime(data_10["fecha"])
        cls.invalid_df10 = data_10.set_index("fecha")
        cls.expected_result_invalid_df10 = dict()

        data_11 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
            "var_str": [
                "a",
                "b",
                "c",
                "c",
                "a",
                "a",
                "b",
                "c",
                "c",
                "a",
                "b",
                "c",
                "c",
                "a",
            ],
        }
        cls.invalid_df11 = pd.DataFrame(data_11).set_index("fecha")
        cls.expected_result_invalid_df11 = dict()

        data_12 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
        }
        cls.invalid_df12 = pd.DataFrame(data_12)
        cls.expected_result_invalid_df12 = dict()

        data_13 = {
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
            "var_str": [
                "a",
                "b",
                "c",
                "c",
                "a",
                "a",
                "b",
                "c",
                "c",
                "a",
                "b",
                "c",
                "c",
                "a",
            ],
        }
        data_13 = pd.DataFrame(data_13)
        data_13["fecha"] = pd.to_datetime(data_13["fecha"])
        cls.invalid_df13 = data_13
        cls.expected_result_invalid_df13 = dict()

        data_14 = {
            "fecha": pd.date_range("2023-01-01", "2023-01-14"),
            "value_l1": [random.randint(20, 40) for _ in range(14)],
            "value_l2": [random.randint(30, 60) for _ in range(14)],
            "value_l3": [random.randint(1, 15) for _ in range(14)],
            "value_l4": [random.randint(20, 60) for _ in range(14)],
            "value_l5": [random.randint(40, 80) for _ in range(14)],
            "value_l6": [random.randint(15, 60) for _ in range(14)],
            "value_l7": [random.randint(20, 50) for _ in range(14)],
            "var_str": [
                "a",
                "b",
                "c",
                "c",
                "a",
                "a",
                "b",
                "c",
                "c",
                "a",
                "b",
                "c",
                "c",
                "a",
            ],
        }
        data_14 = pd.DataFrame(data_14)
        data_14["fecha"] = pd.to_datetime(data_14["fecha"])
        cls.invalid_df14 = data_14
        cls.expected_result_invalid_df14 = dict()

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

        top_corr3_ntop = get_corr_matrix(
            self.valid_df3,
            date_col="fecha",
            years_corr=[2023],
            n_top=5,
            subset=["value_l1", "value_l2", "value_l3", "value_l4", "value_l5"],
        )

        top_corr3_threshold = get_corr_matrix(
            self.valid_df3,
            date_col="fecha",
            years_corr=[2023],
            # subset=["value_l1", "value_l2", "value_l3", "value_l4", "value_l5"],
            threshold=0.2,
        )

        top_corr3 = get_corr_matrix(
            self.valid_df3,
            date_col="fecha",
            years_corr=[2023],
            # subset=["value_l1", "value_l2", "value_l3", "value_l4", "value_l5"],
        )

        top_corr3_big_ntop = get_corr_matrix(
            self.valid_df3,
            date_col="fecha",
            years_corr=[2023],
            n_top=9,
            # subset=["value_l1", "value_l2", "value_l3", "value_l4", "value_l5"],
        )

        top_corr3_big_threshold = get_corr_matrix(
            self.valid_df3,
            date_col="fecha",
            years_corr=[2023],
            # subset=["value_l1", "value_l2", "value_l3", "value_l4", "value_l5"],
            threshold=0.99,
        )

        # top_corr4 = get_corr_matrix(
        #     self.invalid_df4, date_col="fecha", years_corr=[2023], n_top=5
        # )
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df4,
            date_col="fecha",
            years_corr=[2023],
            n_top=5,
        )

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

        # top_corr8 = get_corr_matrix(self.invalid_df8, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df8,
            n_top=5,
        )

        # top_corr9 = get_corr_matrix(self.invalid_df9, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df9,
            n_top=5,
        )

        # top_corr10 = get_corr_matrix(self.invalid_df10, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df10,
            n_top=5,
        )

        # top_corr11 = get_corr_matrix(self.invalid_df11, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df11,
            n_top=5,
        )

        top_corr12 = get_corr_matrix(self.invalid_df12, n_top=5)

        # top_corr13 = get_corr_matrix(self.invalid_df13, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df13,
            n_top=5,
        )

        # top_corr14 = get_corr_matrix(self.invalid_df14, n_top=5)
        self.assertRaises(
            ValueError,
            get_corr_matrix,
            self.invalid_df14,
            n_top=5,
        )
