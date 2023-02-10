import random
from unittest import TestCase

from mango.data import get_ts_dataset
from mango.processing import create_lags_col, create_recurrent_dataset
import pandas as pd


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
