from unittest import TestCase

from mango.processing.file_functions import normalize_path
from mango.processing.processing_time_series import create_lags_col
import pandas as pd


class ProcessingTests(TestCase):
    def setUp(self):
        # Read data to test the functions
        self.df = pd.read_pickle(normalize_path("../data/ts_dataset.pkl"))

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
