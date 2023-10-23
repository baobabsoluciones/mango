from unittest import TestCase

from mango.processing import (
    pickle_copy,
    unique,
    reverse_dict,
    cumsum,
    lag_list,
    row_number,
    flatten,
    df_to_list,
    df_to_dict,
    load_excel,
    lead_list,
)
from mango.tests.const import normalize_path


class ObjectTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pickle_copy(self):
        data = {"a": 1, "b": 2}
        data_copy = pickle_copy(data)
        self.assertEqual(data, data_copy)
        self.assertNotEqual(id(data), id(data_copy))

    def test_unique(self):
        data = [1, 1, 2, 3, 4, 4]
        unique_data = unique(data)
        self.assertEqual(unique_data, [1, 2, 3, 4])

    def test_reverse_dict(self):
        data = {"a": 1, "b": 2}
        reverse_data = reverse_dict(data)
        self.assertEqual(reverse_data, {1: "a", 2: "b"})

    def test_cumsum(self):
        data = [1, 2, 3, 4]
        cumsum_data = cumsum(data)
        self.assertEqual(cumsum_data, [1, 3, 6, 10])

    def test_lag_list(self):
        data = [1, 2, 3, 4]
        lag_data = lag_list(data, 2)
        self.assertEqual(lag_data, [None, None, 1, 2])

    def test_lead_list(self):
        data = [1, 2, 3, 4]
        lead_data = lead_list(data, 2)
        self.assertEqual(lead_data, [3, 4, None, None])

    def test_row_number(self):
        data = [1, 2, 3, 4]
        row_number_data = row_number(data)
        self.assertEqual(row_number_data, [0, 1, 2, 3])
        row_number_data = row_number(data, 2)
        self.assertEqual(row_number_data, [2, 3, 4, 5])

    def test_flatten(self):
        data = [[1, 2], [3, 4]]
        flatten_data = flatten(data)
        self.assertEqual(flatten_data, [1, 2, 3, 4])

    def test_data_frame_to_list(self):
        try:
            import pandas as pd
        except ImportError:
            return True
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        lst = df_to_list(df)
        self.assertEqual(lst, [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}])

    def test_data_frame_to_dict(self):
        file = normalize_path("./data/test.xlsx")
        data = load_excel(file)
        dict_data = df_to_dict(data)
        self.assertEqual(
            dict_data,
            {"Sheet1": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]},
        )
