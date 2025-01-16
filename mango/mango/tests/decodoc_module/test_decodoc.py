from unittest import TestCase

import pandas as pd

from mango.decodoc import decodoc, get_dict


class DecodocTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_decodoc(self):
        input_df = pd.DataFrame({"column1": ["a", "b", "c"], "column2": [1, 2, 3]})

        @decodoc(["input_df"], ["output_df"])
        def foo(input_df):
            """This function does nothing."""
            return input_df.copy()

        _ = foo(input_df)
        info_dict = get_dict()

        for value in info_dict.values():
            self.assertEqual(value["name"], "foo")
            self.assertEqual(value["caller"], "test_decodoc")
            self.assertEqual(value["docstring"], "This function does nothing.")
