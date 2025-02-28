import io
import os
from unittest import TestCase, mock

import numpy as np
from mango.processing import row_number
from mango.table.pytups_table import Table
from mango.table.table_tools import mean
from mango.tests.const import normalize_path
from pytups import TupList, SuperDict


class TestTable(TestCase):
    default_data = [
        {"Name": "Albert", "Age": 20},
        {"Name": "Bernard", "Age": 25},
        {"Name": "Charlie", "Age": 30},
        {"Name": "Daniel", "Age": 35},
    ]

    default_data2 = [
        {"Name": "Albert", "Age": 20, "Male": True, "Points": 5, "Under_25": True},
        {"Name": "Bernard", "Age": 25, "Male": True, "Points": 8, "Under_25": True},
        {"Name": "Charlie", "Age": 30, "Male": True, "Points": 4, "Under_25": False},
        {"Name": "Daniel", "Age": 35, "Male": True, "Points": 6, "Under_25": False},
    ]

    df_id = [
        {"Name": "Albert", "Id": 1},
        {"Name": "Bernard", "Id": 2},
        {"Name": "Charlie", "Id": 3},
        {"Name": "Elisa", "Id": 4},
    ]

    long_df = [
        {"Name": "Albert", "variable": "Male", "value": True},
        {"Name": "Bernard", "variable": "Male", "value": True},
        {"Name": "Charlie", "variable": "Male", "value": True},
        {"Name": "Daniel", "variable": "Male", "value": True},
        {"Name": "Albert", "variable": "Age", "value": 20},
        {"Name": "Bernard", "variable": "Age", "value": 25},
        {"Name": "Charlie", "variable": "Age", "value": 30},
        {"Name": "Daniel", "variable": "Age", "value": 35},
        {"Name": "Albert", "variable": "Points", "value": 5},
        {"Name": "Bernard", "variable": "Points", "value": 8},
        {"Name": "Charlie", "variable": "Points", "value": 4},
        {"Name": "Daniel", "variable": "Points", "value": 6},
        {"Name": "Albert", "variable": "Under_25", "value": True},
        {"Name": "Bernard", "variable": "Under_25", "value": True},
        {"Name": "Charlie", "variable": "Under_25", "value": False},
        {"Name": "Daniel", "variable": "Under_25", "value": False},
    ]

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def assertStdout(self, func, expected_output, mock_stdout, msg=None):
        """
        Assert that function func print the expected output with argument arg

        :param func: a function
        :param expected_output: the expected output
        :param mock_stdout: passed automatically by mock.patch
        :param msg: message
        """
        func()
        self.assertEqual(mock_stdout.getvalue(), expected_output, msg=msg)

    def test_tuplist_table(self):
        # Test that the table still have tuplist method and stay a table.
        table1 = Table(self.default_data).vapply(lambda v: {**v, **{"a": 1}})
        self.assertIsInstance(table1, Table, msg="still table after vapply")
        table2 = Table(self.default_data).kapply(lambda k: k)
        self.assertIsInstance(table2, Table, msg="still table after kapply")
        table3 = Table(self.default_data).kvapply(lambda k, v: {**v, **{"n": k}})
        self.assertIsInstance(table3, Table, msg="still table after kvapply")
        table4 = Table(self.default_data).copy_shallow()
        self.assertIsInstance(table4, Table, msg="still table after copy_shallow")
        table5 = Table(self.default_data).copy_deep()
        self.assertIsInstance(table5, Table, msg="still table after copy_deep")
        table6 = Table(self.default_data).vfilter(lambda v: v["Name"] == "Albert")
        self.assertIsInstance(table6, Table, "still table after vfilter")

    def test_unique_error(self):
        def unique_error():
            return Table(self.default_data).unique()

        def unique2_error():
            return Table(self.default_data).unique2()

        self.assertEqual(len(unique_error()), 4)
        self.assertEqual(len(unique2_error()), 4)

    def test_take(self):
        result = Table(self.default_data).take(["Name", "Age"])
        expected = TupList(
            [("Albert", 20), ("Bernard", 25), ("Charlie", 30), ("Daniel", 35)]
        )
        self.assertIsInstance(result, TupList, msg="take create a TupList")
        self.assertEqual(result, expected, msg="take works as expected")

    def test_take_tup(self):
        result = Table(self.default_data).take(("Name", "Age"))
        expected = TupList(
            [("Albert", 20), ("Bernard", 25), ("Charlie", 30), ("Daniel", 35)]
        )
        self.assertIsInstance(result, TupList, msg="take create a TupList")
        self.assertEqual(result, expected, msg="take works as expected")

    def test_take_tup_2(self):
        result = Table(self.default_data).take("Name", "Age")
        expected = TupList(
            [("Albert", 20), ("Bernard", 25), ("Charlie", 30), ("Daniel", 35)]
        )
        self.assertIsInstance(result, TupList, msg="take create a TupList")
        self.assertEqual(result, expected, msg="take works as expected")

    def test_mutate_on_tuplist(self):
        msg = "mutate transform a list of tuple into a list of dict"
        result = Table([(1, 2), (3, 4), (5, 6)]).mutate(a=5)
        expected = [{0: 1, 1: 2, "a": 5}, {0: 3, 1: 4, "a": 5}, {0: 5, 1: 6, "a": 5}]
        self.assertEqual(result, expected, msg=msg)

    def test_mutate_constant(self):
        msg = "mutate with a constant for the new column."
        df = Table(self.default_data).mutate(const=5)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "const": 5},
                {"Name": "Bernard", "Age": 25, "const": 5},
                {"Name": "Charlie", "Age": 30, "const": 5},
                {"Name": "Daniel", "Age": 35, "const": 5},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_mutate_constant_len_1(self):
        msg = "mutate with a constant for the new column."
        df = Table([{"Name": "Albert", "Age": 20}]).mutate(const=5)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "const": 5},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_mutate_vect(self):
        msg = "mutate with a vector for the new column."
        points = [5, 8, 4, 6]
        df = Table(self.default_data).mutate(points=points)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "points": 5},
                {"Name": "Bernard", "Age": 25, "points": 8},
                {"Name": "Charlie", "Age": 30, "points": 4},
                {"Name": "Daniel", "Age": 35, "points": 6},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_mutate_vect_wrong_len(self):
        msg = "mutate with a vector of wrong length raise error"
        points = [5, 8, 4]

        def try_mutate():
            return Table(self.default_data).mutate(points=points)

        self.assertRaises(TypeError, try_mutate, msg=msg)

    def test_mutate_func(self):
        msg = "mutate with a function for the new column."
        df = Table(self.default_data).mutate(under_25=lambda v: v["Age"] <= 25)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "under_25": True},
                {"Name": "Bernard", "Age": 25, "under_25": True},
                {"Name": "Charlie", "Age": 30, "under_25": False},
                {"Name": "Daniel", "Age": 35, "under_25": False},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_mutate_exist(self):
        msg = "mutate with a function for an existing column."
        df = Table(self.default_data).mutate(Age=lambda v: v["Age"] * 2)
        expected = Table(
            [
                {"Name": "Albert", "Age": 40},
                {"Name": "Bernard", "Age": 50},
                {"Name": "Charlie", "Age": 60},
                {"Name": "Daniel", "Age": 70},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_mutate_multiple(self):
        msg = "mutate multiple existing columns."
        df = Table(self.default_data).mutate(
            Male=True, Points=[5, 8, 4, 6], Under_25=lambda v: v["Age"] <= 25
        )
        expected = Table(
            [
                {
                    "Name": "Albert",
                    "Age": 20,
                    "Male": True,
                    "Points": 5,
                    "Under_25": True,
                },
                {
                    "Name": "Bernard",
                    "Age": 25,
                    "Male": True,
                    "Points": 8,
                    "Under_25": True,
                },
                {
                    "Name": "Charlie",
                    "Age": 30,
                    "Male": True,
                    "Points": 4,
                    "Under_25": False,
                },
                {
                    "Name": "Daniel",
                    "Age": 35,
                    "Male": True,
                    "Points": 6,
                    "Under_25": False,
                },
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_mutate_safe(self):
        msg = "mutate do not change the original table."
        original_table = Table(self.default_data).copy_deep()
        Table(self.default_data).mutate(Age=lambda v: v["Age"] * 2)
        expected = Table(
            [
                {"Name": "Albert", "Age": 40},
                {"Name": "Bernard", "Age": 50},
                {"Name": "Charlie", "Age": 60},
                {"Name": "Daniel", "Age": 70},
            ]
        )
        self.assertEqual(original_table, self.default_data, msg=msg)
        self.assertNotEqual(expected, self.default_data, msg=msg)

    def test_summarise(self):
        msg = "summarise with group_by"
        df = Table(self.default_data2).summarise(
            group_by="Under_25", Points=sum, default=None
        )
        expected = Table(
            [{"Points": 13, "Under_25": True}, {"Points": 10, "Under_25": False}]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_summarise_group_by_none(self):
        msg = "summarise with group_by = None"
        df = Table(self.default_data2).summarise(
            group_by=None, Points=sum, default=None
        )
        expected = Table([{"Points": 23}])
        self.assertEqual(df, expected, msg=msg)

    def test_select(self):
        msg = "select 2 columns"
        df = Table(self.default_data2).select("Name", "Points")
        expected = Table(
            [
                {"Name": "Albert", "Points": 5},
                {"Name": "Bernard", "Points": 8},
                {"Name": "Charlie", "Points": 4},
                {"Name": "Daniel", "Points": 6},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_select_error(self):
        """unknown column raise error"""

        def try_select():
            return Table(self.default_data2).select("unknown", "Points")

        self.assertRaises(ValueError, try_select)

    def test_select_empty(self):
        msg = "select on empty table return empty table"
        df = Table().select("Name")
        self.assertEqual(df, Table(), msg=msg)

    def test_drop(self):
        msg = "drop 3 columns"
        df = Table(self.default_data2).drop("Under_25", "Points", "Male")
        expected = Table(
            [
                {"Name": "Albert", "Age": 20},
                {"Name": "Bernard", "Age": 25},
                {"Name": "Charlie", "Age": 30},
                {"Name": "Daniel", "Age": 35},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_drop_empty_table(self):
        msg = "drop on empty table"
        df = Table().drop("Under_25", "Points", "Male")

        self.assertEqual(df, Table(), msg=msg)

    def test_rename(self):
        msg = "rename a column"
        col_names = Table(self.default_data2).rename(Points="Value").get_col_names()
        expected = ["Name", "Age", "Male", "Value", "Under_25"]
        self.assertEqual(expected, col_names, msg=msg)

    def test_join_jtype_left(self):
        msg = 'join with type="left"'
        df_id = Table(self.df_id)
        df = Table(self.default_data2).join(df_id, jtype="left").select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_join_wrong_jtype(self):
        """wrong jtype creates an error"""
        df_id = Table(self.df_id)

        def try_join():
            return Table(self.default_data2).join(df_id, jtype="up")

        self.assertRaises(ValueError, try_join)

    def test_left_join(self):
        msg = "left join with other table"
        df_id = Table(self.df_id)
        df = Table(self.default_data2).left_join(df_id).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_empty_table(self):
        msg = "left join with empty table 2"
        empty_table = Table()
        df1 = Table(self.default_data2).left_join(empty_table)
        expected1 = self.default_data2
        self.assertEqual(df1, expected1, msg=msg)
        msg = "left join with empty table 1"
        df2 = Table(empty_table).left_join(self.default_data2)
        expected2 = Table()
        self.assertEqual(df2, expected2, msg=msg)

    def test_left_join_if_empty_table(self):
        msg = "left join with empty table 2 and if_empty_table_2 argument"
        empty_table = Table()
        if_empty_2 = {"Name": None, "Id": None}
        df1 = (
            Table(self.default_data2)
            .left_join(empty_table, if_empty_table_2=if_empty_2)
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": None},
                {"Name": "Bernard", "Id": None},
                {"Name": "Charlie", "Id": None},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df1, expected, msg=msg)
        msg = "left join with empty table 1 and if_empty_table_1 argument"
        if_empty_1 = {"Name": None}
        df2 = Table(empty_table).left_join(
            self.default_data2, if_empty_table_1=if_empty_1
        )
        expected2 = Table()
        self.assertEqual(df2, expected2, msg=msg)

    def test_left_join_wrong_by(self):
        """error when left join with wrong by value"""
        df_id = Table(self.df_id)

        def try_left_join():
            return Table(self.default_data2).left_join(df_id, by="Id")

        self.assertRaises(ValueError, try_left_join)

    def test_left_join2(self):
        msg = "left join with a repeated value"
        df_id = Table(self.df_id + [{"Name": "Albert", "Id": 5}])
        df = Table(self.default_data2).left_join(df_id).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Albert", "Id": 5},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_with_by_dict(self):
        msg = "left join with by as a dict"
        df_id = Table(self.df_id + [{"Name": "Albert", "Id": 5}]).rename(Name="N")
        df = (
            Table(self.default_data2)
            .left_join(df_id, by=dict(Name="N"))
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Albert", "Id": 5},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_with_by_dict2(self):
        msg = "left join with by as a dict and common keys"
        df_id = Table(self.df_id + [{"Name": "Albert", "Id": 5}]).rename(Name="N")
        df = (
            Table(self.default_data2)
            .mutate(N=1)
            .left_join(df_id, by=dict(Name="N"))
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Albert", "Id": 5},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_with_by_dict3(self):
        msg = "left join with by as a dict and common keys"
        df_id = (
            Table(self.df_id + [{"Name": "Albert", "Id": 5}])
            .rename(Name="N")
            .mutate(Name="a")
        )
        df = (
            Table(self.default_data2)
            .left_join(df_id, by=dict(Name="N"))
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Albert", "Id": 5},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_with_by_dict4(self):
        msg = "left join with by as a dict with two keys and common keys"
        df_id = (
            Table(self.df_id + [{"Name": "Albert", "Id": 5}])
            .rename(Name="N")
            .mutate(other=1)
        )
        df = (
            Table(self.default_data2)
            .mutate(N=1)
            .left_join(df_id, by=dict(Name="N", N="other"))
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Albert", "Id": 5},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_with_by_dict_with_suffix(self):
        msg = "left join with by as a dict and common keys"
        df_id = (
            Table(self.df_id + [{"Name": "Albert", "Id": 5}])
            .rename(Name="N")
            .mutate(Name="a")
        )
        df = (
            Table(self.default_data2)
            .left_join(df_id, by=dict(Name="N"), suffix=("_1", "_2"))
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Albert", "Id": 5},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_empty(self):
        msg = "left join with empty = 0"
        df_id = Table(self.df_id)
        df = Table(self.default_data2).left_join(df_id, empty=0).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": 0},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_none(self):
        msg = "left join with None values as keys"
        df_id = Table(self.df_id).add_row(Name=None, Id=5)
        df = (
            Table(self.default_data2)
            .add_row(Name=None)
            .left_join(df_id, by="Name")
            .select("Name", "Id")
        )
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
                {"Name": None, "Id": None},
            ]
        )
        self.assertEqual(expected, df, msg=msg)

    def test_right_join(self):
        msg = "right join with other table"
        df_id = Table(self.df_id)
        df = Table(self.default_data2).right_join(df_id).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Elisa", "Id": 4},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_full_join(self):
        msg = "full join with other table"
        df_id = Table(self.df_id)
        df = Table(self.default_data2).full_join(df_id).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
                {"Name": "Daniel", "Id": None},
                {"Name": "Elisa", "Id": 4},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_inner_join(self):
        msg = "inner join with other table"
        df_id = Table(self.df_id)
        df = Table(self.default_data2).inner_join(df_id).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_left_join_self(self):
        msg = "left join with itself to get all the combinations"
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).left_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2, msg=msg)

    def test_right_join_self(self):
        msg = "right join with itself to get all the combinations"
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).right_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2, msg=msg)

    def test_inner_join_self(self):
        msg = "inner join with itself to get all the combinations"
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).inner_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2, msg=msg)

    def test_full_join_self(self):
        msg = "full join with itself to get all the combinations"
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).full_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2, msg=msg)

    def test_auto_join(self):
        msg = "auto join to get all the combinations"
        df = Table(self.default_data)
        result = df.auto_join().order_by("Name_2")
        self.assertEqual(len(result), len(df) ** 2)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "Name_2": "Albert", "Age_2": 20},
                {"Name": "Bernard", "Age": 25, "Name_2": "Albert", "Age_2": 20},
                {"Name": "Charlie", "Age": 30, "Name_2": "Albert", "Age_2": 20},
                {"Name": "Daniel", "Age": 35, "Name_2": "Albert", "Age_2": 20},
                {"Name": "Albert", "Age": 20, "Name_2": "Bernard", "Age_2": 25},
                {"Name": "Bernard", "Age": 25, "Name_2": "Bernard", "Age_2": 25},
                {"Name": "Charlie", "Age": 30, "Name_2": "Bernard", "Age_2": 25},
                {"Name": "Daniel", "Age": 35, "Name_2": "Bernard", "Age_2": 25},
                {"Name": "Albert", "Age": 20, "Name_2": "Charlie", "Age_2": 30},
                {"Name": "Bernard", "Age": 25, "Name_2": "Charlie", "Age_2": 30},
                {"Name": "Charlie", "Age": 30, "Name_2": "Charlie", "Age_2": 30},
                {"Name": "Daniel", "Age": 35, "Name_2": "Charlie", "Age_2": 30},
                {"Name": "Albert", "Age": 20, "Name_2": "Daniel", "Age_2": 35},
                {"Name": "Bernard", "Age": 25, "Name_2": "Daniel", "Age_2": 35},
                {"Name": "Charlie", "Age": 30, "Name_2": "Daniel", "Age_2": 35},
                {"Name": "Daniel", "Age": 35, "Name_2": "Daniel", "Age_2": 35},
            ]
        )
        self.assertEqual(result, expected, msg=msg)

    def test_filter(self):
        msg = "filter a table"
        table = Table(self.default_data).filter(lambda v: v["Age"] == 20)
        expected = [
            {
                "Name": "Albert",
                "Age": 20,
            }
        ]
        self.assertEqual(table, expected, msg=msg)

    def test_filter_empty(self):
        msg = "filter an empty table"
        df = Table().filter(lambda v: v["Age"] == 20)
        expected = Table()
        self.assertEqual(df, expected, msg=msg)

    def test_get_col_names_fast(self):
        msg = "get column names with get_col_names"
        result = Table(self.default_data).get_col_names(fast=True)
        expected = ["Name", "Age"]
        self.assertEqual(result, expected, msg=msg)

    def test_get_col_names(self):
        msg = "get column names with get_col_names"
        result = Table(self.default_data).get_col_names(fast=True)
        expected = ["Name", "Age"]
        self.assertEqual(result, expected, msg=msg)

    def test_to_columns(self):
        msg = "pivot to a column dict"
        df = Table(self.default_data2).to_columns()
        expected = {
            "Name": ["Albert", "Bernard", "Charlie", "Daniel"],
            "Age": [20, 25, 30, 35],
            "Male": [True, True, True, True],
            "Points": [5, 8, 4, 6],
            "Under_25": [True, True, False, False],
        }
        self.assertEqual(df, expected, msg=msg)

    def test_from_columns(self):
        msg = "create a table from a column dict"
        columns = {
            "Name": ["Albert", "Bernard", "Charlie", "Daniel"],
            "Age": [20, 25, 30, 35],
            "Male": [True, True, True, True],
            "Points": [5, 8, 4, 6],
            "Under_25": [True, True, False, False],
        }
        df = Table.from_columns(columns)
        expected = Table(self.default_data2)
        self.assertEqual(df, expected, msg=msg)

    def test_get_index(self):
        msg = " get row index for a condition"
        result = Table(self.default_data).get_index(lambda v: v["Age"] <= 25)
        expected = [0, 1]
        self.assertEqual(result, expected, msg=msg)

    def test_get_index_empty(self):
        msg = "get_index return empty list when condition is not met"
        result = Table(self.default_data).get_index(lambda v: v["Age"] == 24)
        expected = []
        self.assertEqual(result, expected, msg=msg)

    def test_replace(self):
        msg = "replace a value"
        result = Table(self.default_data).replace(replacement="20", to_replace=20)
        expected = [
            {"Name": "Albert", "Age": "20"},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_replace_empty(self):
        msg = "replace missing values with 0 with replace_empty"
        table = Table(self.default_data) + [{"Name": "Elisa"}]
        result = table.replace_empty(0)
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": "Elisa", "Age": 0},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_replace_empty_last_row(self):
        msg = "replace empty detect additional columns in last row"
        table = Table(self.default_data) + [{"Name": "Elisa", "new_column": 3}]
        result = table.replace_empty(0)
        expected = [
            {"Name": "Albert", "Age": 20, "new_column": 0},
            {"Name": "Bernard", "Age": 25, "new_column": 0},
            {"Name": "Charlie", "Age": 30, "new_column": 0},
            {"Name": "Daniel", "Age": 35, "new_column": 0},
            {"Name": "Elisa", "Age": 0, "new_column": 3},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_replace_empty_dict(self):
        msg = "replace missing values on selected columns with replace_empty"
        table = Table(self.default_data) + [{}]
        result = table.replace_empty({"Name": "Elisa"})
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": "Elisa"},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_replace_empty_dict2(self):
        msg = "replace missing values on selected columns with replace_empty"
        table = Table(self.default_data) + [{}]
        result = table.replace_empty({"Name": "Elisa", "Age": 5})
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": "Elisa", "Age": 5},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_replace_empty_dict3(self):
        msg = "replace missing values on selected columns with replace_empty"
        table = Table(self.default_data) + [{"Name": None, "Age": None}]
        result = table.replace_empty({"Name": "Elisa"})
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": "Elisa", "Age": None},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_replace_nan(self):
        msg = "replace nan with 0 with repalce_nan"
        table = Table(self.default_data) + [{"Name": "Elisa", "Age": np.nan}]
        result = table.replace_nan(0)
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": "Elisa", "Age": 0},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_pivot_longer(self):
        msg = "pivot to a long df"
        df = Table(self.default_data2).pivot_longer(
            cols=["Male", "Age", "Points", "Under_25"]
        )
        expected = Table(self.long_df)
        self.assertEqual(df, expected, msg=msg)

    def test_pivot_wider(self):
        msg = "pivot to a wide df"
        df = Table(self.long_df).pivot_wider(names_from="variable", value_from="value")
        expected = Table(self.default_data2)
        self.assertEqual(df, expected, msg=msg)

    def test_drop_empty(self):
        msg = "drop rows with missing and None values with drop_empty"
        result = Table(self.default_data).mutate(Age=[20, 25, None, 35]).drop_empty()
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Daniel", "Age": 35},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_group_mutate_bad_columns(self):
        msg = "use group mutate to add columns"
        first_row = [
            {
                "Name": "New_user",
                "Age": 21,
                "Male": True,
                "Points": 1,
                "Under_25": True,
                "other_column": True,
            }
        ]
        df = Table(first_row + self.default_data2).group_mutate(
            group_by="Under_25",
            group_mean_age=lambda v: mean(v["Age"]),
            group_points=lambda v: sum(v["Points"]),
            n_group=lambda v: row_number(v["Under_25"]),
        )
        expected = [
            {
                "Name": "New_user",
                "Age": 21,
                "Male": True,
                "Points": 1,
                "other_column": True,
                "Under_25": True,
                "group_mean_age": 22.0,
                "group_points": 14,
                "n_group": 0,
            },
            {
                "Name": "Albert",
                "Age": 20,
                "Male": True,
                "Points": 5,
                "Under_25": True,
                "group_mean_age": 22.0,
                "group_points": 14,
                "n_group": 1,
                "other_column": None,
            },
            {
                "Name": "Bernard",
                "Age": 25,
                "Male": True,
                "Points": 8,
                "Under_25": True,
                "group_mean_age": 22.0,
                "group_points": 14,
                "n_group": 2,
                "other_column": None,
            },
            {
                "Name": "Charlie",
                "Age": 30,
                "Male": True,
                "Points": 4,
                "Under_25": False,
                "group_mean_age": 32.5,
                "group_points": 10,
                "n_group": 0,
            },
            {
                "Name": "Daniel",
                "Age": 35,
                "Male": True,
                "Points": 6,
                "Under_25": False,
                "group_mean_age": 32.5,
                "group_points": 10,
                "n_group": 1,
            },
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_group_mutate(self):
        msg = "use group mutate to add columns"
        df = Table(self.default_data2).group_mutate(
            group_by="Under_25",
            group_mean_age=lambda v: mean(v["Age"]),
            group_points=lambda v: sum(v["Points"]),
            n_group=lambda v: row_number(v["Under_25"]),
        )
        expected = [
            {
                "Name": "Albert",
                "Age": 20,
                "Male": True,
                "Points": 5,
                "Under_25": True,
                "group_mean_age": 22.5,
                "group_points": 13,
                "n_group": 0,
            },
            {
                "Name": "Bernard",
                "Age": 25,
                "Male": True,
                "Points": 8,
                "Under_25": True,
                "group_mean_age": 22.5,
                "group_points": 13,
                "n_group": 1,
            },
            {
                "Name": "Charlie",
                "Age": 30,
                "Male": True,
                "Points": 4,
                "Under_25": False,
                "group_mean_age": 32.5,
                "group_points": 10,
                "n_group": 0,
            },
            {
                "Name": "Daniel",
                "Age": 35,
                "Male": True,
                "Points": 6,
                "Under_25": False,
                "group_mean_age": 32.5,
                "group_points": 10,
                "n_group": 1,
            },
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_drop_empty1(self):
        msg = "drop_empty only drop rows if there are empty or None values"
        df = Table(
            self.default_data2 + [{"Name": "Elisa", "Age": 15, "Points": None}]
        ).drop_empty(["Name"])
        expected = Table(
            [
                {
                    "Name": "Albert",
                    "Age": 20,
                    "Male": True,
                    "Points": 5,
                    "Under_25": True,
                },
                {
                    "Name": "Bernard",
                    "Age": 25,
                    "Male": True,
                    "Points": 8,
                    "Under_25": True,
                },
                {
                    "Name": "Charlie",
                    "Age": 30,
                    "Male": True,
                    "Points": 4,
                    "Under_25": False,
                },
                {
                    "Name": "Daniel",
                    "Age": 35,
                    "Male": True,
                    "Points": 6,
                    "Under_25": False,
                },
                {
                    "Name": "Elisa",
                    "Age": 15,
                    "Male": None,
                    "Points": None,
                    "Under_25": None,
                },
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_drop_empty2(self):
        msg = "drop_empty drop rows with None values"
        df = Table(
            self.default_data2 + [{"Name": "Elisa", "Age": 15, "Points": None}]
        ).drop_empty(["Points"])
        expected = Table(
            [
                {
                    "Name": "Albert",
                    "Age": 20,
                    "Male": True,
                    "Points": 5,
                    "Under_25": True,
                },
                {
                    "Name": "Bernard",
                    "Age": 25,
                    "Male": True,
                    "Points": 8,
                    "Under_25": True,
                },
                {
                    "Name": "Charlie",
                    "Age": 30,
                    "Male": True,
                    "Points": 4,
                    "Under_25": False,
                },
                {
                    "Name": "Daniel",
                    "Age": 35,
                    "Male": True,
                    "Points": 6,
                    "Under_25": False,
                },
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_drop_empty3(self):
        msg = "drop_empty drop rows with missing values"
        df = Table(
            self.default_data2 + [{"Name": "Elisa", "Age": 15, "Points": None}]
        ).drop_empty(["Under_25"])
        expected = Table(
            [
                {
                    "Name": "Albert",
                    "Age": 20,
                    "Male": True,
                    "Points": 5,
                    "Under_25": True,
                },
                {
                    "Name": "Bernard",
                    "Age": 25,
                    "Male": True,
                    "Points": 8,
                    "Under_25": True,
                },
                {
                    "Name": "Charlie",
                    "Age": 30,
                    "Male": True,
                    "Points": 4,
                    "Under_25": False,
                },
                {
                    "Name": "Daniel",
                    "Age": 35,
                    "Male": True,
                    "Points": 6,
                    "Under_25": False,
                },
            ]
        )
        self.assertEqual(df, expected, msg=msg)

    def test_lag_col(self):
        msg = "lag_col creates a lag column"
        df = Table(self.default_data2).select("Name", "Points").lag_col("Points")
        expected = [
            {"Name": "Albert", "Points": 5, "lag_Points_1": None},
            {"Name": "Bernard", "Points": 8, "lag_Points_1": 5},
            {"Name": "Charlie", "Points": 4, "lag_Points_1": 8},
            {"Name": "Daniel", "Points": 6, "lag_Points_1": 4},
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_lag_col_reverse(self):
        msg = "lag_col with i=-1 creates a lead column"
        df = Table(self.default_data2).select("Name", "Points").lag_col("Points", i=-1)
        expected = [
            {"Name": "Albert", "Points": 5, "lead_Points_1": 8},
            {"Name": "Bernard", "Points": 8, "lead_Points_1": 4},
            {"Name": "Charlie", "Points": 4, "lead_Points_1": 6},
            {"Name": "Daniel", "Points": 6, "lead_Points_1": None},
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_distinct(self):
        msg = "distinct only keep the first rows with repeated values"
        df = Table(self.default_data2).distinct("Under_25")
        expected = [
            {"Name": "Albert", "Age": 20, "Male": True, "Points": 5, "Under_25": True},
            {
                "Name": "Charlie",
                "Age": 30,
                "Male": True,
                "Points": 4,
                "Under_25": False,
            },
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_distinct_empty(self):
        msg = "distinct on empty table is empty table"
        df = Table().distinct("Under_25")
        expected = []
        self.assertEqual(df, expected, msg=msg)

    def test_order_by(self):
        msg = "order_by in ascending order (default)"
        df = Table(self.default_data2).select("Name", "Points").order_by("Points")
        expected = [
            {"Name": "Charlie", "Points": 4},
            {"Name": "Albert", "Points": 5},
            {"Name": "Daniel", "Points": 6},
            {"Name": "Bernard", "Points": 8},
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_order_by_empty(self):
        msg = "order_by on empty table is empty table"
        df = Table().order_by("Points")
        expected = []
        self.assertEqual(df, expected, msg=msg)

    def test_order_by_reverse(self):
        msg = "order_by in descending order"
        df = (
            Table(self.default_data2)
            .select("Name", "Points")
            .order_by("Points", reverse=True)
        )
        expected = [
            {"Name": "Bernard", "Points": 8},
            {"Name": "Daniel", "Points": 6},
            {"Name": "Albert", "Points": 5},
            {"Name": "Charlie", "Points": 4},
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_order_by2(self):
        msg = "order_by with two columns"
        df = (
            Table(self.default_data2)
            .select("Name", "Points", "Under_25")
            .order_by(["Under_25", "Points"])
        )
        expected = [
            {"Name": "Charlie", "Points": 4, "Under_25": False},
            {"Name": "Daniel", "Points": 6, "Under_25": False},
            {"Name": "Albert", "Points": 5, "Under_25": True},
            {"Name": "Bernard", "Points": 8, "Under_25": True},
        ]
        self.assertEqual(df, expected, msg=msg)

    def test_drop_nested(self):
        msg = "drop nested columns"
        table = Table(self.default_data).mutate(
            nest=[[dict(a=1)], [dict(a=2)], [dict(a=3)], [dict(a=4)]]
        )
        result = table.drop_nested()
        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_empty_table(self):
        self.assertEqual(Table([]), [], msg="create empty table with empty list")
        self.assertEqual(Table(None), [], msg="create empty table with None")
        self.assertEqual(Table(), [], msg="create empty table with no arg")

    def test_check_empty(self):
        msg = "check on empty table"
        self.assertEqual(Table([], check=True), [], msg=msg)
        self.assertEqual(Table(None, check=True), [], msg=msg)
        self.assertEqual(Table(check=True), [], msg=msg)

    def test_check_error(self):
        # test errors if check = True and bad data.
        self.assertRaises(
            TypeError, Table, data=[1, 2, 3], check=True, msg="check detect int list"
        )
        self.assertRaises(
            TypeError, Table, data=[(1, 2)], check=True, msg="check detect tuple list"
        )
        self.assertRaises(
            TypeError, Table, data=["hello"], check=True, msg="check detect string"
        )
        self.assertRaises(TypeError, Table, data=1, check=True, msg="check detect int")
        self.assertRaises(
            TypeError, Table, data=dict(test=True), check=True, msg="check detect dict"
        )

    def test_use_empty_table(self):
        msg = "empty tables can be used without error"
        table = Table().mutate(c=lambda v: v["a"] + v["b"]).summarise("a").sum_all("b")
        self.assertEqual(table.len(), 0, msg=msg)
        self.assertEqual(table.get_col_names(), [], msg="empty table has no col names")

    def test_add_row(self):
        msg = "add a row with add_row"
        table = Table(self.default_data)
        table2 = table.add_row(Name="new")

        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": "new", "Age": None},
        ]
        self.assertEqual(table2, expected, msg=msg)

    def test_add_row_empty(self):
        msg = "add an empty row with add_row"
        table = Table(self.default_data)
        table2 = table.add_row()

        expected = [
            {"Name": "Albert", "Age": 20},
            {"Name": "Bernard", "Age": 25},
            {"Name": "Charlie", "Age": 30},
            {"Name": "Daniel", "Age": 35},
            {"Name": None, "Age": None},
        ]
        self.assertEqual(table2, expected, msg=msg)

    def test_sorted_exception(self):
        """sorted raise error"""
        table = Table(self.default_data)

        def sort_table():
            return table.sorted()

        self.assertRaises(NotImplementedError, sort_table)

    def test_peek(self):
        msg = "peek create a nice print"
        table = Table(self.default_data)

        def try_peek():
            return table.peek(1, "test_peek")

        expected = (
            "test_peek: Table (4 rows, , 2 columns):\n"
            "0 {'Name': 'Albert', 'Age': 20}\n"
            "...\n"
            "1 {'Name': 'Bernard', 'Age': 25}\n"
            "...\n"
            "3 {'Name': 'Daniel', 'Age': 35}\n\n"
        )
        self.assertStdout(try_peek, expected, msg=msg)

    def test_peek_high_n(self):
        msg = "if n is higher than len/3, print the entire table"
        table = Table(self.default_data)

        def try_peek():
            return table.peek(3)

        expected = (
            "Table (4 rows, 2 columns):\n"
            "0 {'Name': 'Albert', 'Age': 20}\n"
            "1 {'Name': 'Bernard', 'Age': 25}\n"
            "2 {'Name': 'Charlie', 'Age': 30}\n"
            "3 {'Name': 'Daniel', 'Age': 35}\n\n"
        )
        self.assertStdout(try_peek, expected, msg=msg)

    def test_peek_empty(self):
        msg = "peek on empty table"
        table = Table()

        def try_peek():
            return table.peek(3)

        expected = "Empty table\n"
        self.assertStdout(try_peek, expected, msg=msg)

    def test_print(self):
        msg = "print a table"
        table = Table(self.default_data)

        def try_print():
            print(table)

        expected = (
            "Table (4 rows, 2 columns):\n"
            "0 {'Name': 'Albert', 'Age': 20}\n"
            "1 {'Name': 'Bernard', 'Age': 25}\n"
            "2 {'Name': 'Charlie', 'Age': 30}\n"
            "3 {'Name': 'Daniel', 'Age': 35}\n\n"
        )
        self.assertStdout(try_print, expected, msg=msg)

    def test_print_list(self):
        msg = "print a table"
        table = Table([1, 2, 3])

        def try_print():
            print(table)

        expected = "[1, 2, 3]\n"
        self.assertStdout(try_print, expected, msg=msg)

    def test_show_row(self):
        msg = "show row show one row if n2=None"
        table = Table(self.default_data)
        result = table.show_rows(1)
        expected = "1 {'Name': 'Bernard', 'Age': 25}\n"
        self.assertEqual(result, expected, msg)

    def test_head(self):
        msg = "head(2) return the two first rows"
        table = Table(self.default_data)
        result = table.head(2)
        expected = [{"Name": "Albert", "Age": 20}, {"Name": "Bernard", "Age": 25}]
        self.assertEqual(result, expected, msg=msg)
        msg2 = "head return the entire table if n > len(table)"
        self.assertEqual(table.head(100), table, msg=msg2)

    def test_group_by(self):
        msg = "group_by create a dict of lists"
        result = Table(self.default_data2).group_by("Under_25")
        expected = {
            True: [
                {
                    "Name": "Albert",
                    "Age": 20,
                    "Male": True,
                    "Points": 5,
                    "Under_25": True,
                },
                {
                    "Name": "Bernard",
                    "Age": 25,
                    "Male": True,
                    "Points": 8,
                    "Under_25": True,
                },
            ],
            False: [
                {
                    "Name": "Charlie",
                    "Age": 30,
                    "Male": True,
                    "Points": 4,
                    "Under_25": False,
                },
                {
                    "Name": "Daniel",
                    "Age": 35,
                    "Male": True,
                    "Points": 6,
                    "Under_25": False,
                },
            ],
        }
        self.assertEqual(result, expected, msg=msg)

    def test_sum_all(self):
        msg = "sum_all sum every columns"
        result = Table(self.default_data2).drop("Name").sum_all("Under_25")
        expected = [
            {"Age": 45, "Male": 2, "Points": 13, "Under_25": True},
            {"Age": 65, "Male": 2, "Points": 10, "Under_25": False},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_sum_all_no_group(self):
        msg = "sum_all sum every columns"
        result = Table(self.default_data2).drop("Name").sum_all(None)
        expected = [
            {"Age": 110, "Male": 4, "Points": 23, "Under_25": 2},
        ]
        self.assertEqual(result, expected, msg=msg)

    def test_to_set2(self):
        msg1 = "to_set2 on one column creates a list of values"
        result1 = Table(self.default_data2).to_set2("Points")
        expected1 = [8, 4, 5, 6]
        msg2 = "to_set2 on 2 columns creates a list of tuples"
        result2 = Table(self.default_data2).to_set2(["Age", "Points"])
        expected2 = [(20, 5), (30, 4), (25, 8), (35, 6)]
        self.assertEqual(result1, expected1, msg=msg1)
        self.assertEqual(result2, expected2, msg=msg2)

    def test_to_set2_empty(self):
        msg1 = "to_set2 on empty table returns empty table"
        result1 = Table().to_set2("Points")
        expected1 = []
        self.assertEqual(result1, expected1, msg=msg1)

    def test_to_param(self):
        msg1 = "to_param creates a dict"
        result1 = Table(self.default_data2).to_param("Name", "Points")
        expected1 = {"Albert": 5, "Bernard": 8, "Charlie": 4, "Daniel": 6}
        msg2 = "to_param with two keys creates a dict with a tuple as key"
        result2 = Table(self.default_data2).to_param(
            ["Male", "Under_25"], "Name", is_list=True
        )
        expected2 = {
            (True, True): ["Albert", "Bernard"],
            (True, False): ["Charlie", "Daniel"],
        }

        def to_param_error():
            return Table(self.default_data2).to_param(
                ["Male", "Under_25"], "Name", is_list=False
            )

        self.assertEqual(result1, expected1, msg=msg1)
        self.assertEqual(result2, expected2, msg=msg2)
        self.assertRaises(ValueError, to_param_error)

    def test_to_param_empty(self):
        msg = "to_param on empty table return empty dict"
        self.assertEqual(Table().to_param("Name", "Points"), SuperDict(), msg=msg)

    def test_is_unique_true(self):
        msg = "is_unique returns True if all values of the column are unique"
        result = Table(self.default_data2).is_unique("Name")
        self.assertTrue(result, msg=msg)

    def test_is_unique_false(self):
        msg = "is_unique returns False if some values of the column are repeated"
        result = Table(self.default_data2).is_unique("Under_25")
        self.assertFalse(result, msg=msg)

    def test_format_dataset(self):
        msg = "format_dataset return a dict of Table"
        data = dict(t1=self.default_data, t2=self.default_data2, p1=dict(a=3, b=5))
        result = Table.format_dataset(data)
        self.assertIsInstance(result, dict, msg=msg)
        self.assertIsInstance(result["t1"], Table, msg=msg)
        self.assertIsInstance(result["t2"], Table, msg=msg)
        self.assertIsInstance(
            result["p1"], dict, msg="format_dataset keeps dict as dict"
        )

    def test_from_pandas(self):
        try:
            import pandas as pd
        except ImportError:
            return True
        msg = "from_pandas create a Table from a pandas df"
        df = pd.DataFrame.from_records(self.default_data2)
        result = Table.from_pandas(df)
        self.assertEqual(result, self.default_data2, msg=msg)
        self.assertIsInstance(result, Table)

        # Test pandas ImportError
        with mock.patch.dict("sys.modules", {"pandas": None}):
            with self.assertRaises(
                ImportError,
            ) as context:
                Table.from_pandas(df)

            self.assertEqual(
                str(context.exception),
                "Pandas is not present in your system. Try: pip install pandas",
            )

    def test_to_pandas(self):
        try:
            import pandas as pd
        except ImportError:
            return True
        table = Table(self.default_data2)
        result = table.to_pandas()
        expected = pd.DataFrame.from_records(self.default_data2)
        pd.testing.assert_frame_equal(result, expected)

        # Test pandas ImportError
        with mock.patch.dict("sys.modules", {"pandas": None}):
            with self.assertRaises(
                ImportError,
            ) as context:
                table.to_pandas()

            self.assertEqual(
                str(context.exception),
                "Pandas is not present in your system. Try: pip install pandas",
            )

    def test_to_from_json(self):
        msg = "to_json and from_json allow to write and load json files into a Table"
        path = normalize_path("./data/table_to_json.json")
        table = Table(self.default_data2)
        table.to_json(path)
        table2 = Table.from_json(path)
        self.assertEqual(table, table2, msg=msg)
        os.remove(path)

    def test_to_from_pk(self):
        msg = "pk_save and pk_load allow to write and load pickle files into a Table"
        path = normalize_path("./data/table_to_pk.pickle")
        table = Table(self.default_data2)
        table.pk_save(path)
        table2 = Table.pk_load(path)
        self.assertEqual(table, table2, msg=msg)
        os.remove(path)

    def test_dataset_from_json(self):
        msg = "dataset_from_json creates a dict of Tables from a json file"
        path = normalize_path("./data/json_dataset.json")
        result = Table.dataset_from_json(path)
        expected = dict(t1=Table(self.default_data), t2=Table(self.default_data2))
        self.assertEqual(result, expected, msg=msg)

    def test_apply(self):
        msg = "apply a function to a table"
        result = Table(self.default_data2).apply(len)
        expected = len(Table(self.default_data2))
        self.assertEqual(result, expected, msg=msg)

    def test_col_apply(self):
        msg = "apply a function to a column"
        result = Table(self.default_data).col_apply("Age", str)
        expected = [
            {"Name": "Albert", "Age": "20"},
            {"Name": "Bernard", "Age": "25"},
            {"Name": "Charlie", "Age": "30"},
            {"Name": "Daniel", "Age": "35"},
        ]
        self.assertEqual(result, expected, msg=msg)


def test_col_apply_2(self):
    msg = "apply a function to a list of column"
    result = (
        Table(self.default_data)
        .col_apply("Age", str)
        .col_apply(["Age", "Name"], lambda v: v[0])
    )
    expected = [
        {"Name": "A", "Age": "2"},
        {"Name": "B", "Age": "2"},
        {"Name": "C", "Age": "3"},
        {"Name": "D", "Age": "3"},
    ]
    self.assertEqual(result, expected, msg=msg)
