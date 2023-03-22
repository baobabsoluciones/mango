from unittest import TestCase, main
from mango.table.pytups_table import Table
from mango.table.table_tools import mean
from mango.processing import row_number


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

    def test_mutate_constant(self):
        # test mutate with a constant for the new column.
        df = Table(self.default_data).mutate(const=5)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "const": 5},
                {"Name": "Bernard", "Age": 25, "const": 5},
                {"Name": "Charlie", "Age": 30, "const": 5},
                {"Name": "Daniel", "Age": 35, "const": 5},
            ]
        )
        self.assertEqual(df, expected)

    def test_mutate_vect(self):
        # test mutate with a vector for the new column.
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
        self.assertEqual(df, expected)

    def test_mutate_func(self):
        # test mutate with a function for the new column.
        df = Table(self.default_data).mutate(under_25=lambda v: v["Age"] <= 25)
        expected = Table(
            [
                {"Name": "Albert", "Age": 20, "under_25": True},
                {"Name": "Bernard", "Age": 25, "under_25": True},
                {"Name": "Charlie", "Age": 30, "under_25": False},
                {"Name": "Daniel", "Age": 35, "under_25": False},
            ]
        )
        self.assertEqual(df, expected)

    def test_mutate_exist(self):
        # test mutate with a function for an existing column.
        df = Table(self.default_data).mutate(Age=lambda v: v["Age"] * 2)
        expected = Table(
            [
                {"Name": "Albert", "Age": 40},
                {"Name": "Bernard", "Age": 50},
                {"Name": "Charlie", "Age": 60},
                {"Name": "Daniel", "Age": 70},
            ]
        )
        self.assertEqual(df, expected)

    def test_mutate_multiple(self):
        # test mutate with a function for an existing column.
        df = Table(self.default_data).mutate(Male=True, Points=[5,8,4,6], Under_25=lambda v: v["Age"] <= 25)
        expected = Table(
            [
                {'Name': 'Albert', 'Age': 20, 'Male': True, 'Points': 5, 'Under_25': True},
                 {'Name': 'Bernard', 'Age': 25, 'Male': True, 'Points': 8, 'Under_25': True},
                 {'Name': 'Charlie', 'Age': 30, 'Male': True, 'Points': 4, 'Under_25': False},
                 {'Name': 'Daniel', 'Age': 35, 'Male': True, 'Points': 6, 'Under_25': False}
            ]
        )
        self.assertEqual(df, expected)

    def test_mutate_safe(self):
        # test that mutate do not change the original table.
        original_table = Table(self.default_data).copy_deep()
        df = Table(self.default_data).mutate(Age=lambda v: v["Age"] * 2, _safe=True)
        expected = Table(
            [
                {"Name": "Albert", "Age": 40},
                {"Name": "Bernard", "Age": 50},
                {"Name": "Charlie", "Age": 60},
                {"Name": "Daniel", "Age": 70},
            ]
        )
        self.assertEqual(original_table, self.default_data)
        self.assertNotEqual(expected, self.default_data)

    def test_mutate_safe_no_copy(self):
        # test that mutate do not change the original table. Use _safe=False
        original_table = Table(self.default_data).copy_deep()
        df = Table(self.default_data).mutate(Age=lambda v: v["Age"] * 2, _safe=False)
        expected = Table(
            [
                {"Name": "Albert", "Age": 40},
                {"Name": "Bernard", "Age": 50},
                {"Name": "Charlie", "Age": 60},
                {"Name": "Daniel", "Age": 70},
            ]
        )
        self.assertEqual(original_table, self.default_data)
        self.assertNotEqual(expected, self.default_data)

    def test_summarise(self):
        # test summarising points for under or below 25 years old.
        df = Table(self.default_data2).summarise(
            group_by="Under_25", Points=sum, default=None
        )
        expected = Table(
            [{"Points": 13, "Under_25": True}, {"Points": 10, "Under_25": False}]
        )
        self.assertEqual(df, expected)

    def test_select(self):
        # Test selecting 2 columns.
        df = Table(self.default_data2).select("Name", "Points")
        expected = Table(
            [
                {"Name": "Albert", "Points": 5},
                {"Name": "Bernard", "Points": 8},
                {"Name": "Charlie", "Points": 4},
                {"Name": "Daniel", "Points": 6},
            ]
        )
        self.assertEqual(df, expected)

    def test_drop(self):
        # test dropping columns.
        df = Table(self.default_data2).drop("Under_25", "Points", "Male")
        expected = Table(
            [
                {"Name": "Albert", "Age": 20},
                {"Name": "Bernard", "Age": 25},
                {"Name": "Charlie", "Age": 30},
                {"Name": "Daniel", "Age": 35},
            ]
        )
        self.assertEqual(df, expected)

    def test_rename(self):
        # test renaming columns.
        col_names = Table(self.default_data2).rename(Points="Value").get_col_names()
        expected = ["Name", "Age", "Male", "Value", "Under_25"]
        self.assertEqual(col_names, expected)

    def test_left_join(self):
        # test a left join.
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
        self.assertEqual(df, expected)

    def test_left_join2(self):
        # test a left join with a repeated value.
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
        self.assertEqual(df, expected)

    def test_left_join_empty(self):
        # test a left join with empty = 0
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
        self.assertEqual(df, expected)

    def test_right_join(self):
        # test a right join
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
        self.assertEqual(df, expected)

    def test_full_join(self):
        # test a full join
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
        self.assertEqual(df, expected)

    def test_inner_join(self):
        # test a inner join
        df_id = Table(self.df_id)
        df = Table(self.default_data2).inner_join(df_id).select("Name", "Id")
        expected = Table(
            [
                {"Name": "Albert", "Id": 1},
                {"Name": "Bernard", "Id": 2},
                {"Name": "Charlie", "Id": 3},
            ]
        )
        self.assertEqual(df, expected)

    def test_left_join_self(self):
        # test a left join with itself to get all the combinations
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).left_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2)

    def test_right_join_self(self):
        # test a right join with itself to get all the combinations
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).right_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2)

    def test_inner_join_self(self):
        # test an inner join with itself to get all the combinations
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).inner_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2)

    def test_full_join_self(self):
        # test a full join with itself to get all the combinations
        df = Table(self.default_data).mutate(dummy=1)
        result = Table(df).full_join(df, by="dummy")
        self.assertEqual(len(result), len(df) ** 2)

    def test_to_columns(self):
        # test pivoting to a column dict
        df = Table(self.default_data2).to_columns()
        expected = {
            "Name": ["Albert", "Bernard", "Charlie", "Daniel"],
            "Age": [20, 25, 30, 35],
            "Male": [True, True, True, True],
            "Points": [5, 8, 4, 6],
            "Under_25": [True, True, False, False],
        }
        self.assertEqual(df, expected)

    def test_from_columns(self):
        # test creating a table from a column dict
        columns = {
            "Name": ["Albert", "Bernard", "Charlie", "Daniel"],
            "Age": [20, 25, 30, 35],
            "Male": [True, True, True, True],
            "Points": [5, 8, 4, 6],
            "Under_25": [True, True, False, False],
        }
        df = Table.from_columns(columns)
        expected = Table(self.default_data2)
        self.assertEqual(df, expected)

    def test_pivot_longer(self):
        # test pivoting to a long df
        df = Table(self.default_data2).pivot_longer(
            cols=["Male", "Age", "Points", "Under_25"]
        )
        expected = Table(self.long_df)
        self.assertEqual(df, expected)

    def test_pivot_wider(self):
        # test pivoting to a wide df
        df = Table(self.long_df).pivot_wider(names_from="variable", value_from="value")
        expected = Table(self.default_data2)
        self.assertEqual(df, expected)

    def test_group_mutate(self):
        # test a group mutate
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
        self.assertEqual(df, expected)

    def test_drop_empty1(self):
        # test drop_empty on column without empty values
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
        self.assertEqual(df, expected)

    def test_drop_empty2(self):
        # test drop_empty on column with empty values
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
        self.assertEqual(df, expected)

    def test_drop_empty3(self):
        # test drop_empty on column with missing column
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
        self.assertEqual(df, expected)

    def test_lag_col(self):
        # test creating a lag column
        df = Table(self.default_data2).select("Name", "Points").lag_col("Points")
        expected = [
            {"Name": "Albert", "Points": 5, "lag_Points_1": None},
            {"Name": "Bernard", "Points": 8, "lag_Points_1": 5},
            {"Name": "Charlie", "Points": 4, "lag_Points_1": 8},
            {"Name": "Daniel", "Points": 6, "lag_Points_1": 4},
        ]
        self.assertEqual(df, expected)

    def test_lag_col_reverse(self):
        # test creating a lead column
        df = Table(self.default_data2).select("Name", "Points").lag_col("Points", i=-1)
        expected = [
            {"Name": "Albert", "Points": 5, "lead_Points_1": 8},
            {"Name": "Bernard", "Points": 8, "lead_Points_1": 4},
            {"Name": "Charlie", "Points": 4, "lead_Points_1": 6},
            {"Name": "Daniel", "Points": 6, "lead_Points_1": None},
        ]
        self.assertEqual(df, expected)

    def test_distinct(self):
        # test distinct
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
        self.assertEqual(df, expected)

    def test_order_by(self):
        # test order_by in ascending order (default)
        df = Table(self.default_data2).select("Name", "Points").order_by("Points")
        expected = [
            {"Name": "Charlie", "Points": 4},
            {"Name": "Albert", "Points": 5},
            {"Name": "Daniel", "Points": 6},
            {"Name": "Bernard", "Points": 8},
        ]
        self.assertEqual(df, expected)

    def test_oder_by_reverse(self):
        # test order_by in descending order
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
        self.assertEqual(df, expected)

    def test_order_by2(self):
        # test order_by with two columns
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
        self.assertEqual(df, expected)

    def test_empty_table(self):
        # test creation of empty table
        self.assertEqual(Table([]), [])
        self.assertEqual(Table(None), [])
        self.assertEqual(Table(), [])

    def test_check_empty(self):
        # test check on empty table
        self.assertEqual(Table([], check=True), [])
        self.assertEqual(Table(None, check=True), [])
        self.assertEqual(Table(check=True), [])

    def test_check_error(self):
        # test errors if check = True and bad data.
        self.assertRaises(TypeError, Table, data=[1, 2, 3], check=True)
        self.assertRaises(TypeError, Table, data=[(1, 2)], check=True)
        self.assertRaises(TypeError, Table, data=["hello"], check=True)
        self.assertRaises(TypeError, Table, data=1, check=True)
        self.assertRaises(TypeError, Table, data=dict(test=True), check=True)

    def test_use_empty_table(self):
        table = Table().mutate(lambda v: v["a"] + v["b"]).summarise("a").sum_all("b")
        self.assertEqual(table.len(), 0)
