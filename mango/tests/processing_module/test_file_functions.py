import os
import warnings
from unittest import TestCase, mock

from mango.processing import (
    list_files_directory,
    load_json,
    is_json_file,
    is_excel_file,
    load_excel,
    load_excel_sheet,
    write_json,
    load_csv,
    write_excel,
    write_csv,
    pickle_copy,
)
from mango.processing.file_functions import (
    load_csv_light,
    write_csv_light,
    load_excel_light,
    write_excel_light,
    load_str_iterable,
)
from mango.tests.const import normalize_path


class FileTests(TestCase):
    def setUp(self):
        self.data_1 = {
            "Sheet1": [
                {"a": 1.2, "b": 4.1},
                {"a": 2.3, "b": 5.2},
                {"a": 3.1, "b": 6.4},
            ],
            "Sheet2": [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
        }
        self.records_to_df = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        pass

    def tearDown(self):
        pass

    def test_get_all_files(self):
        some_files = [
            normalize_path("../processing\\date_functions.py"),
            normalize_path("../processing\\file_functions.py"),
        ]
        files = list_files_directory(normalize_path("../processing"))
        for file in some_files:
            self.assertIn(file, files)

    def test_get_some_files(self):
        some_files = [
            normalize_path("../../\\requirements-dev.txt"),
            normalize_path("../../\\requirements.txt"),
        ]
        files = list_files_directory(normalize_path("../../"), ["txt"])
        for file in files:
            self.assertIn(normalize_path(file), some_files)

    def test_bad_folder(self):
        self.assertRaises(FileNotFoundError, list_files_directory, "./no-exists")

    def test_read_json(self):
        file = normalize_path("./data/test.json")
        data = load_json(file)
        self.assertIsInstance(data, dict)
        self.assertIn("name", data.keys())

    def test_write_json(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        file = normalize_path("./data/temp.json")
        write_json(data, file)
        data2 = load_json(file)
        self.assertEqual(data, data2)
        os.remove(file)

    def test_check_json_extension(self):
        file = normalize_path("./data/test.json")
        self.assertTrue(is_json_file(file))
        self.assertFalse(is_json_file(file + ".txt"))

    def test_check_excel_extension(self):
        file = normalize_path("./data/test.xlsx")
        self.assertTrue(is_excel_file(file))
        self.assertFalse(is_excel_file(file + ".txt"))

    def test_load_excel_sheet(self):
        try:
            import pandas as pd
        except ImportError:
            warnings.warn("pandas not installed. test cancelled")
            return True
        file = normalize_path("./data/test.xlsx")
        data = load_excel_sheet(file, sheet="Sheet1")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (2, 2))

    @mock.patch.dict("sys.modules", {"pandas": None})
    def test_load_excel_sheet_pd(self):
        file = normalize_path("./data/test.xlsx")
        with self.assertRaises(
            NotImplementedError,
        ) as context:
            load_excel_sheet(file, sheet="Sheet1")

        self.assertEqual(
            str(context.exception),
            "function not yet implemented without pandas",
        )

    def test_excel_file(self):
        import pandas as pd

        file = normalize_path("./data/test.xlsx")
        data = load_excel(file)
        self.assertIn("Sheet1", data.keys())
        self.assertIsInstance(data["Sheet1"], pd.DataFrame)
        self.assertEqual(data["Sheet1"].shape, (2, 2))

    def check_test_excel(self, data):
        self.assertIn("Sheet1", data.keys())
        self.assertIsInstance(data["Sheet1"], list)
        self.assertEqual(data["Sheet1"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    @mock.patch.dict("sys.modules", {"pandas": None})
    def test_excel_file_pd(self):
        file = normalize_path("./data/test.xlsx")
        with warnings.catch_warnings(record=True) as warning_list:
            # Run the function that may raise a warning
            data = load_excel(file)

            # Assert that the warning is raised
            self.assertEqual(len(warning_list), 1)
            self.assertEqual(
                str(warning_list[0].message),
                "pandas is not installed so load_excel_open will be used. Data can only be returned as list of dicts.",
            )
            self.check_test_excel(data)

    def test_excel_light(self):
        file = normalize_path("./data/test.xlsx")
        data = load_excel_light(file)
        self.check_test_excel(data)

    def test_excel_file_to_dict(self):
        file = normalize_path("./data/test.xlsx")
        data = load_excel(file, output="records")
        self.check_test_excel(data)

    def check_test_excel_write(self, data, data2):
        self.assertEqual(data.keys(), data2.keys())
        self.assertEqual(data["Sheet1"], data2["Sheet1"])
        self.assertEqual(data["Sheet2"], data2["Sheet2"])

    def test_write_excel(self):
        data = {
            "Sheet1": {"a": [1, 2, 3], "b": [4, 5, 6]},
            "Sheet2": [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
        }
        file = normalize_path("./data/temp.xlsx")
        write_excel(file, data)

        data2 = load_excel(file, output="records")
        self.assertEqual(data.keys(), data2.keys())
        self.assertEqual(data["Sheet2"], data2["Sheet1"])
        self.assertEqual(data["Sheet2"], data2["Sheet2"])

    @mock.patch.dict("sys.modules", {"pandas": None})
    def test_write_excel_pd(self):
        file = normalize_path("./data/temp.xlsx")

        # Test pandas ImportError
        with warnings.catch_warnings(record=True) as warning_list:
            # Run the function that may raise a warning
            write_excel(file, self.data_1)

            # Assert that the warning is raised
            self.assertEqual(len(warning_list), 1)
            self.assertEqual(
                str(warning_list[0].message),
                "pandas is not installed so write_excel_open will be used.",
            )

            # Assert that the file was written correctly
            data2 = load_excel_light(file)
            self.check_test_excel_write(self.data_1, data2)
        if os.path.exists(file):
            os.remove(file)

    def test_write_excel_light(self):
        file = normalize_path("./data/temp.xlsx")
        write_excel_light(file, self.data_1)

        data2 = load_excel_light(file)
        self.check_test_excel_write(self.data_1, data2)
        os.remove(file)

    def test_write_excel_light_close(self):
        file = normalize_path("./data/temp.xlsx")
        write_excel_light(file, self.data_1)
        os.remove(file)

    def test_write_excel_light_iter(self):
        file = normalize_path("./data/temp.xlsx")
        data = pickle_copy(self.data_1)
        data["Sheet1"] = [{"a": [1, 2], "b": ["a1", "a2"], "c": {"i": 1}}]
        write_excel_light(file, data)

        data2 = load_excel_light(file)
        self.check_test_excel_write(data2, data)
        # os.remove(file)

    def test_load_write_excel_light_sheet(self):
        file = normalize_path("./data/temp.xlsx")
        write_excel_light(file, self.data_1)

        data2 = load_excel_light(file, sheets="Sheet1")
        self.assertEqual(self.data_1["Sheet1"], data2["Sheet1"])
        self.assertFalse("Sheet2" in data2)
        os.remove(file)

    def test_bad_format_write_excel(self):
        file = normalize_path("./data/temp.xlsz")
        data = {"Sheet1": {"a": [1, 2, 3], "b": [4, 5, 6]}}
        self.assertRaises(FileNotFoundError, write_excel, file, data)

    def test_read_bad_excel_file(self):
        file = normalize_path("./data/test.json")
        self.assertRaises(FileNotFoundError, load_excel, file)
        self.assertRaises(FileNotFoundError, load_excel_sheet, file, None)

    def test_read_csv_file(self):
        import pandas as pd

        file = normalize_path("./data/test.csv")
        data = load_csv(file)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (2, 2))

    @mock.patch.dict("sys.modules", {"pandas": None})
    def test_read_csv_file_pd(self):
        file = normalize_path("./data/test.csv")

        # Test pandas ImportError
        with warnings.catch_warnings(record=True) as warning_list:
            # Run the function that may raise a warning
            data = load_csv(file)

            # Assert that the warning is raised
            self.assertEqual(len(warning_list), 1)
            self.assertEqual(
                str(warning_list[0].message),
                "pandas is not installed so load_csv_light will be used.",
            )
            self.assertIsInstance(data, list)
            self.assertEqual([{"a": 1.0, "b": "2"}, {"a": 3.0, "b": "4"}], data)

    def test_read_csv_open(self):
        file = normalize_path("./data/test.csv")
        data = load_csv_light(file)
        self.assertIsInstance(data, list)
        self.assertEqual([{"a": 1.0, "b": "2"}, {"a": 3.0, "b": "4"}], data)

    def test_read_csv_open_2(self):
        file = normalize_path("./data/test2.csv")
        data = load_csv_light(file)
        self.assertIsInstance(data, list)
        self.assertEqual([{"a": 1.2, "b": "2,5"}, {"a": 3.6, "b": "4,5"}], data)

    def test_read_bad_csv_file(self):
        file = normalize_path("./data/test.json")
        self.assertRaises(FileNotFoundError, load_csv, file)

    def test_write_csv(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        file = normalize_path("./data/temp.csv")
        write_csv(file, data)
        data2 = load_csv(file)
        self.assertEqual(data2.shape, (3, 2))
        os.remove(file)
        write_csv(file, self.records_to_df)
        data2 = load_csv(file)
        self.assertEqual(data2.shape, (3, 2))

    @mock.patch.dict("sys.modules", {"pandas": None})
    def test_write_csv_pd(self):
        file = normalize_path("./data/temp.csv")

        # Test pandas ImportError
        with warnings.catch_warnings(record=True) as warning_list:
            # Run the function that may raise a warning
            write_csv(file, self.records_to_df)

            # Assert that the warning is raised
            self.assertEqual(len(warning_list), 1)
            self.assertEqual(
                str(warning_list[0].message),
                "pandas is not installed so write_csv_light will be used.",
            )
            data2 = load_csv_light(file)
            self.assertEqual(data2, self.records_to_df)
            os.remove(file)

    def test_write_csv_light(self):
        file = normalize_path("./data/temp.csv")
        write_csv_light(file, self.records_to_df)
        data2 = load_csv_light(file)
        self.assertEqual(data2, self.records_to_df)
        os.remove(file)

    def test_write_csv_light_2(self):
        file = normalize_path("./data/temp.csv")
        write_csv_light(file, self.records_to_df, sep=";")
        data2 = load_csv_light(file)
        self.assertEqual(self.records_to_df, data2)
        os.remove(file)

    def test_write_csv_bad_format(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        file = normalize_path("./data/temp.xlsz")
        self.assertRaises(FileNotFoundError, write_csv, file, data)

    def test_load_str_iterable(self):
        """
        load_str_iterable should transform string representing pythons objects
        like lists and dict into the relevant object.
        Other strings should be left unchanged.
        """
        expected = [1, 2, 3]
        result = load_str_iterable("[1,2,3]")
        self.assertEqual(expected, result)

        expected = {1: 2}
        result = load_str_iterable("{1:2}")
        self.assertEqual(expected, result)

        expected = "column name"
        result = load_str_iterable("column name")
        self.assertEqual(expected, result)

        expected = "12/11/2024"
        result = load_str_iterable("12/11/2024")
        self.assertEqual(expected, result)

        expected = 12
        result = load_str_iterable(12)
        self.assertEqual(expected, result)
