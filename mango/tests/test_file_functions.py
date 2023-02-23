from unittest import TestCase
import os
import pandas as pd
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
    write_df_to_excel,
    write_csv,
    write_df_to_csv,
)
from mango.tests.const import normalize_path


class FileTests(TestCase):
    def setUp(self):
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
        file = normalize_path("./data/test.xlsx")
        data = load_excel_sheet(file, sheet="Sheet1")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (2, 2))

    def test_excel_file(self):
        file = normalize_path("./data/test.xlsx")
        data = load_excel(file)
        self.assertIn("Sheet1", data.keys())
        self.assertIsInstance(data["Sheet1"], pd.DataFrame)
        self.assertEqual(data["Sheet1"].shape, (2, 2))

    def test_write_excel(self):
        data = {
            "Sheet1": {"a": [1, 2, 3], "b": [4, 5, 6]},
            "Sheet2": [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}],
        }
        file = normalize_path("./data/temp.xlsx")
        write_excel(file, data)

        data2 = load_excel(file)
        self.assertEqual(data.keys(), data2.keys())
        os.remove(file)

    def test_write_df_to_excel(self):
        df = pd.DataFrame.from_records(
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        )
        file = normalize_path("./data/temp.xlsx")
        write_df_to_excel(file, df)
        data2 = load_excel(file)
        self.assertEqual(df.shape, data2["Sheet1"].shape)
        os.remove(file)

    def test_write_df_to_excel_bad_extension(self):
        df = pd.DataFrame.from_records(
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        )
        file = normalize_path("./data/temp.xlsz")
        self.assertRaises(FileNotFoundError, write_df_to_excel, file, df)

    def test_bad_format_write_excel(self):
        file = normalize_path("./data/temp.xlsz")
        data = {"Sheet1": {"a": [1, 2, 3], "b": [4, 5, 6]}}
        self.assertRaises(FileNotFoundError, write_excel, file, data)

    def test_read_bad_excel_file(self):
        file = normalize_path("./data/test.json")
        self.assertRaises(FileNotFoundError, load_excel, file)
        self.assertRaises(FileNotFoundError, load_excel_sheet, file, None)

    def test_read_csv_file(self):
        file = normalize_path("./data/test.csv")
        data = load_csv(file)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (2, 2))

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
        data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        write_csv(file, data)
        data2 = load_csv(file)
        self.assertEqual(data2.shape, (3, 2))
        os.remove(file)

    def test_write_csv_bad_format(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        file = normalize_path("./data/temp.xlsz")
        self.assertRaises(FileNotFoundError, write_csv, file, data)

    def test_write_df_to_csv(self):
        df = pd.DataFrame.from_records(
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        )
        file = normalize_path("./data/temp.csv")
        write_df_to_csv(file, df)
        data2 = load_csv(file)
        self.assertEqual(df.shape, data2.shape)
        os.remove(file)

    def test_write_df_to_csv_bad_extension(self):
        df = pd.DataFrame.from_records(
            [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        )
        file = normalize_path("./data/temp.xlsz")
        self.assertRaises(FileNotFoundError, write_df_to_csv, file, df)
