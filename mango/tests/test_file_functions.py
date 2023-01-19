from unittest import TestCase

from mango.processing import (
    list_files_directory,
    load_json,
    is_json_file,
    is_excel_file,
    load_excel,
    load_excel_sheet,
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
            normalize_path("../\\requirements-dev.txt"),
            normalize_path("../\\requirements.txt"),
        ]
        files = list_files_directory(normalize_path("../"), ["txt"])
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
        pass

    def test_check_json_extension(self):
        file = normalize_path("./data/test.json")
        self.assertTrue(is_json_file(file))
        self.assertFalse(is_json_file(file + ".txt"))

    def test_check_excel_extension(self):
        file = normalize_path("./data/test.xlsx")
        self.assertTrue(is_excel_file(file))
        self.assertFalse(is_excel_file(file + ".txt"))

    def test_load_excel_sheet(self):
        pass

    def test_excel_file(self):
        pass

    def test_write_excel(self):
        pass

    def test_read_bad_excel_file(self):
        file = normalize_path("./data/test.json")
        self.assertRaises(FileNotFoundError, load_excel, file)
        self.assertRaises(FileNotFoundError, load_excel_sheet, file, None)
