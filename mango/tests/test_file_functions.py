from unittest import TestCase

from mango.processing.file_functions import list_files_directory, load_json
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
