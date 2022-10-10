from unittest import TestCase

from processing.file_functions import list_files_directory


class FileTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_all_files(self):
        some_files = [
            "./processing\\date_functions.py",
            "./processing\\file_functions.py",
        ]
        files = list_files_directory("./processing")
        for file in some_files:
            self.assertIn(file, files)

    def test_get_some_files(self):
        some_files = [".\\requirements-dev.txt", ".\\requirements.txt"]
        files = list_files_directory(".", ["txt"])
        for file in some_files:
            self.assertIn(file, some_files)

    def test_bad_folder(self):
        files = list_files_directory("./no-exists")
        self.assertEqual(files, None)
