import os
from unittest import TestCase

from mango.tests.const import normalize_path

import mango


class InitializeTest(TestCase):
    def test_version_matches(self):
        with open(normalize_path(os.path.join("..", "..", "pyproject.toml")), "r") as f:
            pyproject = f.read()
        version = pyproject.split("version = ")[1].split("\n")[0].replace('"', "")
        self.assertEqual(mango.__version__, version)
