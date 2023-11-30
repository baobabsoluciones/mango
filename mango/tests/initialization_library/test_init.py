import mango
from unittest import TestCase
from const import normalize_path


class InitializeTest(TestCase):
    def test_version_matches(self):
        with open(normalize_path(os.path.join("..","..","pyproject.toml")),"r") as f:
            lines = f.read_lines()