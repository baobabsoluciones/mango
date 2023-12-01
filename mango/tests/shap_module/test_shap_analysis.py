from unittest import TestCase

from mango.shap_analysis import ShapAnalyzer
from mango.tests.const import normalize_path


class ObjectTests(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pickle_copy(self):
        data = {"a": 1, "b": 2}
        data_copy = pickle_copy(data)
        self.assertEqual(data, data_copy)
        self.assertNotEqual(id(data), id(data_copy))
