from unittest import TestCase

from mango.table.table_tools import (
    mean,
    is_subset,
    str_key,
    to_len,
    join_lists,
    cumsum,
    invert_dict_list,
    simplify,
)
from pytups import TupList


class TestTools(TestCase):
    default_list = [1, 3, 5, 7, 9]

    def test_mean(self):
        self.assertEqual(mean(1, 2, 3), 2)
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(mean([1, 2, 3], 2), 2)
        self.assertIsNone(mean([]))

    def test_is_subset_true(self):
        set_a = [1, 2, 3, 4, 4, 5]
        set_b = {1, 2, 3, 4, 5, 6, 7}
        set_c = [1, 8]
        set_d = {"1", "2"}
        set_e = {"a", "1", "2", "3"}
        self.assertTrue(is_subset(set_a, set_b))
        self.assertFalse(is_subset(set_b, set_a))
        self.assertFalse(is_subset(set_c, set_b))
        self.assertFalse(is_subset(set_d, set_b))
        self.assertTrue(is_subset(set_d, set_e))

    def test_str_key(self):
        self.assertEqual(str_key({}), {})
        self.assertEqual(str_key({1: "a", 2: "b"}), {"1": "a", "2": "b"})

    def test_str_key_warning(self):
        # Commented due to conflict with pyomo
        # self.assertWarns(SyntaxWarning, str_key, {"1": "a", 1: "b"})
        result = str_key({"1": "a", 1: "b"})
        expected = {"1": "b"}
        self.assertEqual(result, expected)

    def test_to_len(self):
        self.assertEqual(to_len(1, 0), [])
        self.assertEqual(to_len([1], 2), [1, 1])
        self.assertEqual(to_len(1, 5), [1, 1, 1, 1, 1])
        self.assertEqual(to_len("one", 2), ["one", "one"])

    def test_lo_len_list(self):
        self.assertEqual(to_len([1, 2, 3], 5), [1, 2, 3])

    def test_join_lists(self):
        result = join_lists([1, 2], [3, 4], [5, 6])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_join_lists2(self):
        result = join_lists([1, 2], 3, 4, [5, 6])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_cumsum(self):
        self.assertEqual(cumsum([1, 2, 3]), [1, 3, 6])

    def test_invert_dict_list(self):
        dl = TupList([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
        result = invert_dict_list(dl)
        expected = {"a": [1, 2], "b": [2, 3]}
        self.assertEqual(result, expected)

    def test_simplify(self):
        self.assertEqual(simplify([1]), 1)
        self.assertEqual(simplify([1, 2]), [1, 2])
        self.assertEqual(simplify([1, 1, 2]), [1, 2])
        self.assertEqual(simplify(1), 1)
