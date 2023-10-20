from unittest import TestCase

from pytups import TupList

from mango.processing import (
    unique,
    row_number,
    reverse_dict,
    as_list,
    flatten,
    cumsum,
    lag_list,
)
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


class TestTools(TestCase):

    default_list = [1, 3, 5, 7, 9]

    def test_unique(self):
        self.assertEqual(unique([]), [])
        self.assertEqual(unique([1, 2, 3]), [1, 2, 3])
        self.assertEqual(unique([1, 2, 3, 1]), [1, 2, 3])

    def test_unique_str(self):
        result = unique(["1", "b", "d", "f", "f"])
        expected = ["1", "b", "d", "f"]
        self.assertEqual(set(result), set(expected))

    def test_reverse_dict(self):
        result = reverse_dict({"a": 1, "b": 2})
        expected = {1: "a", 2: "b"}
        self.assertEqual(result, expected)

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

    def test_as_list(self):
        self.assertEqual(as_list([]), [])
        self.assertEqual(as_list(1), [1])
        self.assertEqual(as_list("one"), ["one"])
        self.assertEqual(as_list(["one", "two"]), ["one", "two"])
        self.assertEqual(as_list((1, 2)), [1, 2])
        self.assertEqual(as_list([1, 2]), [1, 2])
        self.assertEqual(as_list({1, 2}), [1, 2])
        self.assertEqual(as_list({"a":2}), [{"a":2}])

    def test_str_key(self):
        self.assertEqual(str_key({}), {})
        self.assertEqual(str_key({1: "a", 2: "b"}), {"1": "a", "2": "b"})

    def test_str_key_warning(self):
        self.assertWarns(SyntaxWarning, str_key, {"1": "a", 1: "b"})
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

    def test_flatten(self):
        result = flatten([[1, 2], [3, 4], [5, 6]])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_flatten2(self):
        result = flatten([[1, 2], 3, 4, [5, 6]])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_mean(self):
        self.assertEqual(mean(1, 2, 3), 2)
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(mean([1, 2, 3], 2), 2)
        self.assertIsNone(mean([]))

    def test_row_number(self):
        result = row_number([5, 3, 2, 5, "", [], None])
        expected = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_cumsum(self):
        self.assertEqual(cumsum([1, 2, 3]), [1, 3, 6])

    def test_lag_list(self):
        lst = [1, 2, 3, 4, 5, 6]
        result1 = lag_list(lst)
        expected1 = [None, 1, 2, 3, 4, 5]
        result2 = lag_list(lst, 3)
        expected2 = [None, None, None, 1, 2, 3]
        self.assertEqual(result1, expected1)
        self.assertEqual(result2, expected2)

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
