from unittest import TestCase

from mango.processing import (
    unique,
    row_number,
    reverse_dict,
    as_list,
    flatten,
    lag_list,
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

    def test_as_list(self):
        self.assertEqual(as_list([]), [])
        self.assertEqual(as_list(1), [1])
        self.assertEqual(as_list("one"), ["one"])
        self.assertEqual(as_list(["one", "two"]), ["one", "two"])
        self.assertEqual(as_list((1, 2)), [1, 2])
        self.assertEqual(as_list([1, 2]), [1, 2])
        self.assertEqual(as_list({1, 2}), [1, 2])
        self.assertEqual(as_list({"a": 2}), [{"a": 2}])

    def test_flatten(self):
        result = flatten([[1, 2], [3, 4], [5, 6]])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_flatten2(self):
        result = flatten([[1, 2], 3, 4, [5, 6]])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_row_number(self):
        result = row_number([5, 3, 2, 5, "", [], None])
        expected = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected)

    def test_lag_list(self):
        lst = [1, 2, 3, 4, 5, 6]
        result1 = lag_list(lst)
        expected1 = [None, 1, 2, 3, 4, 5]
        result2 = lag_list(lst, 3)
        expected2 = [None, None, None, 1, 2, 3]
        self.assertEqual(result1, expected1)
        self.assertEqual(result2, expected2)
