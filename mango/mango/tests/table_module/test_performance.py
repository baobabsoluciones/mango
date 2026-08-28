import time
from unittest import TestCase

from mango.table.pytups_tools import get_col_names
from pytups import TupList


class TestGetColNamesPerformance(TestCase):
    """Performance guard for ``get_col_names``.

    ``get_col_names`` dedups column names while scanning every row. The
    implementation uses a ``set`` for the membership test, making it
    O(rows x cols); the previous list-based version was O(rows x cols^2).

    The table below (same columns on every row, the worst case for the old
    version) is sized so the old implementation took several seconds. The new
    one runs in well under a second, so we assert an absolute upper bound with a
    large safety margin. The threshold is machine-dependent by nature; if it
    ever trips on slower hardware, raise ``MAX_SECONDS`` rather than shrinking
    the table, since shrinking it weakens the guard.
    """

    n_rows = 5000
    n_cols = 300
    MAX_SECONDS = 2.0

    @classmethod
    def setUpClass(cls):
        cls.expected_columns = [f"col_{i}" for i in range(cls.n_cols)]
        base_row = {c: 0 for c in cls.expected_columns}
        cls.table = TupList([dict(base_row) for _ in range(cls.n_rows)])

    def test_get_col_names_correctness(self):
        self.assertEqual(
            get_col_names(self.table),
            self.expected_columns,
            msg="columns must be returned in first-seen order",
        )

    def test_get_col_names_is_fast(self):
        start = time.perf_counter()
        result = get_col_names(self.table)
        elapsed = time.perf_counter() - start

        # Sanity: it actually produced the columns (not an early-out on empty).
        self.assertEqual(len(result), self.n_cols)
        self.assertLess(
            elapsed,
            self.MAX_SECONDS,
            msg=(
                f"get_col_names took {elapsed:.3f}s for "
                f"{self.n_rows}x{self.n_cols}; expected < {self.MAX_SECONDS}s. "
                "This suggests the O(rows x cols^2) regression is back."
            ),
        )
