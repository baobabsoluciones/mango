#!/usr/bin/env python3
"""Simple test runner for mango-calendar unit tests."""

import sys
import unittest


def run_tests() -> int:
    """Run all unit tests in the tests directory."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll tests passed!")
        return 0
    else:
        print(
            f"\nTests failed: {len(result.failures)} failures, "
            f"{len(result.errors)} errors"
        )
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
