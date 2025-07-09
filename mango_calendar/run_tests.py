#!/usr/bin/env python3
"""Simple test runner for mango-calendar unit tests."""

import sys
import unittest


def run_tests():
    """Run all unit tests in the tests directory."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = "tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code based on results
    if result.wasSuccessful():
        print("\nAll tests passed!")
        return 0
    else:
        print(
            f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors"
        )
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
