"""Unit tests for __init__.py module."""

import unittest

from mango_calendar import hello


class TestInit(unittest.TestCase):
    """Test cases for __init__.py module."""

    def test_hello_function(self) -> None:
        """Test the hello function."""
        result = hello()

        # Check if result is a string
        self.assertIsInstance(result, str)

        # Check the expected content
        self.assertEqual(result, "Hello from mango-calendar!")

    def test_hello_function_return_type(self) -> None:
        """Test that hello function returns the correct type."""
        result = hello()

        # Check type annotation consistency
        self.assertIsInstance(result, str)

        # Check that it's not empty
        self.assertGreater(len(result), 0)

    def test_hello_function_content(self) -> None:
        """Test the specific content of hello function."""
        result = hello()

        # Check that it contains expected words
        self.assertIn("Hello", result)
        self.assertIn("mango-calendar", result)

        # Check that it starts with "Hello"
        self.assertTrue(result.startswith("Hello"))

        # Check that it ends with "!"
        self.assertTrue(result.endswith("!"))


if __name__ == "__main__":
    unittest.main()
