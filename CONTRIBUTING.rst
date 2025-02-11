Contributing to Mango
==================

We love your input! We want to make contributing to Mango as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

Development Process
-----------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from ``master``.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows our coding standards.
6. Issue that pull request!

Code Style Guidelines
-------------------

Python Code Style
~~~~~~~~~~~~~~~~

1. All code must be formatted using Black formatter with default settings:

   .. code-block:: bash

       # Install Black
       pip install black

       # Format a file
       black file.py

       # Format entire project
       black .

   Black ensures consistent code style across the project. Configure your IDE to run Black on save.

   Key Black formatting rules:
   - Line length is automatically set to 88 characters
   - String quotes are normalized to double quotes for multi-line strings and single quotes for others
   - No extra spaces around delimiters

2. Follow PEP 8 with these specific requirements:
   
   - Use 4 spaces for indentation
   - Maximum line length is 100 characters
   - Use f-strings for string formatting:

   .. code-block:: python

       # Good
       name = "Mango"
       message = f"Welcome to {name}!"

       # Not recommended
       message = "Welcome to {}!".format(name)
       message = "Welcome to " + name + "!"

3. Docstrings must follow reST format:

   .. code-block:: python

       def calculate_mean(numbers):
           """
           Calculate the arithmetic mean of a list of numbers.

           :param numbers: List of numerical values
           :type numbers: list[float]
           :return: Arithmetic mean of the input numbers
           :rtype: float
           :raises ValueError: If the input list is empty
           """
           if not numbers:
               raise ValueError("Cannot calculate mean of empty list")
           return sum(numbers) / len(numbers)

4. No inline comments. Use docstrings and clear variable names instead:

   .. code-block:: python

       # Good
       def process_data(raw_data):
           """
           Process raw data by filtering outliers and normalizing values.
           
           :param raw_data: Raw input data
           :type raw_data: list[float]
           :return: Processed data
           :rtype: list[float]
           """
           filtered_data = [x for x in raw_data if x > 0]
           normalized_data = [x / max(filtered_data) for x in filtered_data]
           return normalized_data

       # Not recommended
       def process_data(raw_data):
           filtered = [x for x in raw_data if x > 0]  # Remove negative values
           norm = [x / max(filtered) for x in filtered]  # Normalize to [0,1]
           return norm

Testing
-------

Unit Testing Guidelines
~~~~~~~~~~~~~~~~~~~~~

We use pytest as our testing framework. All tests should be placed in the ``tests`` directory, following the same structure as the source code.

Directory Structure
^^^^^^^^^^^^^^^^^

.. code-block:: text

    mango/
    ├── mango/
    │   └── module/
    │       └── feature.py
    └── tests/
        └── module/
            └── test_feature.py

Test File Naming
^^^^^^^^^^^^^^^

- Test files should be named ``test_*.py``
- Test classes should be named ``Test*``
- Test methods should be named ``test_*``

Writing Tests
^^^^^^^^^^^

1. Each test function should test one specific functionality:

   .. code-block:: python

       def test_calculate_mean_normal_case():
           """
           Test calculate_mean with a list of positive numbers.
           """
           numbers = [1, 2, 3, 4, 5]
           result = calculate_mean(numbers)
           assert result == 3

       def test_calculate_mean_empty_list():
           """
           Test calculate_mean raises ValueError with empty list.
           """
           with pytest.raises(ValueError) as exc_info:
               calculate_mean([])
           assert str(exc_info.value) == "Cannot calculate mean of empty list"

2. Use descriptive test names that indicate:
   - The function being tested
   - The scenario being tested
   - The expected outcome

3. Follow the Arrange-Act-Assert pattern:

   .. code-block:: python

       def test_process_data_filters_and_normalizes():
           """
           Test that process_data correctly filters negative values and normalizes.
           """
           # Arrange
           input_data = [-1, 0, 2, 4, -3, 6]
           
           # Act
           result = process_data(input_data)
           
           # Assert
           expected = [0, 0.33333, 0.66667, 1]
           assert len(result) == len(expected)
           assert all(abs(a - b) < 0.0001 for a, b in zip(result, expected))

4. Use fixtures for common setup:

   .. code-block:: python

       @pytest.fixture
       def sample_data():
           """
           Fixture providing sample data for testing.
           
           :return: Dictionary with test data
           :rtype: dict
           """
           return {
               "values": [1, 2, 3, 4, 5],
               "expected_mean": 3,
               "expected_std": 1.4142
           }

       def test_calculate_statistics(sample_data):
           """
           Test statistics calculation using fixture data.
           """
           result = calculate_statistics(sample_data["values"])
           assert result["mean"] == sample_data["expected_mean"]
           assert abs(result["std"] - sample_data["expected_std"]) < 0.0001

5. Test edge cases and error conditions:

   .. code-block:: python

       @pytest.mark.parametrize("input_data, expected_error", [
           (None, TypeError),
           ([], ValueError),
           ([1, "2", 3], TypeError),
       ])
       def test_calculate_mean_error_cases(input_data, expected_error):
           """
           Test calculate_mean with various error conditions.
           """
           with pytest.raises(expected_error):
               calculate_mean(input_data)

Running Tests
^^^^^^^^^^^

1. Run all tests:

   .. code-block:: bash

       pytest

2. Run tests with coverage:

   .. code-block:: bash

       pytest --cov=mango

3. Run specific test file:

   .. code-block:: bash

       pytest tests/module/test_feature.py

4. Run tests matching a pattern:

   .. code-block:: bash

       pytest -k "test_calculate"

Test Coverage Requirements
^^^^^^^^^^^^^^^^^^^^^^^

- All new code should have at least 90% test coverage
- Critical paths should have 100% coverage
- Run coverage reports to verify:

  .. code-block:: bash

      pytest --cov=mango --cov-report=html

Mocking and Patching
^^^^^^^^^^^^^^^^^

Use mocking for external dependencies:

.. code-block:: python

    from unittest.mock import patch, MagicMock

    def test_data_processor_with_external_api():
        """
        Test data processor with mocked API calls.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [1, 2, 3]}
        
        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_response
            result = process_external_data("example.com/api")
            
            mock_get.assert_called_once_with("example.com/api")
            assert result == [1, 2, 3]

Documentation
------------

1. Update docstrings for any modified functions
2. Update RST files in ``docs/source`` for new features
3. Add examples to the documentation when appropriate

Example of a well-documented feature:

.. code-block:: python

    def validate_date_range(start_date, end_date):
        """
        Validate that a date range is properly formatted and logical.
        
        :param start_date: Starting date of the range
        :type start_date: datetime.date
        :param end_date: Ending date of the range
        :type end_date: datetime.date
        :return: True if the range is valid
        :rtype: bool
        :raises ValueError: If end_date is before start_date
        
        :Example:
        
        >>> from datetime import date
        >>> start = date(2023, 1, 1)
        >>> end = date(2023, 12, 31)
        >>> validate_date_range(start, end)
        True
        """
        if end_date < start_date:
            raise ValueError("End date cannot be before start date")
        return True

Pull Request Process
------------------

1. Update the README.rst with details of changes to the interface
2. Update the docs/source/changelog.rst with a note describing your changes
3. The PR will be merged once you have the sign-off of two other developers

Issues
------

We use GitHub issues to track public bugs. Report a bug by opening a new issue.

Write bug reports with detail, background, and sample code:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening)

License
-------

By contributing, you agree that your contributions will be licensed under its MIT License.

References
---------

- `PEP 8 -- Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_
- `Sphinx Documentation <https://www.sphinx-doc.org/>`_
- `unittest.mock Documentation <https://docs.python.org/3/library/unittest.mock.html>`_
- `pytest Documentation <https://docs.pytest.org/>`_
- `Black Documentation <https://black.readthedocs.io/en/stable/>`_
