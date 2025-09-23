Mango Calendar
==============

A Python library for working with calendar data, holidays, and date-related features with a focus on Spanish holidays and autonomous communities.

Features
--------

**Spanish Holiday Support**
- Comprehensive support for Spanish national and autonomous community holidays

**Multi-Country Support**
- Built on the ``holidays`` library for international calendar support

**Flexible Data Formats**
- Support for both pandas and polars DataFrames

**Weighted Communities**
- Assign weights to different autonomous communities

**Distance Calculations**
- Calculate distances to holidays for time-series analysis

**Calendar Events**
- Support for commercial events like Black Friday

**COVID Lockdown Data**
- Built-in COVID-19 lockdown period data

**Pivot Functionality**
- Transform calendar data into pivot tables

**Type Hints**
- Full type annotation support for better IDE integration

Installation
------------

**Using uv:**

.. code-block:: bash

   # Create virtual environment with Python 3.11
   uv venv --python 3.11
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   uv add mango-calendar

**Using pip:**

.. code-block:: bash

   pip install mango-calendar

Quick Start
-----------

.. code-block:: python

   from mango_calendar import get_calendar

   # Get Spanish holidays for 2024
   holidays_df = get_calendar(country="ES", start_year=2024, end_year=2025)
   print(holidays_df.head())

Usage Examples
--------------

Basic Holiday Calendar
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mango_calendar import get_calendar

   # Get basic Spanish holidays
   holidays = get_calendar(
       country="ES",
       start_year=2024,
       end_year=2025
   )
   print(holidays[['date', 'name']].head())

Advanced Features
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get holidays with autonomous communities and weights
   holidays_with_communities = get_calendar(
       country="ES",
       start_year=2024,
       end_year=2025,
       communities=True,
       return_weights=True,
       calendar_events=True  # Include Black Friday
   )

   # Calculate distances to holidays
   holidays_with_distances = get_calendar(
       country="ES",
       start_year=2024,
       end_year=2025,
       communities=True,
       return_distances=True,
       distances_config={
           "steps_back": 7,
           "steps_forward": 7
       }
   )

Date Utilities
~~~~~~~~~~~~~~

.. code-block:: python

   from mango_calendar.date_utils import get_holidays_df, get_covid_lockdowns

   # Get holidays in a specific format with window bounds
   holidays_polars = get_holidays_df(
       steps_back=3,
       steps_forward=7,
       start_year=2024,
       country="ES",
       output_format="polars"
   )

   # Get COVID lockdown periods
   covid_data = get_covid_lockdowns()

Pivot Calendar Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a pivot table of holidays
   pivot_calendar = get_calendar(
       country="ES",
       start_year=2024,
       end_year=2025,
       communities=True,
       return_weights=True,
       pivot=True,
       pivot_keep_communities=True
   )

API Reference
-------------

``get_calendar()``
~~~~~~~~~~~~~~~~~~

Main function to retrieve calendar data.

**Parameters:**

- ``country`` (str): Country code (default: "ES")
- ``start_year`` (int): Start year for the calendar (default: 2010)
- ``end_year`` (int): End year for the calendar (default: current year + 2)
- ``communities`` (bool): Include autonomous community holidays (default: False)
- ``weight_communities`` (dict): Custom weights for communities
- ``calendar_events`` (bool): Include events like Black Friday (default: False)
- ``return_weights`` (bool): Return community weights (default: None)
- ``return_distances`` (bool): Return distance calculations (default: False)
- ``distances_config`` (dict): Configuration for distance calculations
- ``name_transformations`` (bool): Apply name transformations (default: True)
- ``pivot`` (bool): Return data in pivot format (default: False)
- ``pivot_keep_communities`` (bool): Keep communities when pivoting (default: False)

**Returns:**

- ``pd.DataFrame``: Calendar data with requested features

``get_holidays_df()``
~~~~~~~~~~~~~~~~~~~~~

Get holidays dataframe with window bounds.

**Parameters:**

- ``steps_back`` (int): Days to go back from holiday
- ``steps_forward`` (int): Days to go forward from holiday
- ``start_year`` (int): Start year (default: 2014)
- ``country`` (str): Country code (default: "ES")
- ``output_format`` (str): "polars" or "pandas" (default: "polars")

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.12+
- uv (recommended) or pip

Setup
~~~~~

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd mango_calendar

   # Install dependencies
   uv sync

   # Run tests
   python run_tests.py

Project Structure
~~~~~~~~~~~~~~~~~

::

   mango_calendar/
   ├── src/
   │   └── mango_calendar/
   │       ├── __init__.py
   │       ├── calendar_features.py
   │       └── date_utils.py
   ├── tests/
   │   ├── test_calendar_features.py
   │   ├── test_date_utils.py
   │   └── test_init.py
   ├── pyproject.toml
   └── README.rst

Dependencies
~~~~~~~~~~~~

- ``holidays>=0.76``: Holiday data
- ``numpy>=2.3.1``: Numerical operations
- ``pandas>=2.3.1``: Data manipulation
- ``polars>=1.31.0``: Fast DataFrame operations
- ``pyarrow>=20.0.0``: Columnar data format
- ``pycountry>=24.6.1``: Country information
- ``unidecode>=1.4.0``: Unicode text processing

Code Quality
------------

This project uses:

- **Ruff**: For linting and code formatting
- **Type hints**: Full type annotation support
- **pytest**: For testing
- **PEP 8**: Code style compliance
- **Docstring conventions**: PEP 257 compliance

Contributing
------------

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/awesome-feature``)
3. Commit your changes (``git commit -m 'Add awesome feature'``)
4. Push to the branch (``git push origin feature/awesome-feature``)
5. Open a Pull Request

Development Guidelines
~~~~~~~~~~~~~~~~~~~~~~

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include tests for new features
- Run ``uv run coverage run -m unittest discover -s tests -p "test_*.py" -v`` before submitting

License
-------

This project is developed by **baobab soluciones** (mango@baobabsoluciones.es).

Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <mailto:mango@baobabsoluciones.es>`_
