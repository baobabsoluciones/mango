All notable changes to the mango_calendar project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Removed
-------
- Dropped Python 3.10 support (end of life since 2026-10); ``requires-python`` is now ``>=3.11``

Security
--------
- ``pyarrow>=23.0.1,<27.0.0`` (Arrow IPC use-after-free advisory)
- Refreshed ``uv.lock`` to patched versions of vulnerable dependencies

[1.0.0] - 2025-09-24
--------------------

Added
-----
- Calendar feature extraction and manipulation
- Date utility functions for data analysis
- Holiday and working day calculations
- Business day logic and calendar operations
- Time period handling and date ranges
- Calendar data validation and processing
- Support for multiple calendar systems
- Comprehensive logging and error handling
- Enhanced calendar functionality and date utilities
- Improved holiday and working day calculations
- Better integration with the mango ecosystem

Changed
-------
- Updated dependencies and package configuration
- Enhanced test coverage and documentation

Features
--------
- Date and time manipulation utilities
- Calendar feature engineering
- Business day calculations
- Holiday detection and handling
- Time period analysis
- Date validation and formatting
- Calendar data processing
- Integration with pandas and numpy
