All notable changes to the mango_time_series project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[1.0.3] - 2026-10-27
--------------------

Changed
-------
- Allow ``numpy>=1.24.4,<3.0.0`` (was ``<2.0.0``)
- Depend on ``mango>=1.0.3,<2.0.0`` instead of the exact pin ``mango==1.0.2``

Removed
-------
- Dropped Python 3.10 support (end of life since 2026-10); ``requires-python`` is now ``>=3.11``

Security
--------
- Require ``mango>=1.0.3``
- ``pyarrow>=23.0.1,<26.0.0`` instead of pinned ``pyarrow==22.0.0`` (Arrow IPC use-after-free advisory; pyarrow 26 requires NumPy 2)
- Refreshed ``uv.lock`` to patched versions of vulnerable dependencies

[1.0.1] - 2025-09-29
--------------------

Added
-----
- Updated mango dependency version
- Core time series data structures and classes
- Exploratory analysis tools for time series data
- Data validation and processing utilities
- Moving averages and time feature extraction
- Seasonal decomposition and stationarity tests
- Heteroscedasticity analysis
- Custom cross-validation for time series forecasting
- Comprehensive logging and error handling
- Enhanced time series analysis capabilities
- Improved data processing and validation functions
- Better integration with the mango ecosystem

Changed
-------
- Updated dependencies and package configuration
- Enhanced test coverage and documentation

Features
--------
- Time series data loading and manipulation
- Statistical analysis and visualization
- Feature engineering for machine learning
- Forecasting metrics and evaluation
- Calendar and holiday handling
- Efficient data processing with pandas and numpy

