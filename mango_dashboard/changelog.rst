All notable changes to the mango_dashboard project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[1.0.3] - 2026-10-27
--------------------

Changed
-------
- Depend on ``mango>=1.0.3,<2.0.0`` and ``mango-time-series>=1.0.3,<2.0.0``
- Added upper bounds to all dependencies to prevent breakages from future major releases

Removed
-------
- Dropped Python 3.10 support (end of life since 2026-10); ``requires-python`` is now ``>=3.11`` (Pillow fixes require 3.11+)

Security
--------
- ``certifi>=2026.10.19,<2027.0.0`` (was ``<2025.0.0``; compromised emSign intermediate CA keys)
- ``pyarrow`` 25.0.1 via ``mango-time-series`` 1.0.3 (was pinned to 22.0.0)
- ``tornado>=6.5.5`` and ``urllib3>=2.8.0``
- Refreshed ``uv.lock`` to patched versions of vulnerable dependencies (Pillow, GitPython, tornado, pyasn1, protobuf, fsspec, streamlit)

[1.0.2] - 2025-10-13
--------------------

Fixed
-----
- Resolved deprecated pkg_resources warning in time series data processing
- Fixed editable parameter defaulting to 0 in file_explorer
- Fixed Spanish character rendering and alphabet support
- Updated importlib.resources usage to eliminate deprecation warnings
- Fixed confusing import alias that caused false positive warnings

Changed
-------
- Migrated from deprecated pkg_resources API to modern importlib.resources
- Updated data_processing.py to use proper importlib.resources syntax
- Enhanced CLI functionality for both file_explorer and time_series modules
- Improved README documentation with accurate configuration information


[1.0.1] - 2025-09-30
--------------------

Added
-----
- Updated mango dependency version
- Interactive web-based dashboards using Streamlit
- File explorer with hierarchical navigation
- Time series data visualization and analysis
- Data upload and processing capabilities
- Multi-language support (English and Spanish)
- Custom UI components and templates
- File handling for various formats (CSV, Excel, JSON)
- Responsive design for different screen sizes
- Enhanced dashboard functionality and user interface
- Improved file exploration capabilities
- Better time series visualization tools

Changed
-------
- Updated dependencies and package configuration
- Enhanced test coverage and documentation

Features
--------
- File management and exploration
- Time series data analysis and plotting
- Interactive data visualization
- Data processing and transformation
- Export and sharing capabilities
- User-friendly interface design
- Cross-platform compatibility
