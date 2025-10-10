All notable changes to the mango_dashboard project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[1.0.2] - 2024-12-27
--------------------

Fixed
-----
- Resolved deprecated pkg_resources warning in time series data processing
- Updated importlib.resources usage to eliminate deprecation warnings
- Fixed confusing import alias that caused false positive warnings

Changed
-------
- Migrated from deprecated pkg_resources API to modern importlib.resources
- Updated data_processing.py to use proper importlib.resources syntax

[1.0.1] - 2024-12-26
--------------------

Added
-----
- Updated mango dependency version

[1.0.0] - 2024-12-24
--------------------

Added
-----
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
