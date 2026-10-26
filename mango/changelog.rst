All notable changes to the mango project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[1.0.3] - 2026-10-26
--------------------

Changed
-------
- Allow ``numpy>=1.24.4,<3.0.0`` and align dependency constraints with the rest of the Mango packages
- Reduced duplicate calls in table methods to improve performance on large tables
- Removed error logs from ``date_functions``, which already raise exceptions (avoids misleading logs in ``as_str``)

Fixed
-----
- ``summarise`` and ``sum_all`` returning unexpected results when a column name is a substring of another

Security
--------
- Raised ``certifi`` to ``>=2026.10.19,<2027.0.0`` (was ``<2025.0.0``) (compromised emSign intermediate CA keys)
- Raised ``requests`` minimum to 2.33.0 (CVE-2026-25645)
- Raised ``urllib3`` minimum to 2.8.0 (CVE-2026-97687, CVE-2026-97688, CVE-2026-97689)

[1.0.1] - 2025-09-29
--------------------

Added
-----
- Updated mango dependency version
- Multi-module project structure with standalone packages
- Integration with uv package manager alongside pip
- Enhanced dependency management with optional dependency groups
- Improved test coverage and modernized test infrastructure
- Comprehensive documentation system with Sphinx and mock imports
- Independent CI/CD workflows for each module
- PyPI publishing automation for all modules

Changed
-------
- Migrated from pkg_resources to importlib.metadata for dependency checking
- Updated project documentation to reflect multi-module architecture
- Consolidated pyproject.toml configuration files
- Enhanced codecov.yml with coverage flags for all modules
- Migrated from pip to uv for faster dependency management
- Restructured codebase into independent, focused modules
- Updated all documentation to reStructuredText format

Removed
-------
- Obsolete functions and deprecated code
- Outdated documentation formats
- Unused configuration files

Fixed
-----
- TOML parsing errors in pyproject.toml
- Test failures related to dynamic versioning
- ModuleNotFoundError for pkg_resources in tests
- Incorrect mocking in test_requirement_check.py
- Documentation build issues with heavy dependencies
- PyPI publishing workflow failures

