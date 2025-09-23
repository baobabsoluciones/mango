Changelog
=========

All notable changes to the mango project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
-----
- Multi-module project structure with standalone packages
- Integration with uv package manager alongside pip
- Enhanced dependency management with optional dependency groups
- Improved test coverage and modernized test infrastructure

Changed
-------
- Migrated from pkg_resources to importlib.metadata for dependency checking
- Updated project documentation to reflect multi-module architecture
- Consolidated pyproject.toml configuration files
- Enhanced codecov.yml with coverage flags for all modules

Fixed
-----
- TOML parsing errors in pyproject.toml
- Test failures related to dynamic versioning
- ModuleNotFoundError for pkg_resources in tests
- Incorrect mocking in test_requirement_check.py

