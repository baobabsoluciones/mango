All notable changes to the mango_genetic project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[1.0.1] - 2024-12-26
--------------------

Added
-----
- Updated mango dependency version

[1.0.0] - 2024-12-24
--------------------

Added
-----
- Core genetic algorithm framework
- Individual and population management classes
- Selection, crossover, mutation, and replacement algorithms
- Configuration system for genetic algorithm parameters
- Base classes for extensible genetic algorithm implementation
- Standalone package structure for genetic algorithms module
- Integration with uv package manager
- Enhanced dependency management with numpy and mango core
- Comprehensive testing infrastructure with pytest

Changed
-------
- Migrated from integrated module to standalone package
- Updated project configuration and dependencies
- Enhanced documentation structure with theory and code separation

Features
--------
- Individual management with customizable fitness functions
- Population control and evolution mechanisms
- Multiple selection strategies
- Various crossover and mutation operators
- Configurable stopping criteria
- Modular architecture for easy extension
