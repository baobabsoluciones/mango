Mango
------

.. image:: https://img.shields.io/pypi/v/mango?label=version&logo=python&logoColor=white&style=for-the-badge&color=E58164
   :alt: PyPI
   :target: https://pypi.org/project/mango/
.. image:: https://img.shields.io/pypi/l/mango?color=blue&style=for-the-badge
  :alt: PyPI - License
  :target: https://github.com/baobabsoluciones/mango/blob/master/LICENSE
.. image:: https://img.shields.io/github/actions/workflow/status/baobabsoluciones/mango/build_docs.yml?label=docs&logo=github&style=for-the-badge
   :alt: GitHub Workflow Status
   :target: https://github.com/baobabsoluciones/mango/actions
.. image:: https://img.shields.io/codecov/c/gh/baobabsoluciones/mango?flag=unit-tests&label=coverage&logo=codecov&logoColor=white&style=for-the-badge&token=0KKRF3J95L
    :alt: Codecov
    :target: https://app.codecov.io/gh/baobabsoluciones/mango

This repository contains multiple libraries developed by the team at Baobab Soluciones: **mango**, **mango_autoencoder**, **mango_calendar**, **mango_dashboard**, **mango_genetic**, and **mango_time_series**. These libraries are the result of the team's experience, understanding, and knowledge of the Python language and its ecosystem.

The functions are divided in different modules so even though everything is imported the dependencies can be installed only for the modules that are needed.

Overview
--------

**Mango** is the core library providing essential tools for data processing, analysis, and machine learning workflows. The ecosystem includes specialized libraries for different domains:

- **mango_autoencoder**: Neural autoencoders for anomaly detection in time series
- **mango_calendar**: Calendar data, holidays, and date-related features with Spanish holiday support
- **mango_dashboard**: Interactive web applications for data exploration and visualization
- **mango_genetic**: Genetic algorithms and evolutionary computation methods
- **mango_time_series**: Comprehensive time series analysis, forecasting, and feature engineering

Each library is designed to work independently or as part of the integrated Mango ecosystem, providing flexibility in dependency management and usage.

Quick Start
-----------

.. code-block:: python

   # Core data processing
   import mango
   
   # Time series anomaly detection
   from mango_autoencoder import AutoEncoder
   
   # Calendar and holiday data
   from mango_calendar import get_calendar
   
   # Genetic algorithms
   from mango_genetic import Population, GeneticBaseConfig
   
   # Time series analysis
   from mango_time_series import TimeSeriesAnalyzer

Installation
============

**Using uv:**

.. code-block:: bash

   uv add mango
   uv add mango-autoencoder
   uv add mango-calendar
   uv add mango-dashboard
   uv add mango-genetic
   uv add mango-time-series

**Using pip:**

.. code-block:: bash

   pip install mango
   pip install mango-autoencoder
   pip install mango-calendar
   pip install mango-dashboard
   pip install mango-genetic
   pip install mango-time-series

Documentation
=============

Full documentation is available at: https://mango.readthedocs.io/

Individual library documentation:

- `Mango Core <https://mango.readthedocs.io/en/latest/code_mango/index.html>`_
- `Mango Autoencoder <https://mango.readthedocs.io/en/latest/code_autoencoder/index.html>`_
- `Mango Calendar <https://mango.readthedocs.io/en/latest/code_mango_calendar/index.html>`_
- `Mango Dashboard <https://mango.readthedocs.io/en/latest/code_mango_dashboard/index.html>`_
- `Mango Genetic <https://mango.readthedocs.io/en/latest/code_mango_genetic/index.html>`_
- `Mango Time Series <https://mango.readthedocs.io/en/latest/code_mango_time_series/index.html>`_

Contributing
============

We welcome contributions! Please see our `contributing guidelines <https://github.com/baobabsoluciones/mango/blob/master/CONTRIBUTING.rst>`_ for more details.

Discussion and Development
==========================

We encourage open discussion and collaboration on development via GitHub issues. If you have ideas, suggestions, or encounter any issues, please feel free to open an issue on our `GitHub repository <https://github.com/baobabsoluciones/mango/issues>`_.

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/baobabsoluciones/mango/blob/master/LICENSE>`_ file for details.

Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <mailto:mango@baobabsoluciones.es>`_