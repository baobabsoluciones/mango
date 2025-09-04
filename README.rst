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

Installation
===========

**Using uv (recommended):**

.. code-block:: bash

   git clone https://github.com/baobabsoluciones/mango.git
   cd mango
   uv venv
   uv sync
   uv run pip install -e .

**Individual libraries with uv:**

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

Contributing
===========

We welcome contributions! Please see our `contributing guidelines <https://github.com/baobabsoluciones/mango/blob/master/CONTRIBUTING.rst>`_ for more details.

Discussion and Development
=========================

We encourage open discussion and collaboration on development via GitHub issues. If you have ideas, suggestions, or encounter any issues, please feel free to open an issue on our `GitHub repository <https://github.com/baobabsoluciones/mango/issues>`_.

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/baobabsoluciones/mango/blob/master/LICENSE>`_ file for details.
