Version 0.2.3
==============

**Released on Jul 23, 2024**

Modified dependency versions in order to not have install issues.


Version 0.2.2
==============

**Released on Jul 4, 2024**

Improvements to Table class and bugfixes:

**New features:**

- improvements to Table class.
- arcgis geolocation calls allow to set up the country.
- arcgis sync distance calculation gives back the correct value for the meters.

Version 0.2.1
==============

**Released on Jan 8, 2024.**

Small bugfixes on genetic algorithms and code reordering for documentation.


Version 0.2.0
==============

**Released on Dec 21, 2023.**

This version includes a new module for genetic algorithms and more!

**New features:**

- Added genetic algorithms that can be used in a wide range of problems.
- Added a new module to perform SHAP analysis.
- Added a function to pivot the calendar.
- Added correlation calculation between series.

**Bugfixes:**

- Minor bugfixes.

Version 0.1.3
==============

**Released on Nov 17, 2023.**

This version includes a small bugfix.

**Bugfixes:**

- Fixed a bug on save button with unique keys on the dashboard.

Version 0.1.2
==============

**Released on Nov 17, 2023.**

This version includes a function to calculate distances with the haversine formula.

**New features:**

- Added a function to calculate distances with the haversine formula.
- Added the possibility to edit excels, csv and json files in the streamlit app.

**Bugfixes:**

- Fixed a bug on streamlit app saving the config file.

Version 0.1.1
==============

**Released on Nov 8, 2023.**

This version includes a small bugfix.

**Bugfixes:**

- Fixed a bug on left join on class Table.
- Fixed a small bug on AEMET pydantic validation.

Version 0.1.0
==============

**Released on Nov 7, 2023.**

This version includes new modules and bugfixes.

**New features:**

- Added improvements to ``Table`` class.
- Adapted ``load_excel`` function to return a list of dictionaries.
- Added new date functions.
- Added a REST client to the AEMET to gather meteorological data.
- Added pydantic validation as a decorator that can be used in any function.
- Added calendar datasets for Spain.
- Moved from setup.py to pyproject.toml.

**Bugfixes:**

- Fixed a bug on left join on class Table.

**Breaking changes:**

- Removed pandas as a main dependency of the library. It is still needed in some submodules.
- Drop images module.

Version 0.0.6
==============

**Released on May 31, 2023.**

Improvements to Table class

Version 0.0.5
==============

**Released on Apr 26, 2023.**

Fixed some bugs on ``Table`` class.

**Bugfixes:**

- Fixed error on ``mutate`` when table had one row.
- Allow group by None in ``sum_all`` and ``summarise``.
- ``as_list`` applied on a dict returns a list of the dict instead of the keys.

Version 0.0.4
==============

**Released on Apr 19, 2023.**

This versions includes changes to the logger class and methods.

**New features:**

- ``Chrono`` class can receive a logger name to be used.
- ``log_time`` decorator can receive a logger name to be used.

Version 0.0.3
==============

**Released on Apr 18, 2023.**

Minor changes to logging class ``Chrono`` and to the default logger.

**New features:**

- Minor changes to logging class ``Chrono``: ``stop`` now reports the duration.
- Default logger has a info level set up.

Version 0.0.2
==============

**Released on Apr 11, 2023.**

This version includes a new class ``Table`` and a new module to perform requests to ArcGIS.

**New features:**

- Added class ``Table``.
- Added direct requests for arcgis od matrix calculation.

Version 0.0.1
==============

Released on Mar 9, 2023.

This is the first version of mango library.
