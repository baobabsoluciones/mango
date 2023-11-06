version 0.1.0
--------------

- **released**: 2023-11-06
- **description**: first beta release of the library.
- **changelog**:
    - fix bug fix on left join of Table class.
    - added improvements to Table class.
    - adapt load_excel function to return a list of dictionaries.
    - added new date functions.
    - drop images module.
    - removed pandas as a main dependency of the library. It is still needed in some submodules.
    - changed from setup.py to pyproject.toml.
    - added calendar datasets for Spain.
    - added a REST client to the AEMET to gather meteorological data.
    - added pydantic validation as a decorator that can be used in any function.


version 0.0.6
--------------

- **released**: 2023-05-31
- **description**: improvements to Table class

version 0.0.5
--------------

- **released**: 2023-04-26
- **description**: fixed some bugs in Table class.
- **changelog**:
    - fixed error on mutate when table had one row.
    - allow group by None in sum_all and summarise
    - as_list applied on a dict returns a list of the dict instead of the keys.

version 0.0.4
--------------

- **released**: 2023-04-19
- **description**: logger classes and methods now can receive a logger name to be used
- **changelog**:
    - Chrono class now receives a logger name to be used
    - `log_time` decorator now receives a logger name to be used

version 0.0.3
--------------

- **released**: 2023-04-18
- **description**: minor changes to logging class Chrono and to the default logger
- **changelog**:
    - minor changes to logging class Chrono: stop now reports the duration
    - default logger has a info level set up

version 0.0.2
--------------

- **released**: 2023-04-11
- **description**: added class Table and added direct requests for arcgis od matrix calculation
- **changelog**:
    - added class Table
    - added direct requests for arcgis od matrix calculationThe request can be made in two modes, sync and async. It defaults to sync and changes to async in case the requests has more than 10 origins or destinations.
    - minor fix to arcgis client

version 0.0.1
--------------

- **released**: 2023-03-09
- **description**: initial version of mango library
