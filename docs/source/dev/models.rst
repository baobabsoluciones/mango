Models
----------


Genetic
========

Here we have the code documentation for the main classes of the genetic algorithm: individual, population, configuration and problem.

.. autoclass:: mango.models.genetic.individual.Individual
    :members:
    :undoc-members:
    :private-members:
    :special-members: __hash__, __eq__


.. autoclass:: mango.models.genetic.population.Population
    :members:
    :undoc-members:
    :private-members:


.. autoclass:: mango.models.genetic.config.GeneticBaseConfig
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

.. autoclass:: mango.models.genetic.problem.Problem
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

There is also some other helper classes that are used by the genetic algorithm that are documented below.

.. autoclass:: mango.models.genetic.shared.exceptions.GeneticDiversity

.. autoclass:: mango.models.genetic.shared.exceptions.ConfigurationError


Neural Networks components
===========================

Activations
~~~~~~~~~~~~

.. automodule:: mango.models.activations

Neural networks
~~~~~~~~~~~~~~~~

.. automodule:: mango.models.neural_networks

Optimization
=============

Pyomo
~~~~~~

The following functions are tools made to work with optimization models created with
the `pyomo library. <http://www.pyomo.org/>`_

.. automodule:: mango.models.pyomo

Machine Learning
================

Metrics
~~~~~~~~

As a part of mango we have implemented some metrics that are used to evaluate the performance of the models. The metrics are implemented in the following module.

.. automodule:: mango.models.metrics

Enumerations
~~~~~~~~~~~~

The enumerations are used to define the type of problem and the type of model.

.. automodule:: mango.models.enums

Experiment tracking
~~~~~~~~~~~~~~~~~~~~

During the training of the models, the user may develop many models and it is important to keep track of the results.
For this purpose, we have implemented several classes that can be used to keep track of the experiments. The classes
are implemented in the following module.

The main class is the MLExperiment class. This class is used to keep track of the results of the experiments. The
MLExperiment class is used to save the results of the experiments in a folder structure and provides some methods to
analyze the results.

.. autoclass:: mango.models.experiment_tracking.MLExperiment
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

MLTracker is a class that can be used to keep track of the experiments. It is a simple manager that uses the folder
where all the experiments are saved. It provides some methods to analyze the results and compare the experiments.

.. autoclass:: mango.models.experiment_tracking.MLTracker
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:


In case does not want to use the MLExperiment class, the user can use the following function to save the results of the
trained model into a folder structure. The model is saved as a pickle file and the
data is saved as csv files. The function also saves a summary of the model in a json file. This way many models
(experiments) can be saved in the same folder and the user can easily compare them.

.. autofunction:: mango.models.export_model

The subfolder structure after running export_model is the following:

If not zipped:

.. code-block:: bash

    base_path
    |-- experiment_LinearRegression_20240111-133955
    |   `-- summary.json
    |   |-- data
    |   |   |-- X_test.csv
    |   |   |-- X_train.csv
    |   |   |-- y_test.csv
    |   |   `-- y_train.csv
    |   `-- model
    |       |-- hyperparameters.json
    |       `-- model.pkl

In case of zipped:

.. code-block:: bash

    base_path
    |-- experiment_LinearRegression_20240111-133955
    |   |-- summary.json
    |   |-- data.zip
    |   `-- model.zip


The following is an example of the summary.json file:

.. code-block:: json

    {
        "model": {
            "name": "LinearRegression",
            "problem_type": "regression",
            "input": "X_train.csv",
            "target": "y_train.csv",
            "hyperparameters": {
                "fit_intercept": true,
                "normalize": false,
                "copy_X": true,
                "n_jobs": null
            },
            "library": "sklearn"
        },
        "results": {
            "train": {
                "r2": 0.9999999999999999,
                "rmse": 0.0,
                "mae": 0.0
            },
            "test": {
                "r2": 0.9999999999999999,
                "rmse": 0.0,
                "mae": 0.0
            }
        }
    }

If save_dataset is set to True, the JSON file will also contain the following:

.. code-block:: json

        {
            "data": {
                "X_train": {
                    "path": "X_train.csv",
                    "shape": [
                        100,
                        2
                    ]
                },
                "y_train": {
                    "path": "y_train.csv",
                    "shape": [
                        100,
                        1
                    ]
                },
                "X_test": {
                    "path": "X_test.csv",
                    "shape": [
                        100,
                        2
                    ]
                },
                "y_test": {
                    "path": "y_test.csv",
                    "shape": [
                        100,
                        1
                    ]
                }
            }
        }

Model experiments