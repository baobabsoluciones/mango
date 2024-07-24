Experiment Tracking
-------------------

This section describes how to use the experiment tracking system.

We will use the california housing dataset from sklearn as an example.

.. code-block:: python

    from sklearn.datasets import fetch_california_housing
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, random_state=0, test_size=0.5)

Now we will create a simple pipeline to train a linear regression model and wrap it in an instance of :class:`MLExperiment<mango.models.experiment_tracking.MLExperiment>`

.. code-block:: python

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from mango.models import MLExperiment
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    experiment = MLExperiment(
        model=pipeline,
        name='California Housing LinearRegression',
        description='LinearRegression on California Housing dataset',
        problem_type='regression',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_validation=X_validation,
        y_validation=y_validation
    )

Once the model is wrapped several metrics are pre-computed and stored in the experiment object.

.. code-block:: python

    print(experiment.metrics["test"])

    {
        "train_score":{
            "r2_score":0.606,
            "mean_squared_error":0.524,
            "mean_absolute_error":0.524,
            "median_absolute_error":0.524,
            "explained_variance_score":0.606
        },
        "test_score":{
            "r2_score":0.606,
            "mean_squared_error":0.524,
            "mean_absolute_error":0.524,
            "median_absolute_error":0.524,
            "explained_variance_score":0.606
        }
    }

This experiment can be registered with the experiment tracking system by calling the :meth:`register<mango.models.experiment_tracking.MLExperiment.register_experiment>` method.

.. code-block:: python

    experiments_folder = "/home/user/experiments"
    experiment.register_experiment(experiments_folder)


The experiment is now registered and can be viewed in the experiment tracking system.

The tracking system is used in python with :class:`MLTracker<mango.models.experiment_tracking.MLTracker>`.

.. code-block:: python

    from mango.models import MLTracker
    tracker = MLTracker(experiments_folder)
    traker.scan_for_experiments(experiment_folder)

If we now create another experiment using a RandomForestRegressor, we can register it with the tracking system and view it. Now we will show another
way of adding the experiment to the tracking system. We will use the :meth:`add_experiment<mango.models.experiment_tracking.MLTracker.add_experiment>` method.
that adds the experiment to the tracking system and also registers (saves into a subfolder) it for future use.

.. code-block:: python

    from sklearn.ensemble import RandomForestRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor())
    ])

    pipeline.fit(X_train, y_train)
    experiment = MLExperiment(
        model=pipeline,
        name='California Housing RandomForestRegressor',
        description='RandomForestRegressor on California Housing dataset',
        problem_type='regression',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    tracker.add_experiment(experiment, experiments_folder)


Once we added different experiments to the tracking system we can use the :meth:`create_compare_df<mango.models.experiment_tracking.MLTracker.create_compare_df>`
to create a dataframe that compares the different experiments and shows their metrics.

.. code-block:: python

    tracker.create_compare_df()

For more information about other methods and usages go to :class:`MLTracker<mango.models.experiment_tracking.MLTracker>`.

.. note::
        This module is still under development and some of the features described in this documentation may not be implemented yet. If you find any bug or have any suggestion, please, open an issue in the `GitHub repository <https://github.com/baobabsoluciones/mango>`_.