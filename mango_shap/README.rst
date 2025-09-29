Mango SHAP
==========

A Python library for model interpretability using SHAP (SHapley Additive exPlanations) values.

Description
-----------

Mango SHAP is a specialized tool for machine learning model interpretability that provides comprehensive SHAP analysis capabilities. It is designed to be highly configurable and easy to use, with advanced visualization and explanation features for various types of models.

Key Features
------------

**Model Support**
- Tree-based models (XGBoost, LightGBM, CatBoost, scikit-learn)
- Neural networks and deep learning models
- Linear models and generalized linear models
- Custom model support through SHAP's model-agnostic explainers

**Explanation Methods**
- TreeExplainer for tree-based models
- DeepExplainer for neural networks
- LinearExplainer for linear models
- KernelExplainer for model-agnostic explanations
- GradientExplainer for gradient-based models

**Advanced Visualization**
- Summary plots for feature importance
- Waterfall plots for individual predictions
- Force plots for detailed explanations
- Bar plots for global feature importance
- Heatmaps for time series and multi-output models
- Dependence plots for feature interactions

**Data Processing**
- Automatic feature name handling
- Data preprocessing and validation
- Support for various data formats (pandas, numpy, polars)
- Missing value handling

**Export and Integration**
- Export explanations to various formats (CSV, JSON, HTML)
- Integration with Jupyter notebooks
- Interactive visualizations with Plotly
- Batch processing capabilities

Installation
------------

**Using uv:**

.. code-block:: bash

   # Create virtual environment with Python 3.10+
   uv venv --python 3.10
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   uv add mango-shap

**Using pip:**

.. code-block:: bash

   pip install mango-shap

Dependencies
------------

- Python >= 3.10
- SHAP >= 0.45.0
- NumPy >= 1.24.4
- Pandas >= 2.0.3
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Plotly >= 5.15.0
- Scikit-learn >= 1.3.2
- XGBoost >= 1.7.0
- LightGBM >= 4.0.0
- CatBoost >= 1.2.0

Basic Usage
-----------

.. code-block:: python

   from mango_shap import SHAPExplainer
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier

   # Load your data
   X_train = pd.read_csv('train_features.csv')
   y_train = pd.read_csv('train_target.csv')
   X_test = pd.read_csv('test_features.csv')

   # Train a model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   # Create SHAP explainer
   explainer = SHAPExplainer(
       model=model,
       data=X_train,
       problem_type="binary_classification",
       model_name="my_model",
       shap_folder="./shap_results"
   )

   # Generate visualizations
   explainer.summary_plot(class_name=1, file_path_save="summary_plot.png")
   explainer.bar_summary_plot(file_path_save="bar_plot.png")
   explainer.waterfall_plot(query="age > 30", path_save="./waterfall_plots/")
   explainer.partial_dependence_plot(
       feature="age", 
       interaction_feature="income",
       file_path_save="pdp_plot.png"
   )

   # Get samples by SHAP value
   high_impact_samples = explainer.get_sample_by_shap_value(
       shap_value=0.5,
       feature_name="age",
       operator=">="
   )

   # Perform comprehensive analysis
   explainer.make_shap_analysis(
       queries=["age > 30", "income < 50000"],
       pdp_tuples=[("age", "income"), ("education", None)]
   )

Advanced Usage: Pipeline Support
---------------------------------

The library supports scikit-learn pipelines with automatic feature name extraction:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier

   # Create a pipeline
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('classifier', RandomForestClassifier(n_estimators=100))
   ])
   
   # Train the pipeline
   pipeline.fit(X_train, y_train)

   # Create SHAP explainer with pipeline
   explainer = SHAPExplainer(
       model=pipeline,
       data=X_train,
       problem_type="binary_classification",
       model_name="pipeline_model"
   )

   # The explainer automatically handles feature transformation
   # and extracts feature names from the pipeline

Advanced Usage: Metadata Handling
---------------------------------

Handle metadata columns that shouldn't be used for SHAP calculations:

.. code-block:: python

   # Data with metadata columns
   data_with_metadata = pd.DataFrame({
       'feature1': [1, 2, 3, 4, 5],
       'feature2': [10, 20, 30, 40, 50],
       'id': ['A', 'B', 'C', 'D', 'E'],  # metadata
       'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']  # metadata
   })

   explainer = SHAPExplainer(
       model=model,
       data=data_with_metadata,
       metadata=['id', 'timestamp'],  # These columns will be excluded from SHAP calculations
       problem_type="regression"
   )

Advanced Usage: Query-based Analysis
------------------------------------

Perform analysis on specific subsets of your data:

.. code-block:: python

   # Analyze specific groups
   explainer.waterfall_plot(
       query="age > 65 and income > 100000",
       path_save="./senior_high_income_analysis/"
   )

   # Get samples with high SHAP values for a specific feature
   high_impact_samples = explainer.get_sample_by_shap_value(
       shap_value=0.3,
       feature_name="age",
       operator=">="
   )

   print(f"Found {len(high_impact_samples)} samples with high age impact")

Project Structure
-----------------

::

   mango_shap/
   ├── mango_shap/
   │   ├── __init__.py              # Main package initialization
   │   ├── explainer.py             # Main SHAP explainer class
   │   ├── explainers/
   │   │   ├── __init__.py          # Explainers package
   │   │   ├── tree_explainer.py    # Tree-based model explainer
   │   │   ├── deep_explainer.py    # Neural network explainer
   │   │   ├── linear_explainer.py  # Linear model explainer
   │   │   └── kernel_explainer.py  # Model-agnostic explainer
   │   ├── visualizers/
   │   │   ├── __init__.py          # Visualizers package
   │   │   ├── summary_plot.py      # Summary plot generator
   │   │   ├── waterfall_plot.py    # Waterfall plot generator
   │   │   ├── force_plot.py        # Force plot generator
   │   │   └── dependence_plot.py   # Dependence plot generator
   │   ├── utils/
   │   │   ├── __init__.py          # Utils package
   │   │   ├── data_processing.py   # Data processing utilities
   │   │   ├── export_utils.py      # Export functionality
   │   │   └── validation.py        # Input validation
   │   ├── tests/                   # Unit tests
   │   │   └── test_explainer.py    # Main explainer tests
   │   └── logging/                 # Logging utilities
   │       └── logger.py            # Logging configuration
   ├── pyproject.toml              # Project configuration
   └── uv.lock                     # Dependency lock file

Documentation
-------------

For detailed documentation, visit the `Mango Documentation <https://baobabsoluciones.github.io/mango/>`_.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <https://baobabsoluciones.es/>`_
