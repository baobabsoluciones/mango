Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~

- Initial release of mango-shap library
- SHAP explainer support for tree-based models (XGBoost, LightGBM, CatBoost, scikit-learn)
- SHAP explainer support for deep learning models (TensorFlow/Keras, PyTorch)
- SHAP explainer support for linear models
- Model-agnostic SHAP explainer using kernel method
- Summary plot visualization for feature importance
- Bar summary plot for global feature importance
- Waterfall plot visualization for individual predictions with query support
- Partial dependence plot visualization for feature interactions
- Data processing utilities for SHAP analysis
- Export utilities for CSV, JSON, and HTML formats
- Input validation utilities
- Comprehensive logging system
- Support for various data formats (pandas, numpy, polars)
- Interactive visualizations with Plotly
- Integration with Jupyter notebooks

Enhanced Features
~~~~~~~~~~~~~~~~~

- **Pipeline Support**: Full support for scikit-learn pipelines with automatic feature name extraction
- **Problem Type Support**: Comprehensive support for regression, binary classification, and multiclass classification
- **Metadata Handling**: Automatic exclusion of metadata columns from SHAP calculations
- **Query-based Analysis**: Filter and analyze specific subsets of data using pandas queries
- **Sample Filtering**: Get samples based on SHAP value thresholds with custom operators
- **Comprehensive Analysis**: Automated generation of complete SHAP analysis reports
- **Model Type Detection**: Automatic detection of model types and appropriate SHAP explainers
- **Feature Name Extraction**: Automatic extraction of feature names from various model types
- **Class-based Analysis**: Support for class-specific analysis in classification problems
- **Directory Management**: Automatic creation of output directories for organized results
