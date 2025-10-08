import os
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger
from sklearn.pipeline import Pipeline

from .explainers import TreeExplainer, DeepExplainer, LinearExplainer, KernelExplainer
from .utils import DataProcessor, ExportUtils, InputValidator


class SHAPExplainer:
    """
    SHAP explainer for machine learning model interpretability and analysis.

    This class provides a comprehensive interface for generating SHAP (SHapley Additive exPlanations)
    explanations for various types of machine learning models. It supports multiple model types
    including tree-based models, neural networks, linear models, and kernel-based models.

    The explainer can handle different problem types (regression, binary classification, multiclass
    classification) and provides automatic model type detection. It supports data preprocessing,
    metadata handling, and comprehensive analysis workflows with export capabilities.

    :param model: Trained machine learning model or sklearn Pipeline
    :type model: Any
    :param data: Background data used for SHAP calculations (numpy array or pandas DataFrame)
    :type data: Union[np.ndarray, pd.DataFrame]
    :param problem_type: Type of machine learning problem
    :type problem_type: str
    :param model_name: Name identifier for the model (used for saving results)
    :type model_name: str
    :param metadata: Column names to treat as metadata (excluded from SHAP calculations)
    :type metadata: Optional[Union[str, List[str]]]
    :param shap_folder: Directory path for saving SHAP analysis results
    :type shap_folder: Optional[str]
    :param model_type: Specific model type ('tree', 'deep', 'linear', 'kernel') - auto-detected if None
    :type model_type: Optional[str]

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X, y)
        >>> explainer = SHAPExplainer(
        ...     model=model,
        ...     data=X,
        ...     problem_type="binary_classification",
        ...     model_name="rf_classifier"
        ... )
        >>> shap_values = explainer.explain(X[:10])
        >>> print(f"SHAP values shape: {shap_values.shape}")
    """

    def __init__(
        self,
        model: Any,
        data: Union[np.ndarray, pd.DataFrame],
        problem_type: str = "regression",
        model_name: str = "model",
        metadata: Optional[Union[str, List[str]]] = None,
        shap_folder: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> None:
        """
        Initialize the SHAP explainer with model and data.

        Sets up the explainer by processing the input model and data, detecting the model type,
        initializing the appropriate SHAP explainer, and generating initial SHAP values.
        Validates inputs and prepares the explainer for analysis operations.

        :param model: Trained machine learning model or sklearn Pipeline
        :type model: Any
        :param data: Background data used for SHAP calculations
        :type data: Union[np.ndarray, pd.DataFrame]
        :param problem_type: Type of machine learning problem
        :type problem_type: str
        :param model_name: Name identifier for the model
        :type model_name: str
        :param metadata: Column names to treat as metadata
        :type metadata: Optional[Union[str, List[str]]]
        :param shap_folder: Directory path for saving results
        :type shap_folder: Optional[str]
        :param model_type: Specific model type - auto-detected if None
        :type model_type: Optional[str]
        :return: None
        :rtype: None
        """
        self.logger = get_configured_logger()

        # Set attributes
        self.problem_type = problem_type
        self.model_name = model_name
        self.shap_folder = shap_folder
        self.metadata = metadata
        self.data = data

        # Initialize validator and data processor
        self.validator = InputValidator()
        self.data_processor = DataProcessor()
        self.export_utils = ExportUtils()

        # Process model and data
        self._set_estimator(model)

        # Filter metadata BEFORE setting explainer and calculating SHAP values
        self._filter_metadata()

        self._set_explainer(model_type)

        # Get SHAP values (after data processing is complete)
        self.shap_values = self._explainer.shap_values(self._x_transformed)

        self.logger.info(
            f"SHAP explainer initialized with {self._model_type} explainer"
        )

    @property
    def problem_type(self) -> str:
        """
        Get the current problem type.

        Returns the type of machine learning problem being analyzed.
        Valid types include regression, binary classification, and multiclass classification.

        :return: Current problem type string
        :rtype: str
        """
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: str) -> None:
        """
        Validate and set the problem type.

        Sets the problem type for the SHAP explainer, validating that it's one of
        the supported types. This affects how SHAP values are computed and interpreted.

        :param problem_type: Type of machine learning problem
        :type problem_type: str
        :return: None
        :rtype: None
        :raises ValueError: If problem_type is not one of the valid options
        """
        problem_type_options = [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]
        if problem_type not in problem_type_options:
            raise ValueError(
                f"Invalid problem_type. Valid options are: {problem_type_options}"
            )
        self._problem_type = problem_type

    @property
    def model_name(self) -> str:
        """
        Get the current model name.

        Returns the identifier name for the model being analyzed.
        This name is used for saving results and organizing outputs.

        :return: Current model name string
        :rtype: str
        """
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        """
        Set the model name identifier.

        Updates the model name used for identification and result organization.
        This name will be used when saving SHAP analysis results and reports.

        :param model_name: Name identifier for the model
        :type model_name: str
        :return: None
        :rtype: None
        """
        self._model_name = model_name

    @property
    def shap_folder(self) -> Optional[str]:
        """
        Get the current SHAP folder path.

        Returns the directory path where SHAP analysis results are saved.
        Returns None if no folder has been set.

        :return: Current SHAP folder path or None
        :rtype: Optional[str]
        """
        return self._shap_folder

    @shap_folder.setter
    def shap_folder(self, shap_folder: Optional[str]) -> None:
        """
        Validate and set the SHAP folder path.

        Sets the directory path for saving SHAP analysis results. If the directory
        doesn't exist, it will be created automatically. Set to None to disable
        automatic saving.

        :param shap_folder: Directory path for saving results or None
        :type shap_folder: Optional[str]
        :return: None
        :rtype: None
        :raises OSError: If directory creation fails
        """
        if shap_folder is None:
            pass
        elif not os.path.exists(shap_folder):
            try:
                os.makedirs(shap_folder)
            except OSError:
                raise OSError(f"Creation of the directory {shap_folder} failed")
        self._shap_folder = shap_folder

    @property
    def data(self) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get the current background data.

        Returns the background dataset used for SHAP calculations.
        This data serves as the reference for computing expected values.

        :return: Current background data (numpy array or pandas DataFrame)
        :rtype: Union[np.ndarray, pd.DataFrame]
        """
        return self._data

    @data.setter
    def data(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Validate and set the background data.

        Sets the background dataset used for SHAP calculations. The data is validated
        to ensure it's in the correct format and processed for use with the explainer.
        Metadata columns are preserved for analysis but filtered from SHAP calculations.

        :param data: Background dataset for SHAP calculations
        :type data: Union[np.ndarray, pd.DataFrame]
        :return: None
        :rtype: None
        :raises ValueError: If data is not a valid numpy array or pandas DataFrame
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("data must be a pandas DataFrame or a numpy array")

        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)

        # Keep all data for the explainer (including metadata)
        # Metadata filtering will be handled in analysis methods
        self._data_with_metadata = data.copy()
        self._data = data

    @property
    def metadata(self) -> List[str]:
        """
        Get the current metadata columns.

        Returns the list of column names that are treated as metadata.
        These columns are excluded from SHAP calculations but preserved for analysis.

        :return: List of metadata column names
        :rtype: List[str]
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[Union[str, List[str]]]) -> None:
        """
        Validate and set the metadata columns.

        Sets the columns that should be treated as metadata (excluded from SHAP calculations).
        Metadata columns are preserved in the dataset but not used for feature importance
        analysis. Can accept a single column name or a list of column names.

        :param metadata: Column name(s) to treat as metadata
        :type metadata: Optional[Union[str, List[str]]]
        :return: None
        :rtype: None
        :raises ValueError: If metadata is not a string or list of strings
        """
        if metadata is None:
            metadata = []
        if not isinstance(metadata, (str, list)):
            raise ValueError("metadata must be a string or list of strings")
        if isinstance(metadata, str):
            metadata = [metadata]
        self._metadata = metadata

    def _get_class_index(self, class_ind_name: Union[str, int]) -> int:
        """
        Get the index of a class based on its name or index.

        Converts a class identifier (either name or index) to the corresponding
        integer index used internally by the model. This is useful for handling
        both string class names and numeric indices in classification problems.

        :param class_ind_name: Class name (string) or class index (integer)
        :type class_ind_name: Union[str, int]
        :return: Integer index of the class
        :rtype: int
        :raises ValueError: If class name is not found or index is out of range
        """
        if isinstance(class_ind_name, str):
            try:
                class_index = list(self._model.classes_).index(class_ind_name)
            except ValueError:
                raise ValueError(f"class_name must be one of {self._model.classes_}")
        elif isinstance(class_ind_name, int):
            if len(self._model.classes_) <= class_ind_name:
                raise ValueError(
                    f"class_index must be less than {len(self._model.classes_)}"
                )
            class_index = class_ind_name
        else:
            raise ValueError("class_name must be a string or an integer")
        return class_index

    def _set_estimator(self, estimator: Any) -> None:
        """
        Extract the model from a pipeline and configure feature names.

        Processes the input estimator to extract the actual model (handling sklearn
        pipelines) and sets up feature names. For pipelines, it extracts the final
        estimator and applies any preprocessing steps. Also handles metadata filtering
        and validates that the data is not empty.

        :param estimator: Machine learning model or sklearn Pipeline
        :type estimator: Any
        :return: None
        :rtype: None
        :raises ValueError: If model is None or data is empty
        """
        if estimator is None:
            raise ValueError("Model cannot be None")

        # Validate that data is not empty
        if hasattr(self._data, "shape") and self._data.shape[0] == 0:
            raise ValueError("Data cannot be empty")
        elif hasattr(self._data, "__len__") and len(self._data) == 0:
            raise ValueError("Data cannot be empty")

        if isinstance(estimator, Pipeline):
            self._model = estimator.steps[-1][1]
            if len(estimator.steps) > 1:
                transformer = Pipeline(estimator.steps[0:-1])
                self._x_transformed = transformer.transform(self._data)
                self._feature_names = transformer.get_feature_names_out()
            else:
                self._x_transformed = self._data
                if isinstance(self._data, np.ndarray):
                    self._feature_names = [
                        f"Feature {i}" for i in range(self._data.shape[1])
                    ]
                else:
                    # Use DataFrame column names if available, otherwise get from model
                    if isinstance(self._data, pd.DataFrame):
                        self._feature_names = list(self._data.columns)
                    else:
                        self._feature_names = self._get_feature_names(self._model)
        else:
            self._model = estimator
            self._x_transformed = self._data
            if isinstance(self._data, np.ndarray):
                self._feature_names = [
                    f"Feature {i}" for i in range(self._data.shape[1])
                ]
            else:
                # Use DataFrame column names if available, otherwise get from model
                if isinstance(self._data, pd.DataFrame):
                    self._feature_names = list(self._data.columns)
                else:
                    self._feature_names = self._get_feature_names(self._model)

    def _filter_metadata(self):
        """Filter out metadata columns from feature names and data if they exist."""
        if self._metadata and isinstance(self._data, pd.DataFrame):
            # Get all column names
            all_columns = list(self._data.columns)
            # Filter out metadata columns from feature names
            self._feature_names = [
                col for col in all_columns if col not in self._metadata
            ]
            # Filter out metadata columns from transformed data
            self._x_transformed = self._x_transformed.drop(columns=self._metadata)

    def _clean_kwargs_for_plot(self, kwargs):
        """Remove 'show' parameter from kwargs to avoid duplicate parameter error."""
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop("show", None)
        return plot_kwargs

    @staticmethod
    def _get_feature_names(estimator: Any) -> List[str]:
        """
        Extract feature names from a machine learning estimator.

        Attempts to retrieve feature names from various model types including
        scikit-learn models, LightGBM, XGBoost, and other frameworks. Falls back
        to generating generic feature names if none are available.

        :param estimator: The machine learning model to extract feature names from
        :type estimator: Any
        :return: List of feature names
        :rtype: List[str]

        Example:
            >>> feature_names = SHAPExplainer._get_feature_names(trained_model)
            >>> print(f"Feature names: {feature_names}")
        """
        try:
            # sklearn >= 1.0
            feature_names = estimator.feature_names_in_
        except AttributeError:
            try:
                # LightGBM
                feature_names = estimator.feature_name_
            except AttributeError:
                try:
                    # XGBoost
                    feature_names = estimator.feature_names
                except AttributeError:
                    # Fallback: generate generic feature names
                    if hasattr(estimator, "n_features_in_"):
                        feature_names = [
                            f"feature_{i}" for i in range(estimator.n_features_in_)
                        ]
                    elif hasattr(estimator, "n_features_"):
                        feature_names = [
                            f"feature_{i}" for i in range(estimator.n_features_)
                        ]
                    else:
                        # Last resort: try to infer from the model
                        # Default fallback
                        feature_names = [f"feature_{i}" for i in range(5)]
        return feature_names

    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """
        Automatically detect the type of machine learning model.

        Analyzes the model class name to determine the appropriate SHAP explainer type.
        Supports detection of tree-based models, neural networks, linear models, and
        falls back to kernel explainer for unknown model types.

        :param model: The machine learning model to analyze
        :type model: Any
        :return: Model type string ('tree', 'deep', 'linear', or 'kernel')
        :rtype: str

        Example:
            >>> model_type = SHAPExplainer._detect_model_type(random_forest_model)
            >>> print(f"Detected model type: {model_type}")  # 'tree'
        """
        model_class = model.__class__.__name__.lower()

        if any(
            tree_type in model_class
            for tree_type in ["tree", "forest", "boosting", "xgb", "lgb", "catboost"]
        ):
            return "tree"
        elif any(
            deep_type in model_class
            for deep_type in ["neural", "deep", "keras", "tensorflow", "pytorch"]
        ):
            return "deep"
        elif any(
            linear_type in model_class
            for linear_type in ["linear", "logistic", "ridge", "lasso"]
        ):
            return "linear"
        else:
            return "kernel"

    def _set_explainer(self, model_type: Optional[str] = None) -> None:
        """
        Set the appropriate SHAP explainer based on the model type.

        Initializes the correct SHAP explainer (TreeExplainer, DeepExplainer, LinearExplainer,
        or KernelExplainer) based on the detected or specified model type. This method
        automatically detects the model type if not provided and creates the appropriate
        explainer instance.

        :param model_type: Specific model type ('tree', 'deep', 'linear', 'kernel') or None for auto-detection
        :type model_type: Optional[str]
        :return: None
        :rtype: None
        """
        if model_type is None:
            model_type = self._detect_model_type(self._model)

        self._model_type = model_type

        if model_type == "tree":
            self._explainer = TreeExplainer(self._model, self._x_transformed)
        elif model_type == "deep":
            self._explainer = DeepExplainer(self._model, self._x_transformed)
        elif model_type == "linear":
            self._explainer = LinearExplainer(self._model, self._x_transformed)
        else:
            self._explainer = KernelExplainer(self._model, self._x_transformed)

    @staticmethod
    def _save_fig(title: str, file_path_save: str) -> None:
        """
        Save the current matplotlib figure with a title.

        Saves the currently active matplotlib figure to the specified file path
        with the given title. The figure is automatically closed after saving
        to free up memory resources.

        :param title: Title to display on the saved figure
        :type title: str
        :param file_path_save: File path where the figure should be saved
        :type file_path_save: str
        :return: None
        :rtype: None
        """
        import matplotlib.pyplot as plt

        fig1 = plt.gcf()
        fig1.suptitle(title)
        fig1.tight_layout()
        fig1.savefig(file_path_save)
        plt.close()

    def explain(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate SHAP values for the given input data.

        Processes the input data and generates SHAP explanations using the initialized
        explainer. The method handles data validation, preprocessing, and returns
        SHAP values that explain the model's predictions for each input sample.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Maximum number of evaluations for kernel explainer (optional)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> shap_values = explainer.explain(test_data)
            >>> print(f"SHAP values shape: {shap_values.shape}")
            >>> # For binary classification: (n_samples, n_features, 2)
            >>> # For regression: (n_samples, n_features)
        """
        self.logger.info("Generating SHAP values")

        # Validate and process data
        self.validator.validate_data(data)
        processed_data = self.data_processor.process_data(data)

        # Generate SHAP values
        if hasattr(self._explainer, "shap_values"):
            shap_values = self._explainer.shap_values(
                processed_data, max_evals=max_evals
            )
        else:
            shap_values = self._explainer(processed_data)

        self.logger.info(f"Generated SHAP values with shape: {shap_values.shape}")
        return shap_values

    def summary_plot(
        self,
        class_name: Union[str, int] = 1,
        file_path_save: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Create a summary plot of SHAP values.

        Generates a summary plot showing the distribution of SHAP values for each feature.
        For classification problems, plots values for the specified class. The plot
        displays feature importance and the impact direction of each feature.

        :param class_name: Class name or index to plot for classification problems
        :type class_name: Union[str, int]
        :param file_path_save: File path to save the plot (optional)
        :type file_path_save: Optional[str]
        :param kwargs: Additional arguments passed to shap.summary_plot
        :return: None
        :rtype: None
        :raises ValueError: If the save path directory does not exist
        """
        if file_path_save is not None:
            if not os.path.exists(os.path.dirname(file_path_save)):
                raise ValueError(
                    f"Path {os.path.dirname(file_path_save)} does not exist"
                )

        # Generate SHAP values for the background data used in the explainer
        shap_values_for_plot = self._explainer.shap_values(self._x_transformed)

        # Handle different SHAP values shapes
        if self._problem_type != "regression":
            # For classification, handle different shapes
            if len(shap_values_for_plot.shape) == 3:
                if shap_values_for_plot.shape[0] == len(self._model.classes_):
                    # Shape: (n_classes, n_samples, n_features)
                    selected_shap_values = shap_values_for_plot[
                        self._get_class_index(class_name)
                    ]
                else:
                    # Shape: (n_samples, n_features, n_classes)
                    selected_shap_values = shap_values_for_plot[
                        :, :, self._get_class_index(class_name)
                    ]
            else:
                # Fallback for unexpected shapes
                selected_shap_values = shap_values_for_plot
        else:
            # For regression, use values directly
            selected_shap_values = shap_values_for_plot

        shap.summary_plot(
            selected_shap_values,
            self._x_transformed,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            **self._clean_kwargs_for_plot(kwargs),
        )

        if file_path_save is not None:
            self._save_fig(
                title=(
                    f"Summary Plot class {self._model.classes_[self._get_class_index(class_name)]}"
                    if self._problem_type != "regression"
                    else "Summary Plot"
                ),
                file_path_save=(
                    f"{file_path_save}.png"
                    if not file_path_save.endswith(".png")
                    else file_path_save
                ),
            )

    def bar_summary_plot(self, file_path_save: Optional[str] = None, **kwargs) -> None:
        """
        Create a bar summary plot of SHAP values.

        Generates a bar plot showing the mean absolute SHAP values for each feature,
        providing a clear view of overall feature importance. This plot type is
        particularly useful for comparing feature importance across the dataset.

        :param file_path_save: File path to save the plot (optional)
        :type file_path_save: Optional[str]
        :param kwargs: Additional arguments passed to shap.summary_plot
        :return: None
        :rtype: None
        :raises ValueError: If the save path directory does not exist
        """
        if file_path_save is not None:
            if not os.path.exists(os.path.dirname(file_path_save)):
                raise ValueError(
                    f"Path: {os.path.dirname(file_path_save)} does not exist"
                )

        # Generate SHAP values for the background data used in the explainer
        shap_values_for_plot = self._explainer.shap_values(self._x_transformed)

        # For bar plot, we need to handle classification vs regression differently
        if self._problem_type != "regression":
            # For classification, handle different shapes
            if len(shap_values_for_plot.shape) == 3:
                if shap_values_for_plot.shape[0] == len(self._model.classes_):
                    # Shape: (n_classes, n_samples, n_features) - use first class
                    shap_values_for_bar = shap_values_for_plot[0]
                else:
                    # Shape: (n_samples, n_features, n_classes) - use first class
                    shap_values_for_bar = shap_values_for_plot[:, :, 0]
            else:
                # Fallback for unexpected shapes
                shap_values_for_bar = shap_values_for_plot
        else:
            # For regression, use the values directly
            shap_values_for_bar = shap_values_for_plot

        shap.summary_plot(
            shap_values_for_bar,
            plot_type="bar",
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            **self._clean_kwargs_for_plot(kwargs),
        )

        if file_path_save is not None:
            self._save_fig(
                title="Bar Summary Plot",
                file_path_save=(
                    f"{file_path_save}.png"
                    if not file_path_save.endswith(".png")
                    else file_path_save
                ),
            )

    def waterfall_plot(
        self, instance_idx: int = 0, path_save: Optional[str] = None, **kwargs
    ) -> None:
        """
        Create waterfall plot for a specific instance.

        Generates a waterfall plot for the specified instance index.
        For classification problems, creates plots for each class. Waterfall
        plots show how each feature contributes to the final prediction, starting from
        the expected value and adding/subtracting feature contributions.

        :param instance_idx: Index of the instance to plot
        :type instance_idx: int
        :param path_save: Directory path to save the plots (optional)
        :type path_save: Optional[str]
        :param kwargs: Additional arguments passed to shap.waterfall_plot
        :return: None
        :rtype: None
        :raises ValueError: If path_save is not a directory or instance_idx is out of bounds
        """
        if path_save is not None and not os.path.isdir(path_save):
            raise ValueError("path_save must be a directory")

        # Validate instance_idx
        if instance_idx < 0 or instance_idx >= len(self._data):
            raise ValueError(
                f"instance_idx {instance_idx} is out of bounds. Data has {len(self._data)} instances."
            )

        # Generate SHAP values for the background data used in the explainer
        shap_values_for_plot = self._explainer.shap_values(self._x_transformed)

        if self._problem_type in [
            "binary_classification",
            "multiclass_classification",
        ]:
            for j, class_name in enumerate(self._model.classes_):
                # Get expected value for this class
                if hasattr(self._explainer, "expected_value"):
                    if isinstance(self._explainer.expected_value, (list, np.ndarray)):
                        expected_val = self._explainer.expected_value[j]
                    else:
                        expected_val = self._explainer.expected_value
                else:
                    # Fallback: use mean of predictions
                    expected_val = 0.0

                # Handle different SHAP values shapes
                if len(shap_values_for_plot.shape) == 3:
                    if shap_values_for_plot.shape[0] == len(self._model.classes_):
                        # Shape: (n_classes, n_samples, n_features)
                        values = shap_values_for_plot[j][instance_idx]
                    else:
                        # Shape: (n_samples, n_features, n_classes)
                        values = shap_values_for_plot[instance_idx][:, j]
                else:
                    # Fallback for unexpected shapes
                    values = shap_values_for_plot[instance_idx]

                shap.waterfall_plot(
                    shap.Explanation(
                        values=values,
                        base_values=expected_val,
                        data=self._data.iloc[instance_idx],
                        feature_names=self._feature_names,
                    ),
                    show=kwargs.get("show", False),
                    **self._clean_kwargs_for_plot(kwargs),
                )

                if path_save is not None:
                    self._save_fig(
                        title=f"Waterfall plot instance {instance_idx} class {class_name}",
                        file_path_save=os.path.join(
                            path_save,
                            f"waterfall_plot_instance_{instance_idx}_class_{class_name}.png",
                        ),
                    )
        else:
            # Get expected value for regression
            if hasattr(self._explainer, "expected_value"):
                expected_val = self._explainer.expected_value
            else:
                # Fallback: use mean of predictions
                expected_val = 0.0

            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_for_plot[instance_idx],
                    base_values=expected_val,
                    data=self._data.iloc[instance_idx],
                    feature_names=self._feature_names,
                ),
                show=kwargs.get("show", False),
                **self._clean_kwargs_for_plot(kwargs),
            )

            if path_save is not None:
                self._save_fig(
                    title=f"Waterfall plot instance {instance_idx}",
                    file_path_save=os.path.join(
                        path_save,
                        f"waterfall_plot_instance_{instance_idx}.png",
                    ),
                )

    def partial_dependence_plot(
        self,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = None,
        class_name: Optional[Union[str, int]] = None,
        file_path_save: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Create a partial dependence plot.

        Generates a partial dependence plot showing how the SHAP values of a feature
        change with respect to the feature's values. Can optionally show interactions
        with another feature. For classification problems, plots values for the
        specified class.

        :param feature: Feature name or index to plot
        :type feature: Union[str, int]
        :param interaction_feature: Optional feature to show interaction with
        :type interaction_feature: Optional[Union[str, int]]
        :param class_name: Class name or index for classification problems
        :type class_name: Optional[Union[str, int]]
        :param file_path_save: File path to save the plot (optional)
        :type file_path_save: Optional[str]
        :param kwargs: Additional arguments passed to shap.dependence_plot
        :return: None
        :rtype: None
        """
        # Generate SHAP values for the background data used in the explainer
        shap_values_for_plot = self._explainer.shap_values(self._x_transformed)

        # Set default class_name for classification problems
        if self._problem_type != "regression" and class_name is None:
            # Default to class 1 for binary classification
            class_name = 1

        # Handle different SHAP values shapes
        if self._problem_type != "regression":
            # For classification, handle different shapes
            if len(shap_values_for_plot.shape) == 3:
                if shap_values_for_plot.shape[0] == len(self._model.classes_):
                    # Shape: (n_classes, n_samples, n_features)
                    selected_shap_values = shap_values_for_plot[
                        self._get_class_index(class_name)
                    ]
                else:
                    # Shape: (n_samples, n_features, n_classes)
                    selected_shap_values = shap_values_for_plot[
                        :, :, self._get_class_index(class_name)
                    ]
            else:
                # Fallback for unexpected shapes
                selected_shap_values = shap_values_for_plot
        else:
            # For regression, use values directly
            selected_shap_values = shap_values_for_plot

        shap.dependence_plot(
            feature,
            selected_shap_values,
            self._x_transformed,
            interaction_index=interaction_feature,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            **self._clean_kwargs_for_plot(kwargs),
        )
        if file_path_save is not None:
            self._save_fig(
                title=(
                    f"Partial dependence plot: {feature} and {interaction_feature}"
                    if interaction_feature is not None
                    else f"Partial dependence plot: {feature}"
                ),
                file_path_save=file_path_save,
            )

    def get_sample_by_shap_value(
        self,
        feature_idx: int,
        highest: bool = True,
        class_name: Optional[Union[str, int]] = None,
    ) -> dict:
        """
        Get the sample with the highest or lowest SHAP value for a specific feature.

        Returns information about the instance with the most extreme (highest or lowest)
        SHAP value for the specified feature. Useful for understanding which data points
        have the most significant feature contributions.

        :param feature_idx: Index of the feature to analyze
        :type feature_idx: int
        :param highest: If True, return sample with highest SHAP value; if False, lowest
        :type highest: bool
        :param class_name: Class name or index for classification problems (optional)
        :type class_name: Optional[Union[str, int]]
        :return: Dictionary containing instance_idx, shap_value, and feature_value
        :rtype: dict

        Example:
            >>> # Get sample with highest SHAP value for first feature
            >>> result = explainer.get_sample_by_shap_value(
            ...     feature_idx=0,
            ...     highest=True
            ... )
            >>> print(f"Instance {result['instance_idx']} has SHAP value {result['shap_value']}")
        """
        if feature_idx >= len(self._feature_names):
            raise ValueError(
                f"Feature index {feature_idx} out of range. Max index: {len(self._feature_names) - 1}"
            )

        # Handle 3D SHAP values for classification problems
        shap_values_for_analysis = self.shap_values

        if (
            self._problem_type != "regression"
            and len(shap_values_for_analysis.shape) == 3
        ):
            # For classification, select the appropriate class
            if class_name is None:
                # Default to first class for binary classification
                if self._problem_type == "binary_classification":
                    class_name = 1
                else:
                    class_name = 0

            class_index = self._get_class_index(class_name)

            # Handle different SHAP values shapes
            if shap_values_for_analysis.shape[0] == len(self._model.classes_):
                # Shape: (n_classes, n_samples, n_features)
                shap_values_for_analysis = shap_values_for_analysis[class_index]
            else:
                # Shape: (n_samples, n_features, n_classes)
                shap_values_for_analysis = shap_values_for_analysis[:, :, class_index]

        # Get SHAP values for the specified feature
        feature_shap_values = shap_values_for_analysis[:, feature_idx]

        # Find the extreme value
        if highest:
            instance_idx = np.argmax(feature_shap_values)
        else:
            instance_idx = np.argmin(feature_shap_values)

        shap_value = feature_shap_values[instance_idx]
        feature_value = (
            self._x_transformed.iloc[instance_idx, feature_idx]
            if hasattr(self._x_transformed, "iloc")
            else self._x_transformed[instance_idx, feature_idx]
        )

        return {
            "instance_idx": int(instance_idx),
            "shap_value": float(shap_value),
            "feature_value": float(feature_value),
        }

    def make_shap_analysis(
        self,
        queries: Optional[List[str]] = None,
        pdp_tuples: Optional[List[tuple]] = None,
    ) -> None:
        """
        Perform comprehensive SHAP analysis with automated visualization generation.

        Executes a complete SHAP analysis workflow, generating summary plots, bar plots,
        waterfall plots (if queries provided), and partial dependence plots (if tuples provided).
        Creates organized directory structure and saves all visualizations automatically.
        Handles both classification and regression problems with appropriate plot variations.

        :param queries: List of pandas query strings for generating waterfall plots
        :type queries: Optional[List[str]]
        :param pdp_tuples: List of tuples (feature, interaction_feature) for partial dependence plots
        :type pdp_tuples: Optional[List[tuple]]
        :return: None
        :rtype: None
        :raises ValueError: If shap_folder is not set (None)
        """
        # Check path to save plots
        if self._shap_folder is None:
            raise ValueError(
                "Set path to save plots: the attribute shap_folder is None"
            )
        base_path = os.path.join(
            self._shap_folder,
            "shap_analysis",
            self.model_name,
        )

        list_paths = [
            os.path.join(base_path, "summary/"),
            os.path.join(base_path, "individual/"),
            os.path.join(base_path, "partial_dependence/"),
        ]
        # Make dirs to save plots
        _ = [os.makedirs(os.path.dirname(path), exist_ok=True) for path in list_paths]

        # Make summary plot
        if self._problem_type in ["binary_classification", "multiclass_classification"]:
            _ = [
                self.summary_plot(
                    class_name=self._get_class_index(class_index),
                    file_path_save=os.path.join(
                        base_path,
                        "summary",
                        f"summary_class_{class_index}.png",
                    ),
                )
                for class_index in range(len(self._model.classes_))
            ]
            self.bar_summary_plot(
                file_path_save=os.path.join(
                    base_path,
                    "summary",
                    "barplot.png",
                )
            )
            if queries is not None:
                for i, query in enumerate(queries):
                    self.waterfall_plot(
                        query=query,
                        path_save=os.path.join(
                            base_path,
                            "individual",
                        ),
                    )
            if pdp_tuples:
                for pdp_tuple in pdp_tuples:
                    _ = [
                        self.partial_dependence_plot(
                            feature=pdp_tuple[0],
                            interaction_feature=pdp_tuple[1],
                            class_name=self._get_class_index(class_index),
                            file_path_save=os.path.join(
                                base_path,
                                "partial_dependence",
                                f"pdp_class_{class_index}_{pdp_tuple[0]}{'_' + pdp_tuple[1] if pdp_tuple[1] is not None else ''}.png",
                            ),
                        )
                        for class_index in range(len(self._model.classes_))
                    ]

        elif self._problem_type == "regression":
            self.summary_plot(
                file_path_save=os.path.join(
                    base_path,
                    "summary",
                    "summary.png",
                )
            )
            self.bar_summary_plot(
                file_path_save=os.path.join(
                    base_path,
                    "summary",
                    "barplot.png",
                )
            )
            if queries is not None:
                for i, query in enumerate(queries):
                    self.waterfall_plot(
                        query=query,
                        path_save=os.path.join(
                            base_path,
                            "individual",
                        ),
                    )
            if pdp_tuples:
                for pdp_tuple in pdp_tuples:
                    self.partial_dependence_plot(
                        feature=pdp_tuple[0],
                        interaction_feature=pdp_tuple[1],
                        file_path_save=os.path.join(
                            base_path,
                            "partial_dependence",
                            f"pdp_{pdp_tuple[0]}{'_' + pdp_tuple[1] if pdp_tuple[1] is not None else ''}.png",
                        ),
                    )

    def export_explanations(
        self,
        output_path: str,
        format: str = "csv",
        class_name: Optional[Union[str, int]] = None,
    ) -> None:
        """
        Export SHAP explanations to file in various formats.

        Saves SHAP values, data, and feature names to a file in the specified format.
        Supports multiple export formats including CSV, JSON, and HTML for different
        use cases and downstream analysis requirements.

        :param output_path: File path where the explanations should be saved
        :type output_path: str
        :param format: Export format ('csv', 'json', 'html')
        :type format: str
        :param class_name: Class name or index for classification problems (default: first class)
        :type class_name: Optional[Union[str, int]]
        :return: None
        :rtype: None
        """
        # Handle 3D SHAP values for classification problems
        shap_values_for_export = self.shap_values

        if (
            self._problem_type != "regression"
            and len(shap_values_for_export.shape) == 3
        ):
            # For classification, select the appropriate class
            if class_name is None:
                # Default to first class for binary classification
                if self._problem_type == "binary_classification":
                    class_name = 1
                else:
                    class_name = 0

            class_index = self._get_class_index(class_name)

            # Handle different SHAP values shapes
            if shap_values_for_export.shape[0] == len(self._model.classes_):
                # Shape: (n_classes, n_samples, n_features)
                shap_values_for_export = shap_values_for_export[class_index]
            else:
                # Shape: (n_samples, n_features, n_classes)
                shap_values_for_export = shap_values_for_export[:, :, class_index]

        self.export_utils.export(
            shap_values=shap_values_for_export,
            data=self._x_transformed,
            feature_names=self._feature_names,
            output_path=output_path,
            format=format,
        )
