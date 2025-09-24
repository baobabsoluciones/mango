import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mango.logging import get_configured_logger
from mango.shap_analysis.const import TREE_EXPLAINER_MODELS, KERNEL_EXPLAINER_MODELS
from sklearn.pipeline import Pipeline

log = get_configured_logger(__name__)


class ShapAnalyzer:
    """
    Comprehensive SHAP (SHapley Additive exPlanations) analysis tool for machine learning models.

    This class provides a unified interface for generating various SHAP visualizations
    and analyses for machine learning models. It supports regression, binary classification,
    and multiclass classification problems with both tree-based and kernel explainers.

    The analyzer can generate summary plots, bar plots, waterfall plots, and partial
    dependence plots, as well as filter data based on SHAP values.

    :param problem_type: Type of machine learning problem ("regression", "binary_classification", "multiclass_classification")
    :type problem_type: str
    :param model_name: Name identifier for the model
    :type model_name: str
    :param estimator: Trained machine learning model or sklearn Pipeline
    :type estimator: object
    :param data: Training or test data for SHAP analysis
    :type data: Union[pd.DataFrame, np.ndarray]
    :param metadata: Column names to exclude from analysis (metadata columns)
    :type metadata: Union[str, List[str]], optional
    :param shap_folder: Directory path to save SHAP plots and analysis
    :type shap_folder: str, optional

    Example:
        >>> from mango.shap_analysis import ShapAnalyzer
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> import pandas as pd
        >>>
        >>> # Create sample data and model
        >>> X = pd.DataFrame({'age': [25, 30, 35], 'income': [50000, 60000, 70000]})
        >>> y = [100, 120, 140]
        >>> model = RandomForestRegressor().fit(X, y)
        >>>
        >>> # Initialize analyzer
        >>> analyzer = ShapAnalyzer(
        ...     problem_type="regression",
        ...     model_name="income_predictor",
        ...     estimator=model,
        ...     data=X,
        ...     shap_folder="./shap_outputs"
        ... )
        >>>
        >>> # Generate plots
        >>> analyzer.summary_plot(show=True)
        >>> analyzer.bar_summary_plot(show=True)
        >>> analyzer.make_shap_analysis()
    """

    def __init__(
        self,
        *,
        problem_type: str,
        model_name: str,
        estimator: object,
        data: Union[pd.DataFrame, np.ndarray],
        metadata: Union[str, List[str]] = None,
        shap_folder: str = None,
    ):
        """
        Initialize the SHAP analyzer with model and data.

        Sets up the analyzer by validating inputs, extracting the model from
        potential pipelines, configuring the appropriate SHAP explainer, and
        computing SHAP values for the provided data.

        :param problem_type: Type of machine learning problem
        :type problem_type: str
        :param model_name: Name identifier for the model
        :type model_name: str
        :param estimator: Trained machine learning model or sklearn Pipeline
        :type estimator: object
        :param data: Training or test data for SHAP analysis
        :type data: Union[pd.DataFrame, np.ndarray]
        :param metadata: Column names to exclude from analysis
        :type metadata: Union[str, List[str]], optional
        :param shap_folder: Directory path to save SHAP plots
        :type shap_folder: str, optional
        :raises ValueError: If problem_type is invalid or model is not supported
        :raises OSError: If shap_folder cannot be created
        """
        # Set attributes
        self.problem_type = problem_type
        self._model_name = model_name
        self.shap_folder = shap_folder
        self.metadata = metadata
        self.data = data

        # Assign model
        self._set_estimator(estimator)

        # Assign shap explainer
        self._set_explainer()

        # Get shap values
        log.info(f"Computing SHAP values for {self._model_name}")
        self.shap_values = self._explainer.shap_values(self._x_transformed)
        log.info(f"SHAP analysis initialized successfully for {self._model_name}")

    @property
    def problem_type(self):
        """
        Get the problem type of the model.

        Returns the type of machine learning problem this analyzer is configured for.

        :return: Problem type ("regression", "binary_classification", or "multiclass_classification")
        :rtype: str
        """
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: str):
        """
        Validate and set the problem type.

        Validates that the problem type is one of the supported options and
        sets the internal attribute.

        :param problem_type: Type of machine learning problem
        :type problem_type: str
        :raises ValueError: If problem_type is not supported
        """
        problem_type_options = [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]
        if problem_type not in problem_type_options:
            log.error(f"Invalid problem_type: {problem_type}")
            raise ValueError(
                f"Invalid problem_type. Valid options are: {problem_type_options}"
            )
        self._problem_type = problem_type

    @property
    def model_name(self):
        """
        Get the model name.

        Returns the name identifier for the model being analyzed.

        :return: Model name
        :rtype: str
        """
        return self._model_name

    @property
    def shap_folder(self):
        """
        Get the SHAP folder path.

        Returns the directory path where SHAP plots and analysis results are saved.

        :return: SHAP folder path
        :rtype: str
        """
        return self._shap_folder

    @shap_folder.setter
    def shap_folder(self, shap_folder: str):
        """
        Validate and set the SHAP folder path.

        Validates the folder path and creates the directory if it doesn't exist.

        :param shap_folder: Directory path for saving SHAP outputs
        :type shap_folder: str
        :raises OSError: If the directory cannot be created
        """
        if shap_folder is None:
            pass
        elif not os.path.exists(shap_folder):
            try:
                os.makedirs(shap_folder)
                log.info(f"Created SHAP folder: {shap_folder}")
            except OSError as e:
                log.error(f"Failed to create directory {shap_folder}: {e}")
                raise OSError(f"Creation of the directory {shap_folder} failed")

        self._shap_folder = shap_folder

    @property
    def data(self):
        """
        Get the processed data used for SHAP analysis.

        Returns the data with metadata columns removed, ready for SHAP analysis.

        :return: Processed data for SHAP analysis
        :rtype: Union[pd.DataFrame, np.ndarray]
        """
        return self._data

    @data.setter
    def data(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        Validate and process the input data.

        Validates the data format, removes metadata columns, and prepares
        the data for SHAP analysis.

        :param data: Input data for SHAP analysis
        :type data: Union[pd.DataFrame, np.ndarray]
        :raises ValueError: If data is not a supported format
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            log.error(f"Invalid data type: {type(data)}")
            raise ValueError(f"data must be a pandas DataFrame or a numpy array")
        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)

        # Filter to drop metadata columns
        self._data_with_metadata = data.copy()
        if self._metadata:
            if isinstance(data, pd.DataFrame):
                data = data.drop(columns=self._metadata, errors="ignore")
                log.debug(f"Removed metadata columns: {self._metadata}")

        self._data = data

    @property
    def metadata(self):
        """
        Get the metadata column names.

        Returns the list of column names that are considered metadata and
        excluded from SHAP analysis.

        :return: List of metadata column names
        :rtype: List[str]
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Union[str, List[str]]):
        """
        Validate and set the metadata column names.

        Converts metadata to a list format and validates the input.

        :param metadata: Column names to exclude from analysis
        :type metadata: Union[str, List[str]]
        :raises ValueError: If metadata is not a string or list
        """
        if metadata is None:
            metadata = []
        if not isinstance(metadata, (str, list)):
            log.error(f"Invalid metadata type: {type(metadata)}")
            raise ValueError(f"metadata must be a string or list")
        if isinstance(metadata, str):
            metadata = [metadata]

        self._metadata = metadata

    @property
    def shap_explainer(self):
        """
        Get the SHAP explainer object.

        Returns the configured SHAP explainer used for computing SHAP values.

        :return: SHAP explainer object
        :rtype: shap.Explainer
        """
        return self._explainer

    def _get_class_index(self, class_ind_name: Union[str, int]):
        """
        Get the index of a class based on its name or index.

        Converts class names to their corresponding indices for use in SHAP
        analysis. Supports both string class names and integer indices.

        :param class_ind_name: Name or index of the class
        :type class_ind_name: Union[str, int]
        :return: Integer index of the class
        :rtype: int
        :raises ValueError: If class name is not found or index is out of range

        Example:
            >>> analyzer._get_class_index("positive")  # Returns 1
            >>> analyzer._get_class_index(0)           # Returns 0
        """
        if isinstance(class_ind_name, str):
            try:
                class_index = list(self._model.classes_).index(class_ind_name)
            except ValueError:
                log.error(
                    f"Class name '{class_ind_name}' not found in {self._model.classes_}"
                )
                raise ValueError(f"class_name must be one of {self._model.classes_}")
        elif isinstance(class_ind_name, int):
            if len(self._model.classes_) <= class_ind_name:
                log.error(
                    f"Class index {class_ind_name} out of range for {len(self._model.classes_)} classes"
                )
                raise ValueError(
                    f"class_index must be less than {len(self._model.classes_)}"
                )
            class_index = class_ind_name
        else:
            log.error(f"Invalid class_ind_name type: {type(class_ind_name)}")
            raise ValueError(f"class_name must be a string or an integer")
        return class_index

    def _set_explainer(self):
        """
        Set the appropriate SHAP explainer based on the model type.

        Automatically selects the best SHAP explainer for the given model type:
        - TreeExplainer for tree-based models
        - KernelExplainer for other models

        :raises ValueError: If the model type is not supported
        """
        model_type = type(self._model).__name__
        if model_type in TREE_EXPLAINER_MODELS:
            log.info(f"Using TreeExplainer for {model_type}")
            self._explainer = shap.TreeExplainer(self._model)

        elif model_type in KERNEL_EXPLAINER_MODELS:
            log.info(f"Using KernelExplainer for {model_type}")
            self._explainer = shap.KernelExplainer(
                self._model.predict, shap.sample(self._data, 5)
            )

        else:
            log.error(f"Unsupported model type: {model_type}")
            raise ValueError(
                f"Model {model_type} is not supported by ShapAnalyzer class"
            )

    def _set_estimator(self, estimator):
        """
        Extract and configure the model from an estimator or pipeline.

        Handles both standalone models and sklearn Pipelines. For pipelines,
        it extracts the final estimator and applies any transformers to the data.
        Also extracts feature names for proper labeling in SHAP plots.

        :param estimator: Trained model or sklearn Pipeline
        :type estimator: object
        :raises AttributeError: If the model doesn't have required feature name attributes
        """
        if isinstance(estimator, Pipeline):
            log.info("Processing sklearn Pipeline")
            self._model = estimator.steps[-1][1]
            if len(estimator.steps) > 1:
                transformer = Pipeline(estimator.steps[0:-1])
                self._x_transformed = transformer.transform(self._data)
                self._feature_names = transformer.get_feature_names_out()
                log.info(f"Applied {len(estimator.steps)-1} transformers")
            else:
                self._x_transformed = self._data
                if isinstance(self._data, np.ndarray):
                    self._feature_names = [
                        f"Feature {i}" for i in range(self._data.shape[1])
                    ]
                else:
                    self._feature_names = self._get_feature_names(self._model)
        else:
            log.info(f"Processing standalone model: {type(estimator).__name__}")
            self._model = estimator
            self._x_transformed = self._data
            if isinstance(self._data, np.ndarray):
                self._feature_names = [
                    f"Feature {i}" for i in range(self._data.shape[1])
                ]
            else:
                self._feature_names = self._get_feature_names(self._model)

    @staticmethod
    def _get_feature_names(estimator):
        """
        Extract feature names from a trained estimator.

        Attempts to get feature names from various model types:
        - sklearn models: uses feature_names_in_ attribute
        - LightGBM models: uses feature_name_ attribute

        :param estimator: Trained model to extract feature names from
        :type estimator: object
        :return: List of feature names
        :rtype: list[str]
        :raises AttributeError: If the model doesn't have feature name attributes
        """
        try:
            # sklearn
            feature_names = estimator.feature_names_in_
        except AttributeError:
            try:
                # LightGBM
                feature_names = estimator.feature_name_
            except AttributeError:
                log.error(
                    f"Model {type(estimator).__name__} has no feature name attributes"
                )
                raise AttributeError(
                    "Model does not have attribute feature_names_in_ or feature_name_"
                )
        return feature_names

    @staticmethod
    def _save_fig(title: str, file_path_save: str):
        """
        Save the current matplotlib figure with a title.

        Saves the current figure to the specified path with proper formatting
        and closes the figure to free memory.

        :param title: Title for the figure
        :type title: str
        :param file_path_save: Path where to save the figure
        :type file_path_save: str
        :return: None
        """
        fig1 = plt.gcf()
        fig1.suptitle(title)
        fig1.tight_layout()
        fig1.savefig(file_path_save)
        plt.close()
        log.debug(f"Saved figure: {file_path_save}")

    def bar_summary_plot(self, file_path_save: str = None, **kwargs):
        """
        Create a bar plot showing mean absolute SHAP values for each feature.

        Generates a horizontal bar chart where each bar represents the mean
        absolute SHAP value for a feature, providing a clear view of feature
        importance. Features are sorted by importance (highest to lowest).

        :param file_path_save: Path to save the plot (optional)
        :type file_path_save: str, optional
        :param kwargs: Additional keyword arguments passed to shap.summary_plot
        :return: None

        Example:
            >>> analyzer.bar_summary_plot(show=True)
            >>> analyzer.bar_summary_plot(file_path_save="bar_plot.png")
        """
        if file_path_save != None:
            if not os.path.exists(os.path.dirname(file_path_save)):
                log.error(
                    f"Directory does not exist: {os.path.dirname(file_path_save)}"
                )
                raise ValueError(
                    f"Path: {os.path.dirname(file_path_save)} does not exist"
                )

        log.info("Generating bar summary plot")
        shap.summary_plot(
            self.shap_values,
            plot_type="bar",
            class_names=(
                self._model.classes_ if self._problem_type != "regression" else None
            ),
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            **kwargs,
        )

        if file_path_save != None:
            self._save_fig(
                title="Bar Summary Plot",
                file_path_save=(
                    f"{file_path_save}.png"
                    if not file_path_save.endswith(".png")
                    else file_path_save
                ),
            )

    def summary_plot(
        self, class_name: Union[str, int] = 1, file_path_save: str = None, **kwargs
    ):
        """
        Create a summary plot showing SHAP values for all features and samples.

        Generates a plot where each point represents a SHAP value for a feature
        and sample. Features are ordered by importance, and the color represents
        the feature value (high/low). This provides insight into how each feature
        affects predictions across all samples.

        :param class_name: Class to plot for (classification only, default: 1)
        :type class_name: Union[str, int], optional
        :param file_path_save: Path to save the plot (optional)
        :type file_path_save: str, optional
        :param kwargs: Additional keyword arguments passed to shap.summary_plot
        :return: None

        Example:
            >>> analyzer.summary_plot(show=True)
            >>> analyzer.summary_plot(class_name="positive", file_path_save="summary.png")
        """
        if file_path_save != None:
            if not os.path.exists(os.path.dirname(file_path_save)):
                log.error(
                    f"Directory does not exist: {os.path.dirname(file_path_save)}"
                )
                raise ValueError(
                    f"Path {os.path.dirname(file_path_save)} does not exist"
                )

        log.info(f"Generating summary plot for class: {class_name}")
        shap.summary_plot(
            (
                self.shap_values[self._get_class_index(class_name)]
                if self._problem_type != "regression"
                else self.shap_values
            ),
            self._x_transformed,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            **kwargs,
        )

        if file_path_save != None:
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

    def waterfall_plot(self, query: str, path_save: str = None, **kwargs):
        """
        Create waterfall plots for samples matching a query.

        Generates waterfall plots showing how each feature contributes to the
        final prediction for individual samples. The plot shows the base value
        and how each feature pushes the prediction up or down.

        :param query: Pandas query string to filter samples
        :type query: str
        :param path_save: Directory path to save waterfall plots
        :type path_save: str, optional
        :param kwargs: Additional keyword arguments passed to shap.waterfall_plot
        :return: None
        :raises ValueError: If path_save is not a directory or no data matches query

        Example:
            >>> analyzer.waterfall_plot("age > 30", path_save="./waterfalls")
            >>> analyzer.waterfall_plot("income > 50000", show=True)
        """
        if path_save is not None and not os.path.isdir(path_save):
            log.error(f"path_save must be a directory: {path_save}")
            raise ValueError("path_save must be a directory")

        log.info(f"Filtering data with query: {query}")
        filter_data = self._data_with_metadata.query(query).copy()
        if filter_data.shape[0] == 0:
            log.error(f"No data found for query: {query}")
            raise ValueError(f"No data found for query: {query}")
        else:
            log.info(f"Found {filter_data.shape[0]} samples matching query")
            list_idx = filter_data.index.to_list()
            for i, idx in enumerate(list_idx):
                if self._problem_type in [
                    "binary_classification",
                    "multiclass_classification",
                ]:
                    for j, class_name in enumerate(self._model.classes_):
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=self.shap_values[j][idx],
                                base_values=self._explainer.expected_value[j],
                                data=self._data.iloc[idx],
                                feature_names=self._feature_names,
                            ),
                            show=kwargs.get("show", False),
                            **kwargs,
                        )

                        if path_save != None:
                            self._save_fig(
                                title=f"Waterfall plot query: {query} (Sample {i})",
                                file_path_save=os.path.join(
                                    path_save,
                                    f"waterfall_class_{class_name}_sample_{i}.png",
                                ),
                            )
                else:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=self.shap_values[idx],
                            base_values=self._explainer.expected_value,
                            data=self._data.iloc[idx],
                            feature_names=self._feature_names,
                        ),
                        show=kwargs.get("show", False),
                        **kwargs,
                    )

                    if path_save != None:
                        self._save_fig(
                            title=f"Waterfall plot query: {query} (Sample {i})",
                            file_path_save=os.path.join(
                                path_save, f"waterfall_sample_{i}.png"
                            ),
                        )

    def partial_dependence_plot(
        self,
        feature: Union[str, int],
        interaction_feature: Union[str, int],
        class_name: Union[str, int] = None,
        file_path_save: str = None,
        **kwargs,
    ):
        """
        Create a partial dependence plot showing feature interactions.

        Generates a scatter plot showing how the SHAP values of one feature
        depend on the values of another feature, revealing potential interactions
        between features.

        :param feature: Primary feature to analyze
        :type feature: Union[str, int]
        :param interaction_feature: Feature to show interaction with
        :type interaction_feature: Union[str, int]
        :param class_name: Class to plot for (classification only)
        :type class_name: Union[str, int], optional
        :param file_path_save: Path to save the plot
        :type file_path_save: str, optional
        :param kwargs: Additional keyword arguments passed to shap.dependence_plot
        :return: None

        Example:
            >>> analyzer.partial_dependence_plot("age", "income", show=True)
            >>> analyzer.partial_dependence_plot(0, 1, file_path_save="pdp.png")
        """
        log.info(
            f"Generating partial dependence plot: {feature} vs {interaction_feature}"
        )
        shap.dependence_plot(
            feature,
            (
                self.shap_values[self._get_class_index(class_name)]
                if self._problem_type != "regression"
                else self.shap_values
            ),
            self._x_transformed,
            interaction_index=interaction_feature,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            **kwargs,
        )
        if file_path_save != None:
            self._save_fig(
                title=(
                    f"Partial dependence plot: {feature} and {interaction_feature}"
                    if interaction_feature != None
                    else f"Partial dependence plot: {feature}"
                ),
                file_path_save=file_path_save,
            )

    def get_sample_by_shap_value(
        self,
        shap_value,
        feature_name: Union[str, int],
        class_name: Union[str, int] = None,
        operator: str = ">=",
    ):
        """
        Filter data samples based on SHAP values for a specific feature.

        Returns a subset of the data where samples have SHAP values for the
        specified feature that meet the given condition (>= or <=). This is
        useful for analyzing samples with high/low feature importance.

        :param shap_value: Threshold SHAP value for filtering
        :type shap_value: float
        :param feature_name: Name or index of the feature to filter by
        :type feature_name: Union[str, int]
        :param class_name: Class to filter for (classification only)
        :type class_name: Union[str, int], optional
        :param operator: Comparison operator (">=" or "<=")
        :type operator: str
        :return: Filtered DataFrame with matching samples
        :rtype: pd.DataFrame
        :raises ValueError: If operator is invalid or feature not found

        Example:
            >>> high_impact = analyzer.get_sample_by_shap_value(0.5, "age", operator=">=")
            >>> low_impact = analyzer.get_sample_by_shap_value(-0.3, "income", operator="<=")
        """
        operator_dict = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        if operator not in operator_dict.keys():
            log.error(f"Invalid operator: {operator}")
            raise ValueError(
                f"Operator {operator} not valid. Valid operators are: {operator_dict.keys()}"
            )

        if feature_name not in self._feature_names:
            log.error(f"Feature {feature_name} not found in {self._feature_names}")
            raise ValueError(
                f"Feature {feature_name} is not in model. Must be one of: {self._feature_names}"
            )
        index_feature = list(self._feature_names).index(feature_name)
        log.info(
            f"Filtering samples with {operator} {shap_value} for feature '{feature_name}'"
        )

        if self._problem_type in ["binary_classification", "multiclass_classification"]:
            return self._data_with_metadata[
                operator_dict[operator](
                    self.shap_values[
                        list(self._model.classes_).index(
                            self._get_class_index(class_name)
                        )
                    ][:, index_feature],
                    shap_value,
                )
            ].copy()
        else:
            return self._data_with_metadata[
                operator_dict[operator](self.shap_values[:, index_feature], shap_value)
            ].copy()

    def make_shap_analysis(
        self, queries: List[str] = None, pdp_tuples: List[tuple] = None
    ):
        """
        Generate a comprehensive SHAP analysis with multiple plot types.

        Creates a complete SHAP analysis including summary plots, bar plots,
        waterfall plots (if queries provided), and partial dependence plots
        (if feature tuples provided). All plots are saved to organized directories.

        :param queries: List of pandas query strings for waterfall plots
        :type queries: List[str], optional
        :param pdp_tuples: List of (feature1, feature2) tuples for partial dependence plots
        :type pdp_tuples: List[tuple], optional
        :return: None
        :raises ValueError: If shap_folder is not set

        Example:
            >>> analyzer.make_shap_analysis()
            >>> analyzer.make_shap_analysis(
            ...     queries=["age > 30", "income > 50000"],
            ...     pdp_tuples=[("age", "income"), ("height", "weight")]
            ... )
        """
        # Check path to save plots
        if self._shap_folder == None:
            log.error("shap_folder is not set")
            raise ValueError(
                "Set path to save plots: the attribute shap_folder is None"
            )
        base_path = os.path.join(
            self._shap_folder,
            "shap_analysis",
            self.model_name,
        )
        log.info(f"Starting comprehensive SHAP analysis for {self.model_name}")

        list_paths = [
            os.path.join(base_path, "summary/"),
            os.path.join(base_path, "individual/"),
            os.path.join(base_path, "partial_dependence/"),
        ]
        # Make dirs to save plots
        _ = [os.makedirs(os.path.dirname(path), exist_ok=True) for path in list_paths]
        log.info(f"Created analysis directories in: {base_path}")

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
            if queries != None:
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
                                f"pdp_class_{class_index}_{pdp_tuple[0]}{'_' + pdp_tuple[1] if pdp_tuple[1] != None else ''}.png",
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
            if queries != None:
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
                            f"pdp_{pdp_tuple[0]}{'_'+pdp_tuple[1] if pdp_tuple[1] != None else ''}.png",
                        ),
                    )
