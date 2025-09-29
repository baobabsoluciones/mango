"""Main SHAP explainer class for model interpretability."""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import shap
from sklearn.pipeline import Pipeline
from .explainers import TreeExplainer, DeepExplainer, LinearExplainer, KernelExplainer
from .visualizers import SummaryPlot, WaterfallPlot, ForcePlot, DependencePlot
from .utils import DataProcessor, ExportUtils, InputValidator
from .logging.logger import get_logger


class SHAPExplainer:
    """
    Main class for SHAP model interpretability analysis.

    This class provides a unified interface for generating SHAP explanations
    for various types of machine learning models, including support for
    pipelines, different problem types, and comprehensive analysis workflows.

    :param model: Trained machine learning model or pipeline
    :param data: Data used for SHAP calculations
    :param problem_type: Type of problem ('regression', 'binary_classification', 'multiclass_classification')
    :param model_name: Name of the model for saving results
    :param metadata: Columns to treat as metadata (not used for SHAP calculations)
    :param shap_folder: Folder to save SHAP analysis results
    :param model_type: Type of model ('tree', 'deep', 'linear', 'kernel') - auto-detected if None
    :param feature_names: Names of features in the data - auto-detected if None
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
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the SHAP explainer."""
        self.logger = get_logger(__name__)

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
        self._set_explainer(model_type)

        # Get SHAP values
        self.shap_values = self._explainer.shap_values(self._x_transformed)

        self.logger.info(
            f"SHAP explainer initialized with {self._model_type} explainer"
        )

    @property
    def problem_type(self) -> str:
        """Get the problem type."""
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: str) -> None:
        """Validate and set the problem type."""
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
        """Get the model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        """Set the model name."""
        self._model_name = model_name

    @property
    def shap_folder(self) -> Optional[str]:
        """Get the SHAP folder."""
        return self._shap_folder

    @shap_folder.setter
    def shap_folder(self, shap_folder: Optional[str]) -> None:
        """Validate and set the SHAP folder."""
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
        """Get the data."""
        return self._data

    @data.setter
    def data(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Validate and set the data."""
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("data must be a pandas DataFrame or a numpy array")

        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)

        # Filter to drop metadata columns
        self._data_with_metadata = data.copy()
        if self._metadata:
            if isinstance(data, pd.DataFrame):
                data = data.drop(columns=self._metadata, errors="ignore")

        self._data = data

    @property
    def metadata(self) -> List[str]:
        """Get the metadata columns."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Optional[Union[str, List[str]]]) -> None:
        """Validate and set the metadata columns."""
        if metadata is None:
            metadata = []
        if not isinstance(metadata, (str, list)):
            raise ValueError("metadata must be a string or list of strings")
        if isinstance(metadata, str):
            metadata = [metadata]
        self._metadata = metadata

    def _get_class_index(self, class_ind_name: Union[str, int]) -> int:
        """Get the index of a class based on its name or index."""
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
        """Extract the model from a pipeline and get feature names."""
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
                    self._feature_names = self._get_feature_names(self._model)
        else:
            self._model = estimator
            self._x_transformed = self._data
            if isinstance(self._data, np.ndarray):
                self._feature_names = [
                    f"Feature {i}" for i in range(self._data.shape[1])
                ]
            else:
                self._feature_names = self._get_feature_names(self._model)

    @staticmethod
    def _get_feature_names(estimator: Any) -> List[str]:
        """Get feature names from an estimator."""
        try:
            # sklearn
            feature_names = estimator.feature_names_in_
        except AttributeError:
            try:
                # LightGBM
                feature_names = estimator.feature_name_
            except AttributeError:
                raise AttributeError(
                    "Model does not have attribute feature_names_in_ or feature_name_"
                )
        return feature_names

    def _detect_model_type(self, model: Any) -> str:
        """Automatically detect the type of model."""
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
        """Set the SHAP explainer based on the model type."""
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
        """Save the current figure with a title."""
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
        Generate SHAP values for the given data.

        :param data: Data to explain
        :param max_evals: Maximum number of evaluations for kernel explainer
        :return: SHAP values
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

        :param class_name: Class to plot for classification problems
        :param file_path_save: Path to save the plot
        :param kwargs: Additional arguments for the plot
        """
        if file_path_save is not None:
            if not os.path.exists(os.path.dirname(file_path_save)):
                raise ValueError(
                    f"Path {os.path.dirname(file_path_save)} does not exist"
                )

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

        :param file_path_save: Path to save the plot
        :param kwargs: Additional arguments for the plot
        """
        if file_path_save is not None:
            if not os.path.exists(os.path.dirname(file_path_save)):
                raise ValueError(
                    f"Path: {os.path.dirname(file_path_save)} does not exist"
                )

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
        self, query: str, path_save: Optional[str] = None, **kwargs
    ) -> None:
        """
        Create waterfall plots for samples matching a query.

        :param query: Query to filter the data
        :param path_save: Path to save the plots
        :param kwargs: Additional arguments for the plot
        """
        if path_save is not None and not os.path.isdir(path_save):
            raise ValueError("path_save must be a directory")

        filter_data = self._data_with_metadata.query(query).copy()
        if filter_data.shape[0] == 0:
            raise ValueError(f"No data found for query: {query}")
        else:
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

                        if path_save is not None:
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

                    if path_save is not None:
                        self._save_fig(
                            title=f"Waterfall plot query: {query} (Sample {i})",
                            file_path_save=os.path.join(
                                path_save, f"waterfall_sample_{i}.png"
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

        :param feature: Feature to plot
        :param interaction_feature: Interaction feature
        :param class_name: Class for classification problems
        :param file_path_save: Path to save the plot
        :param kwargs: Additional arguments for the plot
        """
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
        shap_value: float,
        feature_name: Union[str, int],
        class_name: Optional[Union[str, int]] = None,
        operator: str = ">=",
    ) -> pd.DataFrame:
        """
        Get samples based on SHAP values for a specific feature.

        :param shap_value: SHAP value threshold
        :param feature_name: Feature name
        :param class_name: Class for classification problems
        :param operator: Comparison operator ('>=', '<=')
        :return: Filtered DataFrame
        """
        operator_dict = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        if operator not in operator_dict.keys():
            raise ValueError(
                f"Operator {operator} not valid. Valid operators are: {operator_dict.keys()}"
            )

        if feature_name not in self._feature_names:
            raise ValueError(
                f"Feature {feature_name} is not in model. Must be one of: {self._feature_names}"
            )
        index_feature = list(self._feature_names).index(feature_name)

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
        self,
        queries: Optional[List[str]] = None,
        pdp_tuples: Optional[List[tuple]] = None,
    ) -> None:
        """
        Perform comprehensive SHAP analysis.

        :param queries: List of queries for waterfall plots
        :param pdp_tuples: List of tuples for partial dependence plots
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
    ) -> None:
        """
        Export SHAP explanations to file.

        :param output_path: Path to save the explanations
        :param format: Export format ('csv', 'json', 'html')
        """
        self.export_utils.export(
            shap_values=self.shap_values,
            data=self._x_transformed,
            feature_names=self._feature_names,
            output_path=output_path,
            format=format,
        )
