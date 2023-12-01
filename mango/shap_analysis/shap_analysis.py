import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class ShapAnalyzer:
    """
    Class to analyze the shap values of a model.

    :doc-author: baobab soluciones

    Usage
    -----

    >>> from mango.shap_analysis import ShapAnalyzer
    >>> shap_analyzer = ShapAnalyzer(
    ...     problem_type="regression",
    ...     estimator=estimator,
    ...     data=data
    ... )
    >>> shap_analyzer.summary_plot(show=True)
    >>> shap_analyzer.bar_summary_plot(show=True)
    >>> sample_shap = shap_analyzer.get_sample_by_shap_value(shap_value=0.5, feature_name="feature_name")
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
        # Tree Explainer models name
        self._TREE_EXPLAINER_MODELS = [
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "CatBoostClassifier",
            "CatBoostRegressor",
            "RandomForestClassifier",
            "RandomForestRegressor",
            "ExtraTreesClassifier",
            "ExtraTreesRegressor",
        ]

        # Kernel Explainer models name
        self._KERNEL_EXPLAINER_MODELS = ["LogisticRegression", "LinearRegression"]

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
        self.shap_values = self._explainer.shap_values(self._x_transformed)

    @property
    def problem_type(self):
        """
        This property is the problem type of the model.
        :return: problem type
        :doc-author: baobab soluciones
        """
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: str):
        """
        Validate the problem_type and set the problem_type attribute of the class.
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
    def model_name(self):
        """
        This property is the model name.
        :return: model name
        :doc-author: baobab soluciones
        """
        return self._model_name

    @property
    def shap_folder(self):
        """
        This property is the shap folder.
        :return: shap folder
        :doc-author: baobab soluciones
        """
        return self._shap_folder

    @shap_folder.setter
    def shap_folder(self, shap_folder: str):
        """
        Validate the shap_folder and set the shap_folder attribute of the class.
        """
        if not os.path.exists(shap_folder):
            try:
                os.makedirs(shap_folder)
            except OSError:
                raise OSError(f"Creation of the directory {shap_folder} failed")

        self._shap_folder = shap_folder

    @property
    def data(self):
        """
        This property is the data used to train the model.
        :return: Data used to train the model
        :doc-author: baobab soluciones
        """
        return self._data

    @data.setter
    def data(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        Validate the data and set the data attribute of the class.
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError(f"data must be a pandas DataFrame or a numpy array")
        if isinstance(data, pd.DataFrame):
            data.reset_index(drop=True, inplace=True)

        # Filter to drop metadata columns
        if self._metadata:
            if isinstance(data, pd.DataFrame):
                self._data_with_metadata = data.copy()
                data = data.drop(columns=self._metadata)

        self._data = data

    @property
    def metadata(self):
        """
        This property is the data used to train the model.
        :return: Columns of data that are metadata
        :doc-author: baobab soluciones
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Union[str, List[str]]):
        """
        Validate the data and set the data attribute of the class.
        """
        if not isinstance(metadata, (str, list)):
            raise ValueError(f"data must be a pandas DataFrame or a numpy array")
        if isinstance(metadata, str):
            metadata = [metadata]

        self._metadata = metadata

    @property
    def shap_explainer(self):
        """
        This property is the shap explainer.
        :return: Shap explainer
        :doc-author: baobab soluciones
        """
        return self._explainer

    def _set_explainer(self):
        """
        Get the shap explainer based on the model type.

        :return: Shap explainer
        :doc-author: baobab soluciones
        """
        if type(self._model).__name__ in self._TREE_EXPLAINER_MODELS:
            self._explainer = shap.TreeExplainer(self._model)

        elif type(self._model).__name__ in self._KERNEL_EXPLAINER_MODELS:
            self._explainer = shap.KernelExplainer(
                self._model.predict, shap.sample(self._data, 5)
            )

        else:
            raise ValueError(
                f"Model {type(self._model).__name__} is not supported by ShapAnalyzer class"
            )

    def _set_estimator(self, estimator):
        """
        The _set_estimator function is used to extract the model from a pipeline.
        It also extracts the feature names and transformed data if there are any transformers in the pipeline.


        :param self: Access the class attributes and methods
        :param estimator: Pass the model to be used for prediction
        :return: The model, the feature names and the transformed data
        :doc-author: baobab soluciones
        """
        if isinstance(estimator, Pipeline):
            self._model = estimator.steps[-1][1]
            if len(estimator.steps) > 1:
                transformer = Pipeline(estimator.steps[0:-1])
                self._x_transformed = transformer.transform(self._data)
                self._feature_names = transformer.get_feature_names_out()
            else:
                self._x_transformed = self._data
                if isinstance(self._data, np.ndarray):
                    self._feature_names = [f"x{i}" for i in range(self._data.shape[1])]
                else:
                    self._feature_names = self._get_feature_names(self._model)
        else:
            self._model = estimator
            self._x_transformed = self._data
            if isinstance(self._data, np.ndarray):
                self._feature_names = [f"x{i}" for i in range(self._data.shape[1])]
            else:
                self._feature_names = self._get_feature_names(self._model)

    @staticmethod
    def _get_feature_names(estimator):
        """
        The _get_feature_names function is a helper function that attempts to get the feature names from an estimator.
        It first tries to get the feature_names_in_ attribute, which is used by sklearn models. If this fails, it then tries
        to get the feature_name attribute, which is used by LightGBM models.

        :param estimator: Pass the model to be used for feature importance
        :return: The feature names of the model
        :doc-author: baobab soluciones
        """
        try:
            # sklearn
            feature_names = estimator.feature_names_in_
        except AttributeError:
            try:
                # LightGBM
                feature_names = estimator.feature_name_
            except AttributeError:
                raise AttributeError(
                    "Model does not have attribute feature_names_in_ or feature_names_"
                )
        return feature_names

    def bar_summary_plot(self, path_save: str = None, **kwargs):
        """
        The bar_summary_plot function is a wrapper for the SHAP summary_plot function.
        It takes in the shap values and plots them as a bar chart, with each feature on the x-axis and its corresponding
        SHAP value on the y-axis. The plot can be sorted by mean absolute value or not, depending on user preference.

        :param self: Make the function a method of the class
        :param path_save: str: Specify the path to save the plot
        :param **kwargs: Pass keyword arguments to the function
        :return: None
        :doc-author: baobab soluciones
        """
        if path_save != None:
            if not os.path.exists(os.path.dirname(path_save)):
                raise ValueError(f"Path: {os.path.dirname(path_save)} does not exist")

        shap.summary_plot(
            self.shap_values,
            plot_type="bar",
            class_names=self._model.classes_
            if self._problem_type != "regression"
            else None,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            sort=kwargs.get("sort", True),
        )

        if path_save != None:
            fig1 = plt.gcf()
            fig1.suptitle("Bar Summary Plot")
            fig1.tight_layout()
            fig1.savefig(
                f"{path_save}.png" if not path_save.endswith(".png") else path_save
            )
            plt.close()

    def summary_plot(self, class_index: int = 1, path_save: str = None, **kwargs):
        """
        The summary_plot function plots the SHAP values of every feature for all samples.
        The plot is a standard deviation centered histogram of the impacts each feature has on the model output.
        The color represents whether that impact was positive or negative and intensity shows how important it was.
        This function works with Numpy arrays or pandas DataFrames as input, and can plot either regression or classification models.

        :param self: Refer to the object itself
        :param class_index: int: Specify which class to plot the summary for
        :param path_save: str: Save the plot as a png file
        :param **kwargs: Pass keyworded, variable-length argument list to a function
        :return: None
        :doc-author: baobab soluciones
        """
        if path_save != None:
            if not os.path.exists(os.path.dirname(path_save)):
                raise ValueError(f"Path {os.path.dirname(path_save)} does not exist")

        shap.summary_plot(
            self.shap_values[class_index]
            if self._problem_type != "regression"
            else self.shap_values,
            self._x_transformed,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            sort=kwargs.get("sort", True),
        )

        if path_save != None:
            fig1 = plt.gcf()
            fig1.suptitle(
                f"Summary Plot class {self._model.classes_[class_index]}"
                if self._problem_type != "regression"
                else "Summary Plot"
            )
            fig1.tight_layout()
            fig1.savefig(
                f"{path_save}.png" if not path_save.endswith(".png") else path_save
            )
            plt.close()

    def waterfall_plot(self, query: str, path_save: str = None, **kwargs):
        """
        The waterfall_plot function plots the SHAP values for a single sample.

        :param self: Make the method belong to the class
        :param str query: Filter the data
        :param str path_save: Specify the path to save the waterfall plot
        :param **kwargs: Pass keyworded, variable-length argument list
        :return: None
        :doc-author: baobab soluciones
        """
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
                        )

                        if path_save != None:
                            fig1 = plt.gcf()
                            fig1.suptitle(f"Waterfall plot query: {query} (Sample {i})")
                            fig1.tight_layout()
                            fig1.savefig(
                                f"{path_save[:-4]}_class_{class_name}_sample_{i}{path_save[-4:]}"
                                if path_save.endswith(".png")
                                else f"{path_save}_{class_name}.png"
                            )
                            plt.close()
                else:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=self.shap_values[idx],
                            base_values=self._explainer.expected_value,
                            data=self._data.iloc[idx],
                            feature_names=self._feature_names,
                        ),
                        show=kwargs.get("show", False),
                    )

                    if path_save != None:
                        fig1 = plt.gcf()
                        fig1.suptitle(f"Waterfall plot query: {query} (Sample {i})")
                        fig1.tight_layout()
                        fig1.savefig(
                            f"{path_save[:-4]}_sample_{i}{path_save[-4:]}"
                            if path_save.endswith(".png")
                            else f"{path_save}.png"
                        )
                        plt.close()

    def get_sample_by_shap_value(
        self, shap_value, feature_name, class_name: str = None, operator: str = ">="
    ):
        """
        The get_sample_by_shap_value function returns a sample of the data that has a shap value for
        a given feature and class greater than or equal to the specified shap_value.

        :param self: Bind the method to a class
        :param shap_value: Specify the value of shap that we want to use as a filter
        :param feature_name: Specify the feature name that we want to use in our analysis
        :param class_name: str: Specify the class for which we want to get samples
        :param operator: str: Specify the operator to use when comparing the shap_value with the feature value
        :return: A dataframe with the samples that have a shap value greater than or equal to the one specified
        :doc-author: baobab soluciones
        """
        operator_dict = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        if (
            self._problem_type in ["binary_classification", "multiclass_classification"]
        ) and (class_name not in self._model.classes_):
            raise ValueError(
                f"Clase {class_name} no asociada al modelo. Debe ser uno de {self._model.classes_}"
            )

        if feature_name not in self._feature_names:
            raise ValueError(
                f"Feature {feature_name} no asociada al modelo. Debe ser uno de {self._feature_names}"
            )
        index_feature = list(self._feature_names).index(feature_name)

        return self._data[
            operator_dict[operator](self.shap_values[:, index_feature], shap_value)
        ].copy()

    def make_shap_analysis(self, queries: List[str] = None):
        """
        The make_shap_analysis function is a wrapper function that calls the summary_plot and bar_summary_plot functions.
        It also checks if the shap folder exists, and creates it if not. It then saves all plots to this folder.

        :param self: Bind the method to an object
        :param List[str] queries: Specify the queries to use in the waterfall plot
        :return: None
        :doc-author: baobab soluciones
        """
        # Check path to save plots
        if self._shap_folder == None:
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
        ]
        # Make dirs to save plots
        _ = [os.makedirs(os.path.dirname(path), exist_ok=True) for path in list_paths]

        # Make summary plot
        if self._problem_type in ["binary_classification", "multiclass_classification"]:
            _ = [
                self.summary_plot(
                    class_index=class_index,
                    path_save=os.path.join(
                        base_path,
                        "summary",
                        f"summary_class_{class_index}.png",
                    ),
                )
                for class_index in range(len(self._model.classes_))
            ]
            self.bar_summary_plot(
                path_save=os.path.join(
                    base_path,
                    "summary",
                    "barplot.png",
                )
            )
            for query in queries:
                self.waterfall_plot(
                    query=query,
                    path_save=os.path.join(
                        base_path,
                        "individual",
                        f"{query}.png",
                    ),
                )

        elif self._problem_type == "regression":
            self.summary_plot(
                path_save=os.path.join(
                    base_path,
                    "summary",
                    "summary.png",
                )
            )
            self.bar_summary_plot(
                path_save=os.path.join(
                    base_path,
                    "summary",
                    "barplot.png",
                )
            )
            for query in queries:
                self.waterfall_plot(
                    query=query,
                    path_save=os.path.join(
                        base_path,
                        "individual",
                        f"{query}.png",
                    ),
                )
