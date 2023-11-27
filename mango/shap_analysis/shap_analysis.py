import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class ShapAnalyzer:
    """
    Class to analyze the shap values of a model.

    :param problem_type: Problem type of the model
    :param estimator: Model to analyze
    :param data: Data used to train the model
    :param metadata: Metadata of the data
    :param shap_folder: Folder where the shap analysis will be saved
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
        estimator: Union[object, dict[str, object]],
        data: Union[pd.DataFrame, np.ndarray],
        metadata: pd.DataFrame = None,
        shap_folder: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shap_analysis"
        ),
    ):
        self.problem_type = problem_type
        self.shap_folder = shap_folder
        self.data = data
        self._metadata = metadata

        # Assign model
        self._get_estimator(estimator)

        # Assign shap explainer
        self._get_explainer()

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
            raise ValueError(f"Path: {shap_folder} does not exist")
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
        self._data = data

    @property
    def shap_explainer(self):
        """
        This property is the shap explainer.
        :return: Shap explainer
        :doc-author: baobab soluciones
        """
        return self._explainer

    def _get_explainer(self):
        """
        Get the shap explainer based on the model type.

        :return: Shap explainer
        :doc-author: baobab soluciones
        """
        if isinstance(
            self._model,
            (
                RandomForestClassifier,
                DecisionTreeClassifier,
                GradientBoostingClassifier,
                XGBClassifier,
                CatBoostClassifier,
                LGBMClassifier,
                RandomForestRegressor,
                DecisionTreeRegressor,
                GradientBoostingRegressor,
                XGBRegressor,
                CatBoostRegressor,
                LGBMRegressor,
            ),
        ):
            # For tree-based models
            shap_explainer = shap.TreeExplainer(self._model)

        elif isinstance(
            self._model,
            (
                LogisticRegression,
                SVC,
                KNeighborsClassifier,
                KNeighborsRegressor,
            ),
        ):
            # For linear models and non-tree-based models
            shap_explainer = shap.KernelExplainer(
                self._model.predict_proba, shap.sample(self._data, 5)
            )

        # Add more conditions for other types of models as needed
        else:
            # Default to KernelExplainer for unsupported models
            shap_explainer = shap.KernelExplainer(
                self._model.predict, shap.sample(self._data, 5)
            )

        return shap_explainer

    def _get_estimator(self, estimator):
        """
        The _get_estimator function is used to extract the model from a pipeline.
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
        :return: A bar plot of the shap values of all features
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
        :return: A plot of the shap values for each feature
        :doc-author: baobab soluciones
        """
        if path_save != None:
            if not os.path.exists(os.path.dirname(path_save)):
                raise ValueError(f"Path {os.path.dirname(path_save)} does not exist")

        # Get shap_values
        if isinstance(self.shap_values, list):
            sh_values = self.shap_values[class_index]
        elif isinstance(self.shap_values, np.ndarray):
            sh_values = self.shap_values
        else:
            raise ValueError(f"shap_values must be a list or a numpy array")

        shap.summary_plot(
            sh_values,
            self._x_transformed,
            feature_names=self._feature_names,
            show=kwargs.get("show", False),
            sort=kwargs.get("sort", True),
        )

        if path_save != None:
            fig1 = plt.gcf()
            fig1.savefig(
                f"{path_save}.png" if not path_save.endswith(".png") else path_save
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
        if self._problem_type != "regression":
            if class_name not in self._model.classes_:
                raise ValueError(
                    f"Clase {class_name} no asociada al modelo. Debe ser uno de {self._model.classes_}"
                )
            index_class = self._model.classes_.tolist().index(class_name)
        else:
            index_class = 0

        if feature_name not in self._feature_names:
            raise ValueError(
                f"Feature {feature_name} no asociada al modelo. Debe ser uno de {self._feature_names}"
            )
        index_feature = list(self._feature_names).index(feature_name)

        # Get shap_values
        if isinstance(self.shap_values, list):
            sh_values = self.shap_values[index_class]
        elif isinstance(self.shap_values, np.ndarray):
            sh_values = self.shap_values
        else:
            raise ValueError(f"shap_values must be a list or a numpy array")

        return self._data[
            operator_dict[operator](sh_values[:, index_feature], shap_value)
        ].copy()
