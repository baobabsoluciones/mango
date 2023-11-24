import os
from typing import Union, List

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ShapAnalyzer:
    # TODO: introduce more than one estimator
    def __init__(
        self,
        *,
        problem_type: str,
        estimator: Union[object, List[object]],
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
        :return: problem type
        """
        return self._problem_type

    @problem_type.setter
    def problem_type(self, problem_type: str):
        # Validate problem_type options
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
        :return: shap folder
        """
        return self._shap_folder

    @shap_folder.setter
    def shap_folder(self, shap_folder: str):
        if not os.path.exists(shap_folder):
            raise ValueError(f"Path: {shap_folder} does not exist")
        self._shap_folder = shap_folder

    @property
    def data(self):
        """
        :return: data
        """
        return self._data

    @data.setter
    def data(self, data: Union[pd.DataFrame, np.ndarray]):
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError(f"data must be a pandas DataFrame or a numpy array")
        self._data = data

    @property
    def shap_explainer(self):
        """
        :return: shap explainer
        """
        return self._explainer

    def _get_explainer(self):
        if self._problem_type == "binary_classification":
            self._explainer = shap.TreeExplainer(
                self._model,
                feature_names=self._feature_names,
            )
        else:
            raise NotImplementedError(
                f"Problem type {self._problem_type} not implemented"
            )

    def _get_estimator(self, estimator):
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
                    self._feature_names = self.get_feature_names(self._model)
        else:
            self._model = estimator
            self._x_transformed = self._data
            if isinstance(self._data, np.ndarray):
                self._feature_names = [f"x{i}" for i in range(self._data.shape[1])]
            else:
                self._feature_names = self.get_feature_names(self._model)

    @classmethod
    def get_feature_names(cls, estimator):
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
        if path_save != None:
            if not os.path.exists(os.path.dirname(path_save)):
                raise ValueError(f"Path: {os.path.dirname(path_save)} does not exist")

        shap.summary_plot(
            self.shap_values,
            plot_type="bar",
            class_names=self._model.classes_,
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
        operator_dict = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        # TODO: Check when the self.problem_type is regression
        if class_name not in self._model.classes_:
            raise ValueError(
                f"Clase {class_name} no asociada al modelo. Debe ser uno de {self._model.classes_}"
            )
        index_class = self._model.classes_.tolist().index(class_name)
        if feature_name not in self._feature_names:
            raise ValueError(
                f"Feature {feature_name} no asociada al modelo. Debe ser uno de {self._feature_names}"
            )
        index_feature = list(self._feature_names).index(feature_name)
        return self._data[
            operator_dict[operator](
                self.shap_values[index_class][:, index_feature], shap_value
            )
        ].copy()
