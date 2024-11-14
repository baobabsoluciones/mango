""" This file contains the constants used in the shap_analysis module. """

TREE_EXPLAINER_MODELS = [
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

KERNEL_EXPLAINER_MODELS = ["LogisticRegression", "LinearRegression"]
