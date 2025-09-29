"""Constants for SHAP model types."""

# Tree-based models that support TreeExplainer
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

# Models that support KernelExplainer (model-agnostic)
KERNEL_EXPLAINER_MODELS = ["LogisticRegression", "LinearRegression"]

# Deep learning models that support DeepExplainer
DEEP_EXPLAINER_MODELS = [
    "Sequential",
    "Model",
    "Functional",
]

# Linear models that support LinearExplainer
LINEAR_EXPLAINER_MODELS = [
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LinearSVC",
    "LinearSVR",
    "SGDClassifier",
    "SGDRegressor",
]
