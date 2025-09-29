"""Example usage of mango_shap library."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from mango_shap import SHAPExplainer


def main():
    """Demonstrate mango_shap functionality."""

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    # Create DataFrame with feature names and metadata
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_names)
    data["id"] = range(len(data))
    data["timestamp"] = pd.date_range("2023-01-01", periods=len(data), freq="H")

    # Split data
    X_train = data.iloc[:800]
    X_test = data.iloc[800:]
    y_train = y[:800]
    y_test = y[800:]

    # Create and train a pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Train the pipeline
    pipeline.fit(X_train.drop(columns=["id", "timestamp"]), y_train)

    # Create SHAP explainer
    explainer = SHAPExplainer(
        model=pipeline,
        data=X_train,
        problem_type="binary_classification",
        model_name="example_model",
        metadata=["id", "timestamp"],
        shap_folder="./shap_results",
    )

    print("SHAP Explainer created successfully!")
    print(f"Model type detected: {explainer._model_type}")
    print(f"Feature names: {explainer._feature_names}")
    print(f"SHAP values shape: {explainer.shap_values.shape}")

    # Generate individual plots
    print("\nGenerating summary plots...")
    explainer.summary_plot(class_name=1, file_path_save="summary_plot.png")
    explainer.bar_summary_plot(file_path_save="bar_plot.png")

    # Generate waterfall plots for specific queries
    print("Generating waterfall plots...")
    explainer.waterfall_plot(query="feature_0 > 1.0", path_save="./waterfall_plots/")

    # Generate partial dependence plots
    print("Generating partial dependence plots...")
    explainer.partial_dependence_plot(
        feature="feature_0",
        interaction_feature="feature_1",
        file_path_save="pdp_plot.png",
    )

    # Get samples with high SHAP values
    print("Finding high-impact samples...")
    high_impact_samples = explainer.get_sample_by_shap_value(
        shap_value=0.3, feature_name="feature_0", operator=">="
    )
    print(f"Found {len(high_impact_samples)} samples with high feature_0 impact")

    # Perform comprehensive analysis
    print("Performing comprehensive SHAP analysis...")
    explainer.make_shap_analysis(
        queries=["feature_0 > 1.0", "feature_1 < -1.0"],
        pdp_tuples=[("feature_0", "feature_1"), ("feature_2", None)],
    )

    # Export explanations
    print("Exporting explanations...")
    explainer.export_explanations(output_path="shap_explanations.csv", format="csv")

    print("Analysis complete! Check the shap_results folder for all outputs.")


if __name__ == "__main__":
    main()
