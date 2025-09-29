"""Export utilities for SHAP explanations."""

import numpy as np
import pandas as pd
import json
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from ..logging.logger import get_logger


class ExportUtils:
    """
    Utility class for exporting SHAP explanations.

    Supports various export formats including CSV, JSON, and HTML.
    """

    def __init__(self) -> None:
        """Initialize the export utilities."""
        self.logger = get_logger(__name__)

    def export(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        output_path: str = "shap_explanations",
        format: str = "csv",
    ) -> None:
        """
        Export SHAP explanations to file.

        :param shap_values: SHAP values to export
        :param data: Data used to generate SHAP values
        :param feature_names: Names of features
        :param output_path: Path to save the explanations
        :param format: Export format ('csv', 'json', 'html')
        """
        self.logger.info(
            f"Exporting SHAP explanations to {output_path} in {format} format"
        )

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "csv":
            self._export_csv(shap_values, data, feature_names, output_path)
        elif format.lower() == "json":
            self._export_json(shap_values, data, feature_names, output_path)
        elif format.lower() == "html":
            self._export_html(shap_values, data, feature_names, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Export completed successfully")

    def _export_csv(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]],
        output_path: Path,
    ) -> None:
        """
        Export SHAP values to CSV format.

        :param shap_values: SHAP values to export
        :param data: Data used to generate SHAP values
        :param feature_names: Names of features
        :param output_path: Path to save the file
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        # Create DataFrame with SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.index.name = "instance"

        # Add instance data if available
        if isinstance(data, pd.DataFrame):
            data_df = data.copy()
            data_df.columns = [f"data_{col}" for col in data_df.columns]
            combined_df = pd.concat([data_df, shap_df], axis=1)
        else:
            data_df = pd.DataFrame(
                data, columns=[f"data_{name}" for name in feature_names]
            )
            combined_df = pd.concat([data_df, shap_df], axis=1)

        # Save to CSV
        csv_path = output_path.with_suffix(".csv")
        combined_df.to_csv(csv_path)
        self.logger.info(f"CSV export saved to {csv_path}")

    def _export_json(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]],
        output_path: Path,
    ) -> None:
        """
        Export SHAP values to JSON format.

        :param shap_values: SHAP values to export
        :param data: Data used to generate SHAP values
        :param feature_names: Names of features
        :param output_path: Path to save the file
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        # Prepare data for JSON export
        export_data = {
            "shap_values": shap_values.tolist(),
            "feature_names": feature_names,
            "data": data.tolist()
            if isinstance(data, np.ndarray)
            else data.to_dict("records"),
            "metadata": {
                "num_instances": shap_values.shape[0],
                "num_features": shap_values.shape[1],
                "data_type": type(data).__name__,
            },
        }

        # Save to JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=2)
        self.logger.info(f"JSON export saved to {json_path}")

    def _export_html(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]],
        output_path: Path,
    ) -> None:
        """
        Export SHAP values to HTML format.

        :param shap_values: SHAP values to export
        :param data: Data used to generate SHAP values
        :param feature_names: Names of features
        :param output_path: Path to save the file
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        # Create HTML content
        html_content = self._generate_html_content(shap_values, data, feature_names)

        # Save to HTML
        html_path = output_path.with_suffix(".html")
        with open(html_path, "w") as f:
            f.write(html_content)
        self.logger.info(f"HTML export saved to {html_path}")

    def _generate_html_content(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
    ) -> str:
        """
        Generate HTML content for SHAP explanations.

        :param shap_values: SHAP values to export
        :param data: Data used to generate SHAP values
        :param feature_names: Names of features
        :return: HTML content string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHAP Explanations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .positive { color: red; }
                .negative { color: blue; }
            </style>
        </head>
        <body>
            <h1>SHAP Explanations</h1>
            <h2>Summary</h2>
            <p>Number of instances: {num_instances}</p>
            <p>Number of features: {num_features}</p>
            
            <h2>SHAP Values</h2>
            <table>
                <tr>
                    <th>Instance</th>
                    {feature_headers}
                </tr>
                {shap_rows}
            </table>
        </body>
        </html>
        """

        # Generate feature headers
        feature_headers = "".join([f"<th>{name}</th>" for name in feature_names])

        # Generate SHAP value rows
        shap_rows = ""
        for i in range(min(100, shap_values.shape[0])):  # Limit to first 100 instances
            row = f"<tr><td>{i}</td>"
            for j, value in enumerate(shap_values[i]):
                color_class = "positive" if value > 0 else "negative"
                row += f'<td class="{color_class}">{value:.4f}</td>'
            row += "</tr>"
            shap_rows += row

        return html_template.format(
            num_instances=shap_values.shape[0],
            num_features=shap_values.shape[1],
            feature_headers=feature_headers,
            shap_rows=shap_rows,
        )
