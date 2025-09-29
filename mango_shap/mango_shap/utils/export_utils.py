import json
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from mango_shap.logging import get_configured_logger


class ExportUtils:
    """
    Utility class for exporting SHAP explanations to various formats.

    Provides methods to export SHAP values and associated data to CSV, JSON,
    and HTML formats. Supports both structured data export for further analysis
    and human-readable formats for reporting and visualization.

    Example:
        >>> exporter = ExportUtils()
        >>> exporter.export(shap_values, data, feature_names, 'results', 'csv')
    """

    def __init__(self) -> None:
        """
        Initialize the export utilities.

        Sets up logging and prepares the exporter for SHAP explanation
        export operations. No parameters are required as the exporter is stateless.

        :return: None
        :rtype: None
        """
        self.logger = get_configured_logger()
        self.logger.info("ExportUtils initialized")

    def export(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        output_path: str = "shap_explanations",
        format: str = "csv",
    ) -> None:
        """
        Export SHAP explanations to file in specified format.

        Exports SHAP values and associated data to the specified output path
        in the chosen format. Supports CSV for tabular data, JSON for structured
        data, and HTML for human-readable reports.

        :param shap_values: SHAP values array to export
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param feature_names: Optional list of feature names
        :type feature_names: Optional[List[str]]
        :param output_path: Path where to save the exported file
        :type output_path: str
        :param format: Export format ('csv', 'json', 'html')
        :type format: str
        :return: None
        :rtype: None

        Example:
            >>> exporter.export(shap_values, data, feature_names, 'results', 'csv')
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
        Export SHAP values to CSV format with combined data.

        Creates a CSV file containing both the original data and SHAP values
        in a combined format. The original data columns are prefixed with 'data_'
        and SHAP values use the provided feature names.

        :param shap_values: SHAP values array to export
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param feature_names: Optional list of feature names for SHAP values
        :type feature_names: Optional[List[str]]
        :param output_path: Path object where to save the CSV file
        :type output_path: Path
        :return: None
        :rtype: None
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
        Export SHAP values to JSON format with metadata.

        Creates a JSON file containing SHAP values, original data, feature names,
        and metadata about the dataset. The JSON format is structured for easy
        programmatic access and analysis.

        :param shap_values: SHAP values array to export
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param feature_names: Optional list of feature names for SHAP values
        :type feature_names: Optional[List[str]]
        :param output_path: Path object where to save the JSON file
        :type output_path: Path
        :return: None
        :rtype: None
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        # Prepare data for JSON export
        export_data = {
            "shap_values": shap_values.tolist(),
            "feature_names": feature_names,
            "data": (
                data.tolist()
                if isinstance(data, np.ndarray)
                else data.to_dict("records")
            ),
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
        Export SHAP values to HTML format for human-readable reports.

        Creates an HTML file with a formatted table displaying SHAP values
        with color coding for positive (red) and negative (blue) values.
        Includes summary statistics and limits display to first 100 instances
        for readability.

        :param shap_values: SHAP values array to export
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param feature_names: Optional list of feature names for SHAP values
        :type feature_names: Optional[List[str]]
        :param output_path: Path object where to save the HTML file
        :type output_path: Path
        :return: None
        :rtype: None
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

    @staticmethod
    def _generate_html_content(
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> str:
        """
        Generate HTML content for SHAP explanations with styling.

        Creates a complete HTML document with embedded CSS styling for displaying
        SHAP values in a formatted table. Includes color coding for positive and
        negative values and limits the display to the first 100 instances.

        :param shap_values: SHAP values array to display
        :type shap_values: np.ndarray
        :param feature_names: List of feature names for table headers
        :type feature_names: List[str]
        :return: Complete HTML document as string
        :rtype: str
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
