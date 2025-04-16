import pandas as pd
import os
from typing import Literal, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def read_file_as_df(file_path: str, file_name: str) -> pd.DataFrame:
    """
    Read a file from a given path and return it as a DataFrame.

    :param file_path: Path to the file to read.
    :param file_name: Name of the file to read.
    :return: DataFrame with the data from the file.
    :raises FileNotFoundError: If the file does not exist.
    :raises TypeError: If the file type is not supported.
    """
    file = os.path.join(file_path, file_name)

    if not os.path.exists(file):
        logging.error(f"File not found: {file}")
        raise FileNotFoundError(f"File not found: {file}")

    try:
        if file.endswith(".tar"):
            result = pd.read_csv(file, compression="tar")
        elif file.endswith(".gz"):
            result = pd.read_csv(file, compression="gzip", sep="|")
        elif file.endswith(".csv"):
            result = pd.read_csv(file)
        else:
            logging.error(f"Unsupported file type: {file}")
            raise TypeError(f"File type not supported: {file}")
    except Exception as e:
        logging.error(f"Error reading file {file}: {e}")
        raise

    logging.info(f"Successfully read file: {file}")
    return result

def process_mitma_data(path: str, group_by: Literal["district", "municipality"] = "district") -> pd.DataFrame:
    """
    Process the MITMA data from the given path.

    :param path: Path to the directory containing the MITMA data files.
    :param group_by: Grouping method. Can be 'district' or 'municipality'.
    :return: DataFrame containing the processed data.
    :raises ValueError: If the group_by parameter is invalid.
    """
    if group_by not in ["district", "municipality"]:
        logging.error("Invalid group_by value. Must be 'district' or 'municipality'.")
        raise ValueError("group_by must be 'district' or 'municipality'")

    if not os.path.isdir(path):
        logging.error(f"Invalid directory path: {path}")
        raise FileNotFoundError(f"Directory not found: {path}")

    df = pd.DataFrame()
    for month in os.listdir(path):
        month_path = os.path.join(path, month)
        if not os.path.isdir(month_path):
            logging.warning(f"Skipping non-directory: {month_path}")
            continue

        for filename in os.listdir(month_path):
            try:
                logging.info(f"Processing file: {filename}")
                mitma_data = read_file_as_df(file_path=month_path, file_name=filename)
                df = pd.concat([df, mitma_data], ignore_index=True)
            except Exception as e:
                logging.error(f"Failed to process file {filename}: {e}")
                continue

    if df.empty:
        logging.warning("No data was processed. Returning an empty DataFrame.")
        return df

    try:
        if group_by == "municipality":
            df['municipality'] = df['zona_pernoctacion'].str[:5]
            df = df.groupby('municipality', as_index=False).sum(numeric_only=True)
        elif group_by == "district":
            df = df.groupby('zona_pernoctacion', as_index=False).sum(numeric_only=True)
    except KeyError as e:
        logging.error(f"Missing required column in data: {e}")
        raise

    logging.info("Data processing completed successfully.")
    return df

def compute_multipliers(data: pd.DataFrame, by: Literal["month"] = "month") -> pd.DataFrame:
    """
    Compute multipliers for each district/municipality compared to January.

    :param data: DataFrame containing the data to compute multipliers for.
    :param by: Method to group by. Currently supports 'month'.
    :return: DataFrame containing the computed multipliers.
    :raises ValueError: If the 'by' parameter is invalid.
    """
    if by != "month":
        logging.error("Invalid 'by' value. Must be 'month'.")
        raise ValueError("by must be 'month'")

    if 'fecha' not in data.columns or 'zona_pernoctacion' not in data.columns or 'personas' not in data.columns:
        logging.error("Data is missing required columns: 'fecha', 'zona_pernoctacion', 'personas'")
        raise KeyError("Data must contain 'fecha', 'zona_pernoctacion', and 'personas' columns.")

    try:
        data['month'] = pd.to_datetime(data['fecha'], format="%Y%m%d").dt.month
        data = data.groupby(['month', 'zona_pernoctacion'], as_index=False).mean(numeric_only=True)

        base_data = data[data['month'] == 1][['zona_pernoctacion', 'personas']].rename(columns={'personas': 'base_personas'})
        merged_data = data.merge(base_data, on='zona_pernoctacion', how='left')
        merged_data['multiplier'] = merged_data['personas'] / merged_data['base_personas']
    except Exception as e:
        logging.error(f"Error computing multipliers: {e}")
        raise

    logging.info("Multipliers computed successfully.")
    return merged_data
