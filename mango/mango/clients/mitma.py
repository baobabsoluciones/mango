"""
MITMA Data Processing Module

This module provides functions to read, process, and compute some statistics for MITMA data.

It requires the already downloaded monthly data files as an input in a folder.

The data can be downloded from https://www.transportes.gob.es/ministerio/proyectos-singulares/estudios-de-movilidad-con-big-data/opendata-movilidad/

estudios_basicos -> por-distritos -> pernoctaciones -> meses-completos
"""

import pandas as pd
import os
import tarfile
from typing import Union, List, Optional, TextIO, BinaryIO, Literal


def load_mitma_data(input_path: str) -> pd.DataFrame:
    """
    Loads and consolidates MITMA mobility data by recursively scanning a specified directory.

    This function searches the `input_path`, including all its subdirectories,
    to locate and process relevant data files. It supports:

    * **Directly:** CSV files, whether plain (`.csv`) or gzipped (`.csv.gz`, `.gz`).
    * **Within TAR archives:** Standard TAR archives (e.g., `.tar`), gzipped TAR archives
        (`.tar.gz`, `.tgz`), or bzipped2 TAR archives (`.tar.bz2`, `.tbz2`) that contain
        the CSV file types mentioned above.

    All CSV files that are successfully parsed according to the expected format
    (see below for details on format) are then concatenated into a single
    pandas DataFrame.

    :param input_path: Path to the directory containing the MITMA data files.
    :type input_path: str
    :return: A pandas DataFrame containing the combined data from all files.
    :rtype: pd.DataFrame

    Usage
    -----
    >>> # Process MITMA data
    >>> from mango.clients.mitma import load_mitma_data
    >>> data = load_mitma_data("path/to/mitma/data")

    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not os.path.isdir(input_path):
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    dataframes = []

    for root, _, files in os.walk(input_path):
        for name in files:
            path = os.path.join(root, name)
            if _is_tar_file(name):
                dataframes.extend(_read_tar_file(path))
            elif _is_data_file(name):
                df = _read_csv_file(path)
                if df is not None:
                    dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def _is_data_file(filename: str) -> bool:
    return filename.endswith((".csv", ".csv.gz", ".gz"))


def _is_tar_file(filename: str) -> bool:
    return filename.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2"))


def _read_csv_file(file: Union[str, TextIO, BinaryIO]) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(
            file,
            compression="gzip",
            parse_dates=["fecha"],
            date_format="%Y%m%d",
            dtype={"zona_residencia": str, "zona_pernoctacion": str, "personas": float},
            sep="|",
        )
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print(f"File: {file}")
        return None


def _read_tar_file(tar_path: str) -> List[pd.DataFrame]:
    dataframes = []
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile() and _is_data_file(member.name):
                    f = tar.extractfile(member)
                    if f:
                        df = _read_csv_file(f)
                        if df is not None:
                            dataframes.append(df)
    except:
        pass
    return dataframes


def groupby_zona_pernoctacion(
    data: pd.DataFrame, group_by: Literal["district", "municipality"] = "district"
) -> pd.DataFrame:
    """
    Group the data by 'zona_pernoctacion' and compute the sum of 'personas'.

    :param data: DataFrame containing the data to group.
    :type data: pd.DataFrame
    :param group_by: Method to group by. Can be 'district' or 'municipality'.
    :type group_by: str
    :return: DataFrame containing the grouped data.
    :rtype: pd.DataFrame

    Usage
    -----
    >>> # Process MITMA data
    >>> from mango.clients.mitma import load_mitma_data
    >>> data = load_mitma_data("path/to/mitma/data")
    >>> # Group by zona_pernoctacion to get the sum of personas each day in each district
    >>> grouped_data = groupby_zona_pernoctacion(data, group_by="district")

    """
    if group_by not in ["district", "municipality"]:
        raise ValueError("group_by must be 'district' or 'municipality'")

    if group_by == "municipality":
        data["municipality"] = data["zona_pernoctacion"].str[:5]
        grouped_data = (
            data.groupby(["fecha", "municipality"])["personas"].sum().reset_index()
        )
    else:
        grouped_data = (
            data.groupby(["fecha", "zona_pernoctacion"])["personas"].sum().reset_index()
        )
    return grouped_data


def compute_multipliers(
    data: pd.DataFrame, by: Literal["month"] = "month"
) -> pd.DataFrame:
    """
    Compute multipliers for each district/municipality compared to January. It supports only datasets with data for all 12 months of a single year.

    :param data: DataFrame containing the data to compute multipliers for.
    :type data: pd.DataFrame
    :param by: Method to group by. Currently, supports 'month'.
    :type by: str
    :return: DataFrame containing the computed multipliers.
    :rtype: pd.DataFrame
    """
    if by != "month":
        raise ValueError("by must be 'month'")

    if (
        "fecha" not in data.columns
        or "zona_pernoctacion" not in data.columns
        or "personas" not in data.columns
    ):
        raise KeyError(
            "Data must contain 'fecha', 'zona_pernoctacion', and 'personas' columns."
        )

    try:
        data["month"] = data["fecha"].dt.month
        data["year"] = data["fecha"].dt.year

        # check if the data contains records from only one year
        unique_years = data["year"].nunique()
        if unique_years != 1:
            raise ValueError("Data must contain records from only one year.")

        # check if all 12 months are present
        months_in_data = data["month"].nunique()
        if months_in_data != 12:
            raise ValueError("Data must contain records for all 12 months.")

        data = data.groupby(["month", "zona_pernoctacion"], as_index=False).mean(
            numeric_only=True
        )

        base_data = data[data["month"] == 1][["zona_pernoctacion", "personas"]].rename(
            columns={"personas": "base_personas"}
        )
        merged_data = data.merge(base_data, on="zona_pernoctacion", how="left")
        merged_data["multiplier"] = (
            merged_data["personas"] / merged_data["base_personas"]
        )
    except Exception as e:
        raise

    return merged_data
