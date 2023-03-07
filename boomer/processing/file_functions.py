import json
from os import listdir
from typing import Union

import pandas as pd


def list_files_directory(directory: str, extensions: list = None):
    """
    The list_files_directory function returns a list of files in the directory specified by the user.
    The function takes two arguments:
        1) The directory to search for files in (str).
        2) A list of file extensions to filter by (list). If no extensions are provided, all files will be returned.

    :param str directory: Specify the directory that you want to list files from
    :param list extensions: Specify the file extensions that should be included
    :return: A list of all filtered files in a directory
    :raises WindowsError: if the directory doesn't exist
    """
    if extensions is None:
        extensions = ["."]
    return [
        rf"{directory}\{f}"
        for f in listdir(rf"{directory}")
        if any([f.__contains__(f"{ext}") for ext in extensions])
    ]


def check_extension(path: str, extension: str):
    """
    The check_extension function checks if a file has the specified extension.

    :param path: Specify the path of the file to be checked
    :param extension: Specify the extension to check against
    :return: A boolean
    :doc-author: baobab soluciones
    """
    return path.endswith(extension)


def is_excel_file(path: str):
    """
    The is_excel_file function checks if a file is an Excel file.

    :param path: Specify the path of the file to be checked
    :return: A boolean
    :doc-author: baobab soluciones
    """
    return (
        check_extension(path, ".xlsx")
        or check_extension(path, ".xls")
        or check_extension(path, ".xlsm")
    )


def is_json_file(path: str):
    """
    The is_json_file function checks if a file is a JSON file.

    :param path: Specify the path of the file to be checked
    :return: A boolean
    :doc-author: baobab soluciones
    """
    return check_extension(path, ".json")


def load_json(path: str, **kwargs):
    """
    The load_json function loads a json file from the specified path and returns it as a dictionary.

    :param path: Specify the path of the file to be loaded
    :return: A dictionary
    :doc-author: baobab soluciones
    """
    with open(path, "r", **kwargs) as f:
        return json.load(f)


def write_json(data: Union[dict, list], path):
    """
    The write_json function writes a dictionary or list to a JSON file.

    :param data: allow the function to accept both a dictionary and list object
    :param path: Specify the path of the file that you want to write to
    :return: None
    :doc-author: baobab soluciones
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)


def load_excel_sheet(path: str, sheet: str, **kwargs):
    """
    The load_excel_sheet function loads a sheet from an Excel file and returns it as a DataFrame.

    :param path: Specify the path of the file to be loaded
    :param sheet: Specify the name of the sheet to be loaded
    :return: A DataFrame
    :doc-author: baobab soluciones
    """
    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    return pd.read_excel(path, sheet_name=sheet, **kwargs)


def load_excel(path):
    """
    The load_excel function loads an Excel file and returns it as a dictionary of DataFrames.

    :param path: Specify the path of the file to be loaded
    :return: A dictionary of DataFrames
    :doc-author: baobab soluciones
    """
    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )
    return pd.read_excel(path, sheet_name=None)


def write_excel(path, data):
    """
    The write_excel function writes a dictionary of DataFrames to an Excel file.

    :param path: Specify the path of the file that you want to write to
    :param data: Specify the dictionary to be written
    :return: None
    :doc-author: baobab soluciones
    """
    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    with pd.ExcelWriter(path) as writer:
        for sheet_name, content in data.items():
            if isinstance(content, list):
                df = pd.DataFrame.from_records(content)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif isinstance(content, dict):
                df = pd.DataFrame.from_dict(content, orient="index")
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            adjust_excel_col_width(writer, df, sheet_name)


def write_df_to_excel(path, data, **kwargs):
    """
    The write_df_to_excel function writes a DataFrame to an Excel file.

    :param path: Specify the path of the file that you want to write to
    :param data: Specify the DataFrame to be written
    :return: None
    :doc-author: baobab soluciones
    """
    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    with pd.ExcelWriter(path) as writer:
        data.to_excel(writer, **kwargs, index=False)


def load_csv(path, **kwargs):
    """
    The load_csv function loads a CSV file and returns it as a DataFrame.

    :param path: Specify the path of the file to be loaded
    :return: A DataFrame
    :doc-author: baobab soluciones
    """
    if not check_extension(path, ".csv"):
        raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    return pd.read_csv(path, **kwargs)


def write_csv(path, data, **kwargs):
    """
    The write_csv function writes a DataFrame to a CSV file.

    :param path: Specify the path of the file that you want to write to
    :param data: Specify the DataFrame to be written
    :return: None
    :doc-author: baobab soluciones
    """
    if not check_extension(path, ".csv"):
        raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    if isinstance(data, list):
        df = pd.DataFrame.from_records(data)
        df.to_csv(path, **kwargs, index=False)
    elif isinstance(data, dict):
        df = pd.DataFrame.from_dict(data)
        df.to_csv(path, **kwargs, index=False)


def write_df_to_csv(path, data, **kwargs):
    """
    The write_csv function writes a DataFrame to a CSV file.

    :param path: Specify the path of the file that you want to write to
    :param data: Specify the DataFrame to be written
    :return: None
    :doc-author: baobab soluciones
    """
    if not check_extension(path, ".csv"):
        raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    data.to_csv(path, **kwargs, index=False)


def adjust_excel_col_width(writer, df: pd.DataFrame, table_name: str, min_len: int = 7):
    """
    Adjusts the width of the column on the Excel file

    :param writer:
    :param :class:`pandas.DataFrame` df:
    :param str table_name:
    :param int min_len:
    """
    for column in df:
        content_len = df[column].astype(str).map(len).max()
        column_width = max(content_len, len(str(column)), min_len) + 2
        col_idx = df.columns.get_loc(column)
        writer.sheets[table_name].set_column(col_idx, col_idx, column_width)
