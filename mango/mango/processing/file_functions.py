import ast
import csv
import json
import warnings
from os import listdir
from typing import Union, Literal, Iterable

import openpyxl as xl
from openpyxl.utils import get_column_letter
from pytups import TupList


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
    # TODO implement open version
    try:
        import pandas as pd
    except ImportError as e:
        raise NotImplementedError("function not yet implemented without pandas")

    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    return pd.read_excel(path, sheet_name=sheet, **kwargs)


def load_excel(
    path,
    dtype="object",
    output: Literal[
        "df", "dict", "list", "series", "split", "tight", "records", "index"
    ] = "df",
    sheet_name=None,
    **kwargs,
):
    """
    The load_excel function loads an Excel file and returns it as a dictionary of DataFrames.

    :param path: Specify the path of the file to be loaded.
    :param dtype: pandas parameter dtype. Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32}.
     Use object to preserve data as stored in Excel and not interpret dtype.
    :param output: data output type. Default is "df" for a dict of pandas dataframes.
    Other options are the orient argument for transforming the dataframe with to_dict. (list of dict is "records").
    :param sheet_name: sheet name to read, if None, read all sheets.
    :param kwargs: other parameters to pass pandas read_excel.
    :return: A dictionary of DataFrames
    :doc-author: baobab soluciones
    """
    try:
        import pandas as pd
    except ImportError:
        warnings.warn(
            "pandas is not installed so load_excel_open will be used. Data can only be returned as list of dicts."
        )
        return load_excel_light(path, sheets=sheet_name)

    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    data = pd.read_excel(path, sheet_name=sheet_name, dtype=dtype, **kwargs)
    if output == "df":
        return data
    else:
        return {k: v.to_dict(orient=output) for k, v in data.items()}


def write_excel(path, data):
    """
    The write_excel function writes a dictionary of DataFrames to an Excel file.

    :param path: Specify the path of the file that you want to write to
    :param data: Specify the dictionary to be written
    :return: None
    :doc-author: baobab soluciones
    """
    try:
        import pandas as pd
    except ImportError:
        warnings.warn("pandas is not installed so write_excel_open will be used.")
        return write_excel_light(path, data)

    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    with pd.ExcelWriter(path) as writer:
        for sheet_name, content in data.items():
            if isinstance(content, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif isinstance(content, list):
                df = pd.DataFrame.from_records(content)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif isinstance(content, dict):
                df = pd.DataFrame.from_dict(content, orient="columns")
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
            adjust_excel_col_width(writer, df, sheet_name)


def load_csv(path, **kwargs):
    """
    The load_csv function loads a CSV file and returns it as a DataFrame.

    :param path: Specify the path of the file to be loaded
    :return: A DataFrame
    :doc-author: baobab soluciones
    """
    try:
        import pandas as pd
    except ImportError as e:
        warnings.warn("pandas is not installed so load_csv_light will be used.")
        return load_csv_light(path)

    if not check_extension(path, ".csv"):
        raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    return pd.read_csv(path, **kwargs)


def load_csv_light(path, sep=None, encoding=None):
    """
    Read csv data from path using csv library.

    :param path: path of the csv file.
    :param sep: column separator in the csv file. (detected automatically if None).
    :param encoding: encoding.
    :return: data as a list of dict.
    """
    # if not check_extension(path, ".csv"):
    #     raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    with open(path, encoding=encoding) as f:
        try:
            dialect = csv.Sniffer().sniff(f.read(1000), delimiters=sep)
        except Exception as e:
            if sep is not None:
                dialect = get_default_dialect(sep, csv.QUOTE_NONNUMERIC)
            else:
                raise ValueError(f"Error in load_csv_light; {e}")
        dialect.quoting = csv.QUOTE_NONNUMERIC
        f.seek(0)
        if sep is None:
            sep = dialect.delimiter
        else:
            dialect.delimiter = sep
        headers = f.readline().split("\n")[0].split(sep)
        try:
            reader = csv.DictReader(f, dialect=dialect, fieldnames=headers)
            data = [row for row in reader]
        except ValueError:
            print("csv loading failed, trying other quote option")
            dialect.quoting = csv.QUOTE_MINIMAL
            reader = csv.DictReader(f, dialect=dialect, fieldnames=headers)
            data = [row for row in reader]

    return data


def get_default_dialect(sep, quoting):
    """
    Get a default dialect for csv reading and writing.

    :param sep: separator
    :return: dialect
    """

    class dialect(csv.Dialect):
        _name = "default"
        lineterminator = "\r\n"

    dialect.quoting = quoting
    dialect.doublequote = True
    dialect.delimiter = sep
    dialect.quotechar = '"'
    dialect.skipinitialspace = False
    return dialect


def write_csv(path, data, **kwargs):
    """
    The write_csv function writes a DataFrame to a CSV file.

    :param path: Specify the path of the file that you want to write to
    :param data: Specify the DataFrame to be written
    :return: None
    :doc-author: baobab soluciones
    """
    try:
        import pandas as pd
    except ImportError as e:
        warnings.warn("pandas is not installed so write_csv_light will be used.")
        return write_csv_light(path, data)

    if not check_extension(path, ".csv"):
        raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    if isinstance(data, list):
        df = pd.DataFrame.from_records(data)
        df.to_csv(path, **kwargs, index=False)
    elif isinstance(data, dict):
        df = pd.DataFrame.from_dict(data)
        df.to_csv(path, **kwargs, index=False)


def write_csv_light(path, data, sep=None, encoding=None):
    """
    Write data to csv using csv library.

    :param path: path of the csv file
    :param data: data as list of dict
    :param sep: separator of the csv file
    :param encoding: encoding
    :return: None
    """
    if not check_extension(path, ".csv"):
        raise FileNotFoundError(f"File {path} is not a CSV file (.csv).")

    if sep is None:
        sep = ","

    dialect = get_default_dialect(sep, csv.QUOTE_NONE)
    with open(path, "w", newline="", encoding=encoding) as f:
        headers = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=headers, dialect=dialect)
        writer.writerow({h: h for h in headers})
        writer.writerows(data)


def adjust_excel_col_width(writer, df, table_name: str, min_len: int = 7):
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


def load_excel_light(path, sheets=None):
    """
    The load_excel function loads an Excel file and returns it as a dictionary of DataFrames.
    It doesn't use pandas.

    :param path: Specify the path of the file to be loaded.
    :param sheets: list of sheets to read. If None, all sheets will be read.
    :return: A dictionary of TupLists
    :doc-author: baobab soluciones
    """
    if not is_excel_file(path):
        raise FileNotFoundError(
            f"File {path} is not an Excel file (.xlsx, .xls, .xlsm)."
        )

    wb = xl.load_workbook(path, read_only=True, keep_vba=False)

    if sheets is None:
        dict_sheets = {ws.title: ws.values for ws in wb}
    else:
        dict_sheets = {ws.title: ws.values for ws in wb if ws.title in sheets}

    dataset = {}
    for name, v in dict_sheets.items():
        data = [row for row in v]
        if len(data):
            dataset[name] = (
                TupList(data[1:])
                .to_dictlist(data[0])
                .vapply(lambda v: {k: load_str_iterable(w) for k, w in v.items()})
            )
        else:
            dataset[name] = []
    wb.close()
    return dataset


def load_str_iterable(v):
    """
    Evaluate the value of an Excel cell and return strings representing python iterables as such.
    Return other strings and other types unchanged.

    :param v: content of an Excel cell
    :return: value of the cell
    """
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError):
            return v
    else:
        return v


def write_excel_light(path, data):
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
    wb = xl.Workbook()
    if len(data):
        wb.remove(wb.active)

    for sheet_name, content in data.items():
        wb.create_sheet(sheet_name)
        if isinstance(content, list):
            ws = wb[sheet_name]
            if len(content):
                ws.append([k for k in content[0].keys()])
                for row in content:
                    ws.append([write_iterables_as_str(v) for v in row.values()])

                tab = get_default_table_style(sheet_name, content)

                ws.add_table(tab)
                adjust_excel_col_width_2(ws)

    wb.save(path)
    wb.close()
    return None


def write_iterables_as_str(v):
    """
    An iterable in an Excel cell should be written as a string.

    :param v: cell content
    :return: cell value
    """
    if isinstance(v, Iterable):
        return str(v)
    else:
        return v


def get_default_table_style(sheet_name, content):
    """
    Get a default style for the table

    :param sheet_name: name of the sheet
    :param content: content of the sheet.
    :return: table object
    """
    from openpyxl.worksheet.table import Table, TableStyleInfo

    #  TODO: look at interesting styles
    style = TableStyleInfo(
        name="style_1",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=False,
        showColumnStripes=False,
    )
    tab = Table(
        displayName=sheet_name,
        ref=f"A1:{get_column_letter(len(content[0]))}{len(content) + 1}",
    )
    tab.tableStyleInfo = style
    return tab


def adjust_excel_col_width_2(ws, min_width=10, max_width=30):
    """
    Adjust the column width of a worksheet.

    :param ws: worksheet object
    :param min_width: minimum width to use.
    :param max_width: maximum width to use
    :return: None
    """
    for k, v in get_column_widths(ws).items():
        ws.column_dimensions[k].width = min(max(v, min_width), max_width)
    return None


def get_column_widths(ws):
    """
    Get the maximum width of the columns of a worksheet.

    :param ws: worksheet object
    :return: a dict with the letter of the columns and their maximum width (ex: {"A":15, "B":12})
    """
    result = {}
    for column_cells in ws.columns:
        width = max(len(str(cell.value)) for cell in column_cells)
        letter = get_column_letter(column_cells[0].column)
        result[letter] = width * 1.2
    return result
