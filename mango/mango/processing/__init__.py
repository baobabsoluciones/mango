from .date_functions import *
from .file_functions import (
    load_json,
    list_files_directory,
    is_json_file,
    is_excel_file,
    load_excel,
    load_excel_light,
    load_excel_sheet,
    load_csv,
    load_csv_light,
    write_json,
    write_excel,
    write_excel_light,
    write_csv,
    write_csv_light,
    is_excel_file,
    is_json_file,
)

try:
    from .data_imputer import DataImputer
except ImportError:

    class DataImputer:
        """
        Placeholder for DataImputer when dependencies are missing.

        :raises ImportError: When instantiated without required dependencies
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"""
                DataImputer requires 'sklearn' and 'polars'.
                Install them with `pip install mango[data]` to use this module.
                """
            )


from .object_functions import (
    pickle_copy,
    unique,
    reverse_dict,
    cumsum,
    lag_list,
    lead_list,
    row_number,
    flatten,
    df_to_list,
    df_to_dict,
    as_list,
)
