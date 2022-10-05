from os import listdir


def list_files_directory(directory: str, extensions: list = None):
    """
    The list_files_directory function returns a list of files in the directory specified by the user.
    The function takes two arguments:
        1) The directory to search for files in (str).
        2) A list of file extensions to filter by (list). If no extensions are provided, all files will be returned.

    :param directory: str: Specify the directory that you want to list files from
    :param extensions: list: Specify the file extensions that should be included
    :return: A list of all filtered files in a directory
    """
    if extensions is None:
        extensions = ['.']
    return [fr'{directory}\{f}' for f in listdir(fr'{directory}')
            if any([f.__contains__(f'{ext}') for ext in extensions])]
