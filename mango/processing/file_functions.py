import json
import os
from os import listdir


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


def load_json(path):
    """
    The load_json function loads a json file from the specified path and returns it as a dictionary.

    :param path: Specify the path of the file to be loaded
    :return: A dictionary
    :doc-author: baobab soluciones
    """
    with open(path, "r") as f:
        return json.load(f)


def normalize_path(rel_path):
    """
    The normalize_path function is used to convert relative paths into absolute paths.
    The normalize_path function takes a relative path as an argument and returns the absolute path.

    :param str rel_path: Specify the path of the file that is being read
    :return: The absolute path of rel_path
    :doc-author: baobab soluciones
    """
    os.path.join(os.path.dirname(__file__), rel_path)
