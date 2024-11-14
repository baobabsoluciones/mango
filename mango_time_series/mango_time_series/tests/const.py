import os

path_to_tests_dir = os.path.dirname(os.path.abspath(__file__))


def normalize_path(rel_path):
    """
    The normalize_path function is used to convert relative paths into absolute paths.
    The normalize_path function takes a relative path as an argument and returns the absolute path.

    :param str rel_path: Specify the path of the file that is being read
    :return: The absolute path of rel_path
    :doc-author: baobab soluciones
    """
    return os.path.join(path_to_tests_dir, rel_path)
