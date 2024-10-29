import os
import pathlib

# Compatibility with Python < 3.11
try:
    import tomllib as toml
except ImportError:
    from pip._vendor import tomli as toml

# Trick for testing suite mockup
import pkg_resources
from pkg_resources import DistributionNotFound

import logging


def check_dependencies(dependencies_name: str, pyproject_path: str = None):
    """
    Verify if optional dependencies have been installed for better ImportError handling.
    :param dependencies_name: optional dependencies name as defined in pyproject.toml
    :return: returns True if all dependencies are satisfied, False if not
    """

    # Handle path
    if not pyproject_path:
        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "pyproject.toml"
        )
    pyproject = pathlib.Path(pyproject_path)
    if not pyproject.exists():
        raise FileNotFoundError(
            f"{pyproject} does not exist. You may pass the path to the function as an argument"
        )

    # Extact data
    pyproject_text = pyproject.read_text()
    pyproject_data = toml.loads(pyproject_text)

    # Try-except to handle invalid dependencies_name
    try:
        optional_dependencies = pyproject_data["project"]["optional-dependencies"][
            dependencies_name
        ]
    except KeyError as e:
        raise KeyError(
            f"Could not find {dependencies_name} as an optional dependency in pyproject.toml"
        )

    # Check requirements with pkg_resources
    try:
        pkg_resources.require(optional_dependencies)
        return True
    except DistributionNotFound:
        logging.warning(
            f"{dependencies_name} dependencies not installed. Please install as follows:\n"
            f"pip install mango[{dependencies_name}]"
        )
        return False
