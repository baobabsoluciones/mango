import os
import pathlib
import tomllib as toml
import pkg_resources
import logging


def check_dependencies(dependencies_name):
    """
    Verify if optional dependencies have been installed for better ImportError handling.
    :param dependencies_name: optional dependencies name as defined in pyproject.toml
    :return: returns True if all dependencies are satisfied, False if not
    """

    # Extract data from pyproject.toml
    path_pyproject = pathlib.Path(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "pyproject.toml")
    )
    pyproject_text = path_pyproject.read_text()
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
    except pkg_resources.DistributionNotFound:
        logging.warning(
            f"{dependencies_name} dependencies not installed. Please install as follows:\n"
            f"pip install mango[{dependencies_name}]"
        )
        return False
