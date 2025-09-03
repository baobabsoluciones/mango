import logging
import os
import pathlib

from importlib.metadata import version, PackageNotFoundError

try:
    import tomllib as _toml
    _read = pathlib.Path.read_bytes
    _loads = _toml.loads
except ModuleNotFoundError:
    try:
        import tomli as _toml
        _read = pathlib.Path.read_bytes
        _loads = _toml.loads
    except ModuleNotFoundError:
        import toml as _toml
        _read = pathlib.Path.read_text
        _loads = _toml.loads


def check_dependencies(dependencies_name: str, pyproject_path: str = None):
    """
    Verify if optional dependencies have been installed for better ImportError handling.
    :param dependencies_name: optional dependencies name as defined in pyproject.toml
    :param pyproject_path: path to pyproject.toml file. If None, will look for it in the parent directory of this file.
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

    # Extract data
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

    # Check requirements with importlib.metadata
    try:
        for dependency in optional_dependencies:
            # Extract package name (remove version constraints)
            package_name = (
                dependency.split(">")[0]
                .split("<")[0]
                .split("=")[0]
                .split("[")[0]
                .strip()
            )
            version(package_name)
        return True
    except PackageNotFoundError:
        logging.warning(
            f"{dependencies_name} dependencies not installed. Please install as follows:\n"
            f"pip install mango[{dependencies_name}]"
        )
        return False
