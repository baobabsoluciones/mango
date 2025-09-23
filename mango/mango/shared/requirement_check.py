import os
import pathlib
from importlib.metadata import version, PackageNotFoundError

from mango.logging import get_configured_logger

try:
    import tomllib as toml
except ModuleNotFoundError:
    try:
        import tomli as toml
    except ModuleNotFoundError:
        import toml

log = get_configured_logger(__name__)


def check_dependencies(dependencies_name: str, pyproject_path: str = None):
    """
    Verify if optional dependencies have been installed for better ImportError handling.

    Checks whether all dependencies specified in a named optional dependency group
    in pyproject.toml are currently installed. This function provides better error
    handling for missing optional dependencies by giving clear installation instructions.

    :param dependencies_name: Name of the optional dependencies group as defined in pyproject.toml
    :type dependencies_name: str
    :param pyproject_path: Path to pyproject.toml file. If None, searches in parent directory
    :type pyproject_path: str, optional
    :return: True if all dependencies are satisfied, False if any are missing
    :rtype: bool
    :raises FileNotFoundError: If pyproject.toml file cannot be found
    :raises KeyError: If the specified dependency group name is not found in pyproject.toml

    Example:
        >>> # Check if visualization dependencies are installed
        >>> check_dependencies("visualization")
        True
        >>>
        >>> # Check with custom pyproject.toml path
        >>> check_dependencies("ml", "/path/to/custom/pyproject.toml")
        False
    """

    # Handle path
    if not pyproject_path:
        pyproject_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "pyproject.toml"
        )
    pyproject = pathlib.Path(pyproject_path)
    if not pyproject.exists():
        log.error(f"pyproject.toml file not found at: {pyproject}")
        raise FileNotFoundError(
            f"{pyproject} does not exist. You may pass the path to the function as an argument"
        )

    # Extract data
    log.debug(f"Reading pyproject.toml from: {pyproject}")
    pyproject_text = pyproject.read_text()
    pyproject_data = toml.loads(pyproject_text)

    # Try-except to handle invalid dependencies_name
    try:
        optional_dependencies = pyproject_data["project"]["optional-dependencies"][
            dependencies_name
        ]
        log.debug(
            f"Found {len(optional_dependencies)} dependencies in group '{dependencies_name}'"
        )
    except KeyError as e:
        log.error(f"Dependency group '{dependencies_name}' not found in pyproject.toml")
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
            log.debug(f"Package '{package_name}' is installed")
        log.info(f"All dependencies in group '{dependencies_name}' are satisfied")
        return True
    except PackageNotFoundError as e:
        log.warning(
            f"{dependencies_name} dependencies not installed. Please install as follows:\n"
            f"pip install mango[{dependencies_name}]"
        )
        return False
