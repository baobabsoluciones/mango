"""Mango Calendar package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mango-calendar")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
