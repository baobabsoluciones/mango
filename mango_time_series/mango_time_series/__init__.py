"""Mango Time Series package."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mango-time-series")
except PackageNotFoundError:
    __version__ = "unknown"
