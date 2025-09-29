"""Mango SHAP package."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mango-shap")
except PackageNotFoundError:
    __version__ = "unknown"
