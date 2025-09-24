"""Mango Autoencoder package."""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mango-autoencoder")
except PackageNotFoundError:
    __version__ = "unknown"
