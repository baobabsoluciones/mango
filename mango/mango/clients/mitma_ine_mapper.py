"""
Module for mapping the INE codes (municipalities, sections) with the MITMA codes (districts, municipalities)

This module contains the mapping based on the available file on the MITMA website.
It processes the data and allows for an easy conversion between the INE municipalities and the MITMA districts.

Mapping file origin https://movilidad-opendata.mitma.es/zonificacion/relacion_ine_zonificacionMitma.csv

Example usage:

>>> from mango.clients.mitma_ine_mapper import MitmaIneMapper
>>> mapper = MitmaIneMapper()

Get the mitma district code from the INE section code
>>> distrito = mapper.seccion_ine_to_distrito_mitma("100601001")
>>> print(f"MITMA district for INE section 100601001: {distrito}")

Get the INE municipality code from the MITMA municipality code
>>> ine_muni = mapper.municipio_mitma_to_municipio_ine("01047_AM")
>>> print(f"INE municipality for MITMA municipality code 01047_AM: {ine_muni}")
"""

import pandas as pd
import requests
from io import StringIO


MAPPING_PATH = "https://movilidad-opendata.mitma.es/zonificacion/relacion_ine_zonificacionMitma.csv"


def _download_csv_text(url: str) -> str:
    """
    Downloads a CSV file from the specified URL and returns the text content.

    :param url: URL of the CSV file to download.
    :type url: str
    :return: Text content of the CSV file.
    :rtype: str
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def _read_csv_from_text(csv_text: str) -> pd.DataFrame:
    """
    Reads CSV data from a string and returns a pandas DataFrame.

    :param csv_text: CSV data as a string.
    :type csv_text: str
    :return: DataFrame containing the CSV data.
    :rtype: pd.DataFrame
    """
    csv_io = StringIO(csv_text)
    return pd.read_csv(csv_io, sep="|", header=0, dtype=str)

def _get_csv_dataframe(url: str) -> pd.DataFrame:
    """
    Combines downloading and reading a CSV into a pandas DataFrame.

    :param url: URL of the CSV file to fetch and read.
    :type url: str
    :return: DataFrame containing the CSV data.
    :rtype: pd.DataFrame
    """
    csv_text = _download_csv_text(url)
    return _read_csv_from_text(csv_text)

def _validate_code(code: str) -> str:
    """
    Validates that the provided code is a string.

    :param code: Code to validate.
    :type code: str
    :return: The same code if valid.
    :rtype: str
    :raises TypeError: If code is not a string.
    """
    if not isinstance(code, str):
        raise TypeError(f"Code must be a string, got {type(code).__name__} instead.")
    return code

class MitmaIneMapper:
    """
    Class for mapping the INE municipalities with the MITMA districts. When instantiated, it reads the mapping file from the MITMA website and stores it in a DataFrame.
    """

    def __init__(self):
        """
        Initialize the MitmaIneMapper class.
        """
        self.mapping_df = _get_csv_dataframe(MAPPING_PATH)
        self._seccion_ine_to_distrito_mitma = dict(zip(self.mapping_df['seccion_ine'], self.mapping_df['distrito_mitma']))
        self._distrito_mitma_to_seccion_ine = dict(zip(self.mapping_df['distrito_mitma'], self.mapping_df['seccion_ine']))

        self._municipio_ine_to_municipio_mitma = dict(zip(self.mapping_df['municipio_ine'], self.mapping_df['municipio_mitma']))
        self._municipio_mitma_to_municipio_ine = dict(zip(self.mapping_df['municipio_mitma'], self.mapping_df['municipio_ine']))

        self._seccion_ine_to_municipio_mitma = dict(zip(self.mapping_df['seccion_ine'], self.mapping_df['municipio_mitma']))
        self._municipio_mitma_to_seccion_ine = dict(zip(self.mapping_df['municipio_mitma'], self.mapping_df['seccion_ine']))

        self._distrito_mitma_to_municipio_ine = dict(zip(self.mapping_df['distrito_mitma'], self.mapping_df['municipio_ine']))
        self._municipio_ine_to_distrito_mitma = dict(zip(self.mapping_df['municipio_ine'], self.mapping_df['distrito_mitma']))


    def get_mapping(self) -> pd.DataFrame:
        """
        Get the mapping DataFrame.

        :return: DataFrame containing the mapping between INE municipalities and MITMA districts.
        :rtype: pd.DataFrame
        """
        return self.mapping_df


    def seccion_ine_to_distrito_mitma(self, code: str) -> str | None:
        """
        Get the MITMA district code from the INE section code.
        :param code: INE section code.
        :type code: str
        :return: MITMA district code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._seccion_ine_to_distrito_mitma.get(code)

    def distrito_mitma_to_seccion_ine(self, code: str) -> str | None:
        """
        Get the INE section code from the MITMA district code.
        :param code: MITMA district code.
        :type code: str
        :return: INE section code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._distrito_mitma_to_seccion_ine.get(code)

    def municipio_ine_to_municipio_mitma(self, code: str) -> str | None:
        """
        Get the MITMA municipality code from the INE municipality code.
        :param code: INE municipality code.
        :type code: str
        :return: MITMA municipality code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._municipio_ine_to_municipio_mitma.get(code)

    def municipio_mitma_to_municipio_ine(self, code: str) -> str | None:
        """
        Get the INE municipality code from the MITMA municipality code.
        :param code: MITMA municipality code.
        :type code: str
        :return: INE municipality code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._municipio_mitma_to_municipio_ine.get(code)

    def seccion_ine_to_municipio_mitma(self, code: str) -> str | None:
        """
        Get the MITMA municipality code from the INE section code.
        :param code: INE section code.
        :type code: str
        :return: MITMA municipality code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._seccion_ine_to_municipio_mitma.get(code)

    def municipio_mitma_to_seccion_ine(self, code: str) -> str | None:
        """
        Get the INE section code from the MITMA municipality code.
        :param code: MITMA municipality code.
        :type code: str
        :return: INE section code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._municipio_mitma_to_seccion_ine.get(code)

    def distrito_mitma_to_municipio_ine(self, code: str) -> str | None:
        """
        Get the INE municipality code from the MITMA district code.
        :param code: MITMA district code.
        :type code: str
        :return: INE municipality code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._distrito_mitma_to_municipio_ine.get(code)

    def municipio_ine_to_distrito_mitma(self, code: str) -> str | None:
        """
        Get the MITMA district code from the INE municipality code.
        :param code: INE municipality code.
        :type code: str
        :return: MITMA district code or None if not found.
        :rtype: str | None
        """
        code = _validate_code(code)
        return self._municipio_ine_to_distrito_mitma.get(code)