"""
Module for mapping between different coding systems used by Spanish official sources.

This module provides the CatastroINEMapper class for mapping between the coding systems
used by the Spanish Cadastre (Catastro) and the Spanish National Statistics Institute (INE).

"""

import pandas as pd
import logging
from mango.clients.catastro import CatastroData
from mango.clients.ine import INEAPIClient
from typing import Optional
import requests
from io import BytesIO

logger = logging.getLogger(__name__)


class CatastroINEMapper:
    """
    Utility class to handle mapping between Catastro and INE municipality codes.

    This class integrates with both Catastro and INE modules to enrich municipality data.
    It processes raw relationship data obtained from:
    https://www.fega.gob.es/es/content/relacion-de-municipios-por-ccaa-con-equivalencias-entre-los-codigos-ine-y-catastro-2025.
    The mapping file url should be checked and updated regularly as the data may change.

    :param load_from_apis: Whether to load additional data from Catastro and INE APIs. (Can take long as it has to fetch the catastro index)
    :type load_from_apis: bool
    :param save_processed: Whether to save the processed mapping to a file.
    :type save_processed: bool
    :param catastro_client: Optional pre-initialized CatastroData client. (RECOMENDED, with cached index)
    :type catastro_client: CatastroData
    :param ine_client: Optional pre-initialized INEAPIClient client.
    :type ine_client: INEAPIClient
    :param processed_file: Path to a pre-processed mapping file (if provided, mapping_file is ignored).
    :type processed_file: str

    Usage
    --------
    >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper
    >>> from mango.clients.catastro import CatastroData
    >>> from mango.clients.ine import INEAPIClient

    Initialize the mapper with API data loading to get the accurate names from both INE and Catastro.
    It is recommended to pass the Catastro client with the cache_dir parameter set up to avoid downloading the data again. (Can take a while)

    >>> mapper = CatastroINEMapper(load_from_apis=True,
    ...                            ine_client=INEAPIClient(),
    ...                            catastro_client=CatastroData(cache=True, cache_file_path=r"catastro_cache.json", verbose=True))
    """

    PROCESSED_FILENAME = "processed_municipalities_mapping.csv"
    MAPPING_FILE_URL = "https://www.fega.gob.es/sites/default/files/files/document/RELACION_MUNICIPIOS_RUSECTOR_AGREGADO_ZONA_CAMPA%C3%91A_2025.xlsx"

    def __init__(
        self,
        load_from_apis: bool = False,
        save_processed: bool = False,
        catastro_client: Optional[CatastroData] = None,
        ine_client: Optional[INEAPIClient] = None,
        processed_file: Optional[str] = None,
    ) -> None:
        self.save_processed = save_processed
        self.load_from_apis = load_from_apis
        self.processed_file = processed_file or self.PROCESSED_FILENAME

        self.catastro_client = catastro_client

        self.ine_client = ine_client

        logger.info(f"Downloading mapping file from {self.MAPPING_FILE_URL}")
        self._load_mapping()

        if load_from_apis:
            self._enrich_with_api_data()

            if save_processed:
                self._save_processed_mapping()

    def _download_mapping_file(self) -> pd.DataFrame:
        """
        Download the mapping file from the predefined URL and load it into memory.

        This method downloads the mapping file and loads it.
        """
        try:
            response = requests.get(self.MAPPING_FILE_URL, stream=True)
            response.raise_for_status()
            logger.info("Mapping file downloaded successfully")
            return pd.read_excel(BytesIO(response.content), dtype=str)
        except Exception as e:
            logger.error(f"Failed to download the mapping file: {e}")
            raise

    def _load_mapping(self) -> None:
        """
        Download and process the raw mapping file.

        This method downloads the raw mapping file, processes it to generate municipality codes,
        and creates a mapping table with relevant columns.
        """
        raw_mapping = self._download_mapping_file()

        raw_mapping["ine_municipality_code"] = raw_mapping["Provincia"].str.zfill(
            2
        ) + raw_mapping["Municipio INE"].str.zfill(3)
        raw_mapping["catastro_municipality_code"] = raw_mapping["Provincia"].str.zfill(
            2
        ) + raw_mapping["Municipio"].str.zfill(3)
        raw_mapping["Municipio INE"] = raw_mapping["Municipio INE"].str.zfill(3)
        raw_mapping["Municipio"] = raw_mapping["Municipio"].str.zfill(3)

        self.mapping = raw_mapping[
            [
                "Provincia",
                "Nombre Provincia",
                "Municipio INE",
                "Municipio",
                "Nombre Municipio",
                "ine_municipality_code",
                "catastro_municipality_code",
            ]
        ][:-2]

        self.mapping.rename(
            {
                "catastro_municipality_name": "Municipio",
                "ine_municipality_name": "Municipio INE",
            },
            inplace=True,
        )

        self.mapping["ine_municipality_name"] = self.mapping["Nombre Municipio"]
        self.mapping["catastro_municipality_name"] = self.mapping["Nombre Municipio"]

        self._create_lookup_dictionaries()

    def _load_processed_mapping(self, filepath: str) -> None:
        """
        Load a pre-processed mapping file.

        :param filepath: Path to the pre-processed mapping file.
        :type filepath: str
        """
        self.mapping = pd.read_csv(filepath, dtype=str)
        self._create_lookup_dictionaries()

    def _create_lookup_dictionaries(self) -> None:
        """
        Create lookup dictionaries for quick access.

        This method generates dictionaries to map between INE and Catastro codes,
        as well as to retrieve municipality names based on codes.
        """
        self.ine_to_catastro = dict(
            zip(
                self.mapping["ine_municipality_code"],
                self.mapping["catastro_municipality_code"],
            )
        )
        self.catastro_to_ine = dict(
            zip(
                self.mapping["catastro_municipality_code"],
                self.mapping["ine_municipality_code"],
            )
        )
        self.ine_code_to_name = dict(
            zip(
                self.mapping["ine_municipality_code"],
                self.mapping["ine_municipality_name"],
            )
        )
        self.catastro_code_to_name = dict(
            zip(
                self.mapping["catastro_municipality_code"],
                self.mapping["catastro_municipality_name"],
            )
        )
        self.code_to_name = dict(
            zip(
                self.mapping["catastro_municipality_code"],
                self.mapping["Nombre Municipio"],
            )
        )

    def _enrich_with_api_data(self) -> None:
        """
        Enrich the mapping with names from Catastro and INE APIs.

        This method fetches additional municipality names from the Catastro and INE APIs
        and updates the mapping table accordingly.
        """
        logger.info("Enriching municipality data with names from Catastro and INE APIs")

        if self.load_from_apis:
            if not self.catastro_client:
                logger.info("Initializing Catastro client")
                self.catastro_client = CatastroData(verbose=False, cache=False)

            if not self.ine_client:
                logger.info("Initializing INE client")
                self.ine_client = INEAPIClient(verbose=False)

            self._enrich_with_catastro_names()

            self._enrich_with_ine_names()

    def _enrich_with_catastro_names(self) -> None:
        """
        Get municipality names from Catastro data.

        This method retrieves municipality names from the Catastro API and updates
        the mapping table with the fetched names.
        """
        if not self.catastro_client:
            logger.warning("Catastro client not initialized, skipping name enrichment")
            return

        logger.info("Fetching municipality names from Catastro")
        try:
            municipalities_df = self.catastro_client.available_municipalities(
                "Buildings"
            )

            catastro_names = dict(
                zip(
                    municipalities_df["catastro_municipality_code"].astype(str),
                    municipalities_df["catastro_municipality_name"],
                )
            )

            for idx, row in self.mapping.iterrows():
                code = row["catastro_municipality_code"]
                if code in catastro_names:
                    self.mapping.at[idx, "catastro_municipality_name"] = catastro_names[
                        code
                    ]

            self.catastro_code_to_name = dict(
                zip(
                    self.mapping["catastro_municipality_code"],
                    self.mapping["catastro_municipality_name"],
                )
            )
            logger.info(
                f"Updated {len(catastro_names)} municipality names from Catastro"
            )

        except Exception as e:
            logger.error(f"Error enriching with Catastro names: {e}")

    def _enrich_with_ine_names(self) -> None:
        """
        Get municipality names from INE data.

        This method retrieves municipality names from the INE API and updates
        the mapping table with the fetched names.
        """
        if not self.ine_client:
            logger.warning("INE client not initialized, skipping name enrichment")
            return

        logger.info("Fetching municipality names from INE")
        try:
            municipalities_df = self.ine_client.municipalities

            ine_names = dict(
                zip(
                    municipalities_df["ine_municipality_code"].astype(str),
                    municipalities_df["ine_municipality_name"],
                )
            )

            for idx, row in self.mapping.iterrows():
                code = row["ine_municipality_code"]
                if code in ine_names:
                    self.mapping.at[idx, "ine_municipality_name"] = ine_names[code]

            self.ine_code_to_name = dict(
                zip(
                    self.mapping["ine_municipality_code"],
                    self.mapping["ine_municipality_name"],
                )
            )
            logger.info(f"Updated {len(ine_names)} municipality names from INE")

        except Exception as e:
            logger.error(f"Error enriching with INE names: {e}")

    def _save_processed_mapping(self) -> None:
        """
        Save the processed mapping to a CSV file.

        This method saves the processed mapping table to a file for future use.
        """
        if self.save_processed:
            try:
                self.mapping.to_csv(self.processed_file, index=False)
                logger.info(f"Saved processed mapping to {self.processed_file}")
            except Exception as e:
                logger.error(f"Error saving processed mapping: {e}")

    def ine_to_catastro_code(self, ine_code: str) -> Optional[str]:
        """
        Convert INE code to Catastro code.

        :param ine_code: INE municipality code.
        :type ine_code: str
        :return: Corresponding Catastro municipality code, or None if not found.
        :rtype: str or None

        Usage
        --------

        Convert INE code to Catastro code

        >>> catastro_code = mapper.ine_to_catastro_code("25203")
        >>> print(catastro_code)
        """
        return self.ine_to_catastro.get(str(ine_code))

    def catastro_to_ine_code(self, catastro_code: str) -> Optional[str]:
        """
        Convert Catastro code to INE code.

        :param catastro_code: Catastro municipality code.
        :type catastro_code: str
        :return: Corresponding INE municipality code, or None if not found.
        :rtype: str or None

        Usage
        --------

        Convert Catastro code to INE code

        >>> ine_code = mapper.catastro_to_ine_code("25252")
        >>> print(ine_code)
        """
        return self.catastro_to_ine.get(str(catastro_code))

    def get_municipality_name(
        self, code: str, code_type: str = "ine", name_source: str = "ine"
    ) -> Optional[str]:
        """
        Get municipality name from the municipality code. The name returned comes from the source specified.

        :param code: Municipality code.
        :type code: str
        :param code_type: Type of code - 'catastro' or 'ine'.
        :type code_type: str
        :param name_source: Source of name - 'ine' or 'catastro'.
        :type name_source: str
        :return: Municipality name or None if not found.
        :rtype: str or None

        Usage
        --------

        Get municipality name from INE code.

        >>> municipality_name = mapper.get_municipality_name("25203", "ine")
        >>> print(municipality_name)

        Get municipality name from Catastro code
        >>> municipality_name = mapper.get_municipality_name("25252", "catastro")
        >>> print(municipality_name)
        """

        code = str(code)

        if name_source.lower() not in ["ine", "catastro"]:
            raise ValueError("name_source must be 'ine' or 'catastro'")

        if code_type.lower() == "ine":
            if name_source.lower() == "catastro":
                catastro_code = self.ine_to_catastro.get(code)
                return self.catastro_code_to_name.get(catastro_code)
            else:
                return self.ine_code_to_name.get(code)
        else:
            if name_source.lower() == "ine":
                ine_code = self.catastro_to_ine.get(code)
                return self.ine_code_to_name.get(ine_code)
            else:
                return self.catastro_code_to_name.get(code)

    def get_mapping_table(self) -> pd.DataFrame:
        """
        Return the full mapping table.

        :return: DataFrame containing the mapping table.
        :rtype: pandas.DataFrame

        Usage
        --------

        Get the full mapping table.

        >>> mapping_table = mapper.get_mapping_table()
        >>> print(mapping_table.head())
        """
        return self.mapping
