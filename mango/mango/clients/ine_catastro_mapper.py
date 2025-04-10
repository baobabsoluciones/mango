import pandas as pd
import logging
from catastro import CatastroData
from ine import INEAPIClient
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
    The mapping file should be updated regularly as the data changes.

    Attributes:
        PROCESSED_FILENAME (str): Default filename for the processed mapping file.
        MAPPING_FILE_URL (str): URL to download the mapping file if it does not exist locally.
    """

    PROCESSED_FILENAME = "processed_municipalities_mapping.csv"
    MAPPING_FILE_URL = "https://www.fega.gob.es/sites/default/files/files/document/RELACION_MUNICIPIOS_RUSECTOR_AGREGADO_ZONA_CAMPA%C3%91A_2025.xlsx"

    def __init__(
        self,
        load_from_apis: bool = True,
        save_processed: bool = True,
        catastro_client: Optional[CatastroData] = None,
        ine_client: Optional[INEAPIClient] = None,
        processed_file: Optional[str] = None,
    ) -> None:
        """
        Initialize the mapper with the mapping file and optionally enrich with data from APIs.

        :param load_from_apis: Whether to load additional data from Catastro and INE APIs.
        :type load_from_apis: bool
        :param save_processed: Whether to save the processed mapping to a file.
        :type save_processed: bool
        :param catastro_client: Optional pre-initialized CatastroData client.
        :type catastro_client: CatastroData
        :param ine_client: Optional pre-initialized INEAPIClient client.
        :type ine_client: INEAPIClient
        :param processed_file: Path to a pre-processed mapping file (if provided, mapping_file is ignored).
        :type processed_file: str
        """
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
        Load and process the raw mapping file.

        This method downloads the raw mapping file, processes it to generate municipality codes,
        and creates a mapping table with relevant columns.
        """
        raw_mapping = self._download_mapping_file()

        raw_mapping["Codigo Municipio INE"] = (raw_mapping["Provincia"].str.zfill(2) +
                                               raw_mapping["Municipio INE"].str.zfill(3))
        raw_mapping["Codigo Municipio Catastro"] = (raw_mapping["Provincia"].str.zfill(2) +
                                                    raw_mapping["Municipio"].str.zfill(3))

        self.mapping = raw_mapping[
            [
                "Provincia",
                "Nombre Provincia",
                "Municipio INE",
                "Municipio",
                "Nombre Municipio",
                "Codigo Municipio INE",
                "Codigo Municipio Catastro",
            ]
        ]

        self.mapping["Nombre INE"] = self.mapping["Nombre Municipio"]
        self.mapping["Nombre Catastro"] = self.mapping["Nombre Municipio"]

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
                self.mapping["Codigo Municipio INE"],
                self.mapping["Codigo Municipio Catastro"],
            )
        )
        self.catastro_to_ine = dict(
            zip(
                self.mapping["Codigo Municipio Catastro"],
                self.mapping["Codigo Municipio INE"],
            )
        )
        self.ine_code_to_name = dict(
            zip(self.mapping["Codigo Municipio INE"], self.mapping["Nombre INE"])
        )
        self.catastro_code_to_name = dict(
            zip(
                self.mapping["Codigo Municipio Catastro"],
                self.mapping["Nombre Catastro"],
            )
        )
        self.code_to_name = dict(
            zip(
                self.mapping["Codigo Municipio Catastro"],
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
                self.catastro_client = CatastroData(verbose=False, cache=True)
                self.catastro_client.load_index()

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
                    municipalities_df["Municipality Code"].astype(str),
                    municipalities_df["Municipality Name"],
                )
            )

            for idx, row in self.mapping.iterrows():
                code = row["Codigo Municipio Catastro"]
                if code in catastro_names:
                    self.mapping.at[idx, "Nombre Catastro"] = catastro_names[code]

            self.catastro_code_to_name = dict(
                zip(
                    self.mapping["Codigo Municipio Catastro"],
                    self.mapping["Nombre Catastro"],
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
                    municipalities_df["Codigo_INE"].astype(str),
                    municipalities_df["NOMBRE"],
                )
            )

            for idx, row in self.mapping.iterrows():
                code = row["Codigo Municipio INE"]
                if code in ine_names:
                    self.mapping.at[idx, "Nombre INE"] = ine_names[code]

            self.ine_code_to_name = dict(
                zip(self.mapping["Codigo Municipio INE"], self.mapping["Nombre INE"])
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
        """
        return self.ine_to_catastro.get(str(ine_code))

    def catastro_to_ine_code(self, catastro_code: str) -> Optional[str]:
        """
        Convert Catastro code to INE code.

        :param catastro_code: Catastro municipality code.
        :type catastro_code: str
        :return: Corresponding INE municipality code, or None if not found.
        :rtype: str or None
        """
        return self.catastro_to_ine.get(str(catastro_code))

    def get_municipality_name(
        self, code: str, code_type: str = "ine", name_source: str = "original"
    ) -> Optional[str]:
        """
        Get municipality name from code.

        :param code: Municipality code.
        :type code: str
        :param code_type: Type of code - 'catastro' or 'ine'.
        :type code_type: str
        :param name_source: Source of name - 'original', 'catastro', 'ine'.
        :type name_source: str
        :return: Municipality name or None if not found.
        :rtype: str or None
        """
        code = str(code)
        if code_type.lower() == "ine":
            if name_source.lower() == "catastro":
                catastro_code = self.ine_to_catastro.get(code)
                return self.catastro_code_to_name.get(catastro_code)
            elif name_source.lower() == "ine":
                return self.ine_code_to_name.get(code)
            else:
                catastro_code = self.ine_to_catastro.get(code)
                return self.code_to_name.get(catastro_code)
        else:
            if name_source.lower() == "ine":
                ine_code = self.catastro_to_ine.get(code)
                return self.ine_code_to_name.get(ine_code)
            elif name_source.lower() == "catastro":
                return self.catastro_code_to_name.get(code)
            else:
                return self.code_to_name.get(code)

    def get_mapping_table(self) -> pd.DataFrame:
        """
        Return the full mapping table.

        :return: DataFrame containing the mapping table.
        :rtype: pandas.DataFrame
        """
        return self.mapping

