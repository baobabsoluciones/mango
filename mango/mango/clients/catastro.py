import geopandas as gpd
import pandas as pd
import requests
import os
import zipfile
from io import BytesIO
from time import sleep
from bs4 import BeautifulSoup
import json
from typing import Union, Optional, List
import re
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CatastroData:
    """
    Module for obtaining cadastral data from the Spanish Catastro.

    Requires an explicit call to `load_index()` after initialization
    before fetching specific municipality data.
    """

    BASE_URLS = {
        "Buildings": "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
        "CadastralParcels": "https://www.catastro.hacienda.gob.es/INSPIRE/CadastralParcels/ES.SDGC.CP.atom.xml",
        "Addresses": "https://www.catastro.hacienda.gob.es/INSPIRE/Addresses/ES.SDGC.AD.atom.xml",
    }

    CACHE_FILE = "catastro_cache.json"

    def __init__(self, debug=False, verbose=False, cache=False, request_timeout=30):
        """
        Initializes the CatastroData module.

        :param debug: If True, sets the logging level to DEBUG for detailed output. Overrides verbose.
        :type debug: bool
        :param verbose: If True and debug is False, sets the logging level to INFO for general information.
        :type verbose: bool
        :param cache: If True, loads/saves the municipality index from/to cache
        :type cache: bool
        :param request_timeout: Timeout in seconds for network requests
        :type request_timeout: int
        """
        self.debug = debug
        self.verbose = verbose
        self.cache = cache
        self.request_timeout = request_timeout
        self.municipalities_links = pd.DataFrame()
        self._index_loaded = False
        self.available_datatypes = list(self.BASE_URLS.keys())

        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled: Showing all log messages.")
        elif self.verbose:
            logger.setLevel(logging.INFO)
            logger.info("Verbose mode enabled: Showing informational messages.")
        else:
            logger.setLevel(logging.WARNING)
            logger.warning(
                "Neither debug nor verbose mode is enabled. Showing warnings, errors, and critical messages only."
            )

    def _fetch_content(self, url) -> Optional[bytes]:
        """
        Fetches content from a URL with error handling.

        :param url: URL to fetch content from
        :type url: str
        :return: Content bytes or None if fetch failed
        :rtype: bytes or None
        """
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            logger.debug(f"Successfully fetched content from: {url}")
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL: {url} - {e}")
            return None

    def _get_soup(self, url) -> Optional[BeautifulSoup]:
        """
        Fetches content and returns a BeautifulSoup object.

        :param url: URL to fetch content from
        :type url: str
        :return: BeautifulSoup object or None if fetch/parse failed
        :rtype: BeautifulSoup or None
        """
        content = self._fetch_content(url)
        if content:
            try:
                soup = BeautifulSoup(content, features="xml")
                logger.debug(f"Successfully parsed XML from: {url}")
                return soup
            except Exception as e:
                logger.error(f"Error parsing XML content from URL: {url} - {e}")
                return None
        return None

    def _fetch_and_parse_links(self) -> pd.DataFrame:
        """
        Fetches and parses territorial and municipality links from the ATOM feeds.

        :return: DataFrame with parsed municipality links
        :rtype: pd.DataFrame
        """
        logger.info(
            "Starting to fetch and parse municipality download links from ATOM feeds..."
        )
        municipalities_data = []

        for dataset, base_url in self.BASE_URLS.items():
            logger.info(f"Processing dataset: {dataset} - Fetching from {base_url}")
            base_soup = self._get_soup(base_url)
            if not base_soup:
                logger.warning(
                    f"Skipping dataset '{dataset}' due to an error fetching or parsing the base URL."
                )
                continue

            for terr_entry in base_soup.find_all("entry"):
                title_element = terr_entry.find("title")
                link_element = terr_entry.find("link", attrs={"href": True})

                if not (
                    title_element
                    and link_element
                    and len(title_element.get_text(strip=True)) >= 22
                ):
                    logger.debug(
                        f"Skipping an invalid territorial entry in dataset '{dataset}'."
                    )
                    continue

                terr_name_full = title_element.get_text(strip=True)
                territorial_link = link_element.get("href")
                try:
                    match = re.search(
                        r"Territorial office (\d{2}) (.*)", terr_name_full
                    )
                    if match:
                        territorial_code = match.group(1).zfill(2)
                        territorial_name = match.group(2).strip()
                    else:
                        territorial_code = str(terr_name_full[19:21]).zfill(2)
                        territorial_name = terr_name_full[22:].strip()

                    logger.debug(
                        f"Processing Territorial Office: Code={territorial_code}, Name='{territorial_name}'"
                    )

                except IndexError:
                    logger.warning(
                        f"Could not parse territorial code/name from title: '{terr_name_full}' in dataset '{dataset}'. Skipping."
                    )
                    continue

                terr_soup = self._get_soup(territorial_link)
                if not terr_soup:
                    logger.warning(
                        f"Skipping territorial office '{territorial_name}' due to an error fetching or parsing the territorial link."
                    )
                    continue

                for mun_entry in terr_soup.find_all("entry"):
                    link_tag = mun_entry.find(
                        "link", rel="enclosure", attrs={"href": True}
                    )
                    title_tag = mun_entry.find("title")

                    if not (link_tag and title_tag):
                        logger.debug(
                            "Skipping a municipality entry with missing link or title."
                        )
                        continue

                    zip_link = link_tag.get("href")
                    municipality_info = title_tag.get_text(strip=True)

                    try:
                        municipality_code = str(municipality_info[:5]).zfill(5)
                        municipality_name = municipality_info[6:].strip()

                        for word in ["buildings", "Cadastral Parcels", "addresses"]:
                            municipality_name = municipality_name.replace(
                                word, "", 1
                            ).strip()
                    except (IndexError, ValueError):
                        logger.warning(
                            f"Could not parse municipality code/name from title: '{municipality_info}'. Skipping."
                        )
                        continue

                    municipalities_data.append(
                        {
                            "Territorial Code": territorial_code,
                            "Territorial Name": territorial_name,
                            "Municipality Code": municipality_code,
                            "Municipality Name": municipality_name,
                            "Datatype": dataset,
                            "Zip Link": zip_link,
                        }
                    )
                    logger.debug(
                        f"Found municipality: Code={municipality_code}, Name='{municipality_name}' (Dataset: {dataset})"
                    )

        logger.info(
            f"Finished parsing municipality links. Found a total of {len(municipalities_data)} links."
        )
        return pd.DataFrame(
            municipalities_data,
            columns=[
                "Territorial Code",
                "Territorial Name",
                "Municipality Code",
                "Municipality Name",
                "Datatype",
                "Zip Link",
            ],
        )

    def load_index(self) -> None:
        """
        Loads the index of available municipalities and their download links.
        Uses cache if enabled and available, otherwise fetches fresh data.
        Must be called before retrieving specific municipality data.

        :return: None
        """
        if self._index_loaded:
            logger.info("Municipality index is already loaded.")
            return

        if self.cache and os.path.exists(self.CACHE_FILE):
            try:
                logger.info(
                    f"Attempting to load municipality index from cache: {self.CACHE_FILE}"
                )
                self.municipalities_links = pd.read_json(self.CACHE_FILE, dtype=False)
                self._index_loaded = True
                logger.info("Successfully loaded municipality index from cache.")
                return
            except (json.JSONDecodeError, ValueError, KeyError, FileNotFoundError) as e:
                logger.warning(
                    f"Failed to load municipality index from cache ({e}). Fetching fresh data."
                )
                if os.path.exists(self.CACHE_FILE):
                    try:
                        os.remove(self.CACHE_FILE)
                        logger.info(
                            f"Potentially corrupted cache file '{self.CACHE_FILE}' removed."
                        )
                    except OSError as remove_err:
                        logger.error(
                            f"Warning: Could not remove potentially corrupt cache file '{self.CACHE_FILE}': {remove_err}"
                        )

        logger.info("Fetching fresh municipality index from the Catastro website...")
        self.municipalities_links = self._fetch_and_parse_links()
        self._index_loaded = True

        if self.cache and not self.municipalities_links.empty:
            try:
                logger.info(
                    f"Saving the fetched municipality index to cache: {self.CACHE_FILE}"
                )
                self.municipalities_links.to_json(self.CACHE_FILE, indent=4)
                logger.info("Municipality index saved to cache successfully.")
            except IOError as e:
                logger.error(f"Error saving municipality index to cache: {e}")
        elif self.cache and self.municipalities_links.empty:
            logger.warning(
                "Skipping cache save because the fetched municipality index is empty."
            )

    def _ensure_index_loaded(self) -> None:
        """
        Checks if the index is loaded and raises an error if not.

        :raises RuntimeError: If municipality index is not loaded
        :return: None
        """
        if not self._index_loaded or self.municipalities_links.empty:
            raise RuntimeError(
                "Municipality index not loaded. Call load_index() first to populate the index."
            )

    def _download_and_extract(self, municipality_code, datatype, subtype=None):
        """
        Downloads and extracts the GML file from the zip archive.

        :param municipality_code: The 5-digit code of the municipality
        :type municipality_code: str
        :param datatype: The type of data ("Buildings", "CadastralParcels", "Addresses")
        :type datatype: str
        :param subtype: Optional subtype for the datatype (e.g., "Buildings", "Building_Parts")
        :type subtype: str
        :return: File-like object containing the extracted GML file
        :raises ValueError: If the municipality or datatype is not found
        :raises ConnectionError: If download fails
        :raises FileNotFoundError: If the expected file is not in the zip
        """
        self._ensure_index_loaded()

        logger.debug(
            f"Filtering for Municipality Code: {municipality_code}, Datatype: {datatype}"
        )

        municipality_code_str = str(municipality_code).zfill(5)

        filtered_links = self.municipalities_links[
            (self.municipalities_links["Datatype"] == datatype)
            & (self.municipalities_links["Municipality Code"] == municipality_code_str)
        ]

        if filtered_links.empty:
            raise ValueError(
                f"No index entry found for Municipality Code '{municipality_code_str}' and Datatype '{datatype}'. Please check the code and datatype."
            )

        zip_link = filtered_links.iloc[0]["Zip Link"]

        if not zip_link or pd.isna(zip_link):
            raise ValueError(
                f"Index entry found for municipality '{municipality_code_str}' and Datatype '{datatype}', but the Zip Link is invalid."
            )

        logger.info(
            f"Downloading {datatype} data for municipality {municipality_code_str} from {zip_link}..."
        )
        zip_content = self._fetch_content(zip_link)

        if not zip_content:
            raise ConnectionError(f"Failed to download zip file from {zip_link}")

        try:
            with zipfile.ZipFile(BytesIO(zip_content), "r") as zip_ref:
                file_suffixes = {
                    "Addresses": [".gml"],
                    "Buildings": {
                        "Buildings": ".building.gml",
                        "Building_Parts": ".buildingpart.gml",
                        "Other_Buildings": ".otherconstruction.gml",
                    },
                    "CadastralParcels": {
                        "CadastralParcels": ".cadastralparcel.gml",
                        "CadastralZonings": ".cadastralzoning.gml",
                    },
                }
                suffix = file_suffixes.get(datatype)
                if not suffix:
                    raise ValueError(
                        f"Internal error: Invalid datatype '{datatype}' specified for suffix lookup."
                    )

                if subtype:
                    if isinstance(suffix, dict):
                        if subtype not in suffix:
                            raise ValueError(
                                f"Invalid subtype '{subtype}' for datatype '{datatype}'. Available subtypes: {list(suffix.keys())}"
                            )
                        suffix = suffix[subtype]
                    else:
                        raise ValueError(
                            f"Subtypes are not supported for datatype '{datatype}'."
                        )
                else:
                    if isinstance(suffix, dict):
                        suffix = next(iter(suffix.values()))
                    elif isinstance(suffix, list):
                        suffix = suffix[0]
                    else:
                        raise ValueError(
                            f"Unexpected suffix type for datatype '{datatype}'."
                        )

                gml_filename = None
                for filename in zip_ref.namelist():
                    if filename.lower().endswith(suffix.lower()):
                        gml_filename = filename
                        break

                if gml_filename:
                    logger.debug(f"Extracting '{gml_filename}' from the zip archive.")
                    return zip_ref.open(gml_filename)
                else:
                    raise FileNotFoundError(
                        f"No file ending with the expected pattern ('{suffix}' or specific Address GML) "
                        f"found in the zip archive from {zip_link} for datatype {datatype}. "
                        f"Files found in archive: {zip_ref.namelist()}"
                    )
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                f"Downloaded file from {zip_link} appears to be an invalid zip archive."
            )
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred during zip file processing: {e}"
            )
            raise

    def available_municipalities(self, datatype: str) -> pd.DataFrame:
        """
        Returns a DataFrame of available municipalities for a given datatype.
        Requires ``load_index()`` to be called first.

        :param datatype: The type of data ("Buildings", "CadastralParcels", "Addresses")
        :type datatype: str
        :return: DataFrame with names and codes of all available municipalities
        :rtype: pd.DataFrame
        :raises ValueError: If an invalid datatype is specified
        :raises RuntimeError: If index is not loaded
        """
        self._ensure_index_loaded()
        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available types are: {self.available_datatypes}"
            )

        df = (
            self.municipalities_links[
                self.municipalities_links["Datatype"] == datatype
            ][["Municipality Code", "Municipality Name"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        logger.info(
            f"Retrieved a list of {len(df)} available municipalities for datatype '{datatype}'."
        )
        return df

    def _get_municipality_data(
        self, municipality_code, datatype, subtype=None
    ) -> gpd.GeoDataFrame:
        """
        Gets a GeoDataFrame for a single municipality and datatype.

        :param municipality_code: The 5-digit code of the municipality
        :type municipality_code: str
        :param datatype: The type of data ("Buildings", "CadastralParcels", "Addresses")
        :type datatype: str
        :param subtype: Optional subtype for the datatype (e.g., "Buildings", "Building_Parts")
        :type subtype: str
        :return: GeoDataFrame with the loaded spatial data
        :rtype: gpd.GeoDataFrame
        :raises ValueError: If the municipality or datatype is not found
        :raises RuntimeError: If index is not loaded
        """
        self._ensure_index_loaded()
        logger.info(
            f"Attempting to retrieve {datatype} data for municipality code '{municipality_code}'..."
        )

        municipality_code_str = str(municipality_code).zfill(5)

        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available types are: {self.available_datatypes}"
            )

        try:
            with self._download_and_extract(
                municipality_code_str, datatype, subtype
            ) as gml_file:
                # TODO: ADD SUPPORT FOR ALL LAYERS OF ADDRESS 'Address' (default), 'ThoroughfareName', 'PostalDescriptor', 'AdminUnitName'
                gdf = gpd.read_file(gml_file)
                logger.info(
                    f"Successfully loaded GeoDataFrame for municipality code '{municipality_code_str}' and datatype '{datatype}'. Found {len(gdf)} features."
                )
                return gdf
        except (
            ValueError,
            FileNotFoundError,
            ConnectionError,
            zipfile.BadZipFile,
            RuntimeError,
        ) as e:
            logger.error(
                f"Failed to get {datatype} data for municipality code '{municipality_code_str}': {e}"
            )
            raise
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred while loading GeoDataFrame for municipality code '{municipality_code_str}' and datatype '{datatype}': {e}"
            )
            raise

    def get_municipality_data(
        self,
        municipality_codes: Union[str, List[str]],
        datatype,
        subtype=None,
        target_crs="EPSG:4326",
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Returns a combined GeoDataFrame for multiple municipalities of the same datatype.
        Main method of the class. Requires ``load_index()`` to be called first.

        Optionally reprojects to the specified ``target_crs``, or defaults to EPSG:4326 for merging all municipalities
        into one GeoDataFrame.

        :param municipality_codes: A list of 5-digit municipality codes or a single code
        :type municipality_codes: Union[str, List[str]]
        :param datatype: The type of data ("Buildings", "CadastralParcels", "Addresses")
        :type datatype: str
        :param target_crs: The target Coordinate Reference System
        :type target_crs: str
        :param subtype: Optional subtype for the datatype (e.g., "Buildings", "Building_Parts")
        :type subtype: str
        :return: Combined data into a GeoDataFrame, or None if no data could be processed
        :rtype: Optional[gpd.GeoDataFrame]
        """
        self._ensure_index_loaded()

        if isinstance(municipality_codes, str):
            municipality_codes = [municipality_codes]
        elif not isinstance(municipality_codes, list):
            raise TypeError(
                "municipality_codes must be a list of strings or a single string."
            )

        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available types are: {self.available_datatypes}"
            )

        gdfs = []
        processed_codes = []
        failed_codes = {}

        for code in municipality_codes:
            sleep(0.1)
            code_str = str(code).zfill(5)
            logger.info(
                f"Processing municipality code: {code_str} for datatype '{datatype}'."
            )
            try:
                gdf = self._get_municipality_data(code_str, datatype, subtype)
                if gdf is not None and not gdf.empty:
                    if gdf.crs and gdf.crs != target_crs:
                        logger.info(
                            f"Reprojecting municipality {code_str} from {gdf.crs} to {target_crs}."
                        )
                        gdf = gdf.to_crs(target_crs)
                    elif not gdf.crs:
                        logger.warning(
                            f"Warning: CRS missing for municipality {code_str}. Assuming '{target_crs}' for concatenation."
                        )

                    gdf["municipality_code"] = code_str
                    gdfs.append(gdf)
                    processed_codes.append(code_str)
                    logger.debug(
                        f"Successfully processed municipality code: {code_str}."
                    )
                else:
                    logger.warning(
                        f"No data returned for municipality code '{code_str}' and datatype '{datatype}'."
                    )
                    failed_codes[code_str] = "No data returned"

            except Exception as e:
                logger.error(
                    f"Error processing municipality code '{code_str}' for datatype '{datatype}': {e}"
                )
                failed_codes[code_str] = str(e)
                continue

        if not gdfs:
            logger.warning("No data collected for any of the requested municipalities.")
            if failed_codes:
                logger.warning(
                    f"Processing failed for the following municipalities: {failed_codes}"
                )
            return None

        logger.info(
            f"Successfully processed data for {len(processed_codes)} municipalities."
        )
        if failed_codes:
            logger.warning(
                f"Processing failed for {len(failed_codes)} municipalities: {failed_codes}"
            )

        logger.info(f"Concatenating data for the processed municipalities.")
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs=target_crs if gdfs else None
        )
        logger.info(
            f"Combined GeoDataFrame created with a total of {len(combined_gdf)} features."
        )
        return combined_gdf