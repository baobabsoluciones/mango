"""
Module for collecting cadastral data from the Spanish Catastro.

This module provides the CatastroData class for retrieving and processing
cadastral data from the Spanish Catastro API.

"""

import logging
import os
import re
import zipfile
from datetime import datetime
from io import BytesIO
from time import sleep
from typing import Union, Optional, List, Dict

import feedparser
import geopandas as gpd
import pandas as pd
import requests

from mango.logging import get_configured_logger

# configure logging level
logger = get_configured_logger(
    logger_type=__name__,
    log_console_level=logging.WARNING,
    mango_color=True,
)


class CatastroData:
    """
    A class for retrieving, processing, and linking cadastral data from the Spanish Catastro API.

    This class provides methods to fetch cadastral data for municipalities, process it into GeoDataFrames,
    and link related datasets such as addresses and buildings. It supports caching for faster later
    data retrieval and allows customization of request intervals and timeouts.

    :param verbose: Enables verbose logging for debugging purposes if set to True.
    :type verbose: bool
    :param cache: Enables caching of the municipality index to a file if set to True.
    :type cache: bool
    :param request_timeout: Timeout for HTTP requests in seconds. Default is 30 seconds.
    :type request_timeout: float
    :param request_interval: Interval between consecutive HTTP requests in seconds. Default is 0.1 seconds.
    :type request_interval: float
    :param cache_file_path: Path to the cache file for storing the municipality index.
    :type cache_file_path: str

    Usage
    --------

    Initialize the CatastroData with caching enabled or disabled if it's the first time initializing:

    >>> from mango.clients.catastro import CatastroData
    >>> catastro = CatastroData(cache=True, request_interval=0.1,
    ...                        cache_file_path="catastro_cache.json")

    """

    BASE_URLS = {
        "Buildings": "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml",
        "CadastralParcels": "https://www.catastro.hacienda.gob.es/INSPIRE/CadastralParcels/ES.SDGC.CP.atom.xml",
        "Addresses": "https://www.catastro.hacienda.gob.es/INSPIRE/Addresses/ES.SDGC.AD.atom.xml",
    }
    MUN_NAME_CLEANUP_WORDS = ["buildings", "Cadastral Parcels", "addresses"]
    TERRITORIAL_TITLE_REGEX = re.compile(r"Territorial office (\d{2}) (.*)")
    FILE_SUFFIXES = {
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

    def __init__(
        self,
        verbose: bool = False,
        cache: bool = False,
        request_timeout: float = 30,
        request_interval: float = 0.1,
        cache_file_path: str = "catastro_cache.json",
    ):
        self.cache = cache
        self.request_timeout = request_timeout
        self.request_interval = request_interval
        self.cache_file = cache_file_path
        self.municipalities_links = pd.DataFrame()
        self._index_loaded = False
        self.available_datatypes = list(self.BASE_URLS.keys())

        if verbose:
            logger.setLevel(logging.DEBUG)

        logger.debug(
            f"CatastroData initializing. Log level: {logging.getLevelName(logger.level)}"
        )

        # load from cache if available or parse fresh data
        loaded_from_cache = False
        if self.cache and os.path.exists(self.cache_file):
            try:
                logger.info(f"Attempting to load index from cache: {self.cache_file}")
                # we specify dtype to ensure codes are read as strings and not integers so they keep their leading zeros
                self.municipalities_links = pd.read_json(
                    self.cache_file,
                    dtype={
                        "catastro_municipality_code": str,
                        "territorial_office_code": str,
                    },
                )
                if not self.municipalities_links.empty:
                    self._index_loaded = True
                    loaded_from_cache = True
                    logger.info("Successfully loaded index from cache.")
                else:
                    logger.warning(
                        "Cache file loaded but resulted in an empty index. Will fetch fresh data."
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to load index from cache ({e}). Will fetch fresh data."
                )
                if os.path.exists(self.cache_file):
                    try:
                        os.remove(self.cache_file)
                        logger.info(
                            f"Removed potentially corrupt cache file '{self.cache_file}'."
                        )
                    except OSError as remove_err:
                        logger.error(
                            f"Could not remove cache file '{self.cache_file}': {remove_err}"
                        )

        if not self._index_loaded:
            logger.info("Fetching fresh municipality index...")
            self.municipalities_links = self._fetch_and_parse_links()
            self._index_loaded = not self.municipalities_links.empty

            if not self._index_loaded:
                logger.error(
                    "Failed to fetch fresh municipality index. CatastroData may not function correctly."
                )
            elif self.cache and not loaded_from_cache:
                # save only if cache enabled and we didn't load from it initially
                try:
                    logger.info(f"Saving fetched index to cache: {self.cache_file}")
                    self.municipalities_links.to_json(self.cache_file, indent=4)
                    logger.info("Index saved to cache.")
                except IOError as e:
                    logger.error(f"Error saving index to cache: {e}")
            elif self.cache and loaded_from_cache:
                # This case should not happen due to logic flow, but adding for completeness
                logger.debug("Index was loaded from cache, no need to save.")

    def _fetch_feed(self, url: str) -> Optional[feedparser.FeedParserDict]:
        """
        Fetches and parses an Atom feed from the given URL.

        :param url: URL of the Atom feed to fetch.
        :return: Parsed feed or None if the feed is empty or an error occurs.
        """
        try:
            feed = feedparser.parse(url)
            if not feed.entries and not feed.feed:
                logger.warning(
                    f"Feed from {url} is empty or could not be fetched properly."
                )
                return None
            return feed
        except Exception as e:
            logger.error(f"Unexpected error fetching feed from {url}: {e}")
            return None

    def _parse_territorial_entry(
            self,
            entry: feedparser.FeedParserDict,
    ) -> Optional[Dict[str, str]]:
        """
        Parses a territorial entry from the feed.

        :param entry: The territorial feed entry to parse.
        :type entry: feedparser.FeedParserDict
        :return: Dictionary with territorial code, name, and link or None if parsing fails.
        """
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        if not title or not link:
            logger.warning(
                f"Missing title or link in territorial entry. Title: '{title}', Link: '{link}'. Skipping."
            )
            return None

        match = self.TERRITORIAL_TITLE_REGEX.search(title)
        if match:
            code = match.group(1)
            name = match.group(2).strip()
            logger.debug(f"Parsed Territorial Office: Code={code}, Name='{name}'")
            return {"territorial_code": code, "territorial_name": name, "link": link}
        else:
            logger.warning(
                f"Could not parse territorial code/name from title: '{title}'. Skipping."
            )
            return None

    def _parse_municipality_entry(
            self,
            entry: feedparser.FeedParserDict,
    ) -> Optional[Dict[str, str]]:
        """
        Parses a municipality entry from the feed.

        :param entry: The municipality feed entry to parse.
        :type entry: feedparser.FeedParserDict
        :return: Dictionary with municipality code, name, and link or None if parsing fails.
        """
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        if not link or not title:
            logger.warning(
                f"Missing title or link in municipality entry. Title: '{title}', Link: '{link}'. Skipping."
            )
            return None

        try:
            parts = title.split("-", 1)
            if len(parts) != 2:
                logger.warning(
                    f"Unknown title structure, expected code-name but got: '{title}'. Skipping."
                )
                return None

            code = parts[0].zfill(5)
            name = parts[1]
            for word in self.MUN_NAME_CLEANUP_WORDS:
                name = name.replace(word, "")
            name = name.strip()
            dataset = parts[1].split(" ")[-1]

        except (IndexError, ValueError) as e:
            logger.warning(
                f"Could not parse municipality code/name from title: '{title}'. Error: {e}. Skipping."
            )
            return None

        logger.debug(
            f"Parsed Municipality in {dataset} dataset: Code={code}, Name='{name}'"
        )
        return {"municipality_code": code, "municipality_name": name, "link": link}

    def _fetch_and_parse_links(self) -> pd.DataFrame:
        """
        Fetches and parses the municipality download links from the Catastro feeds.
        :return: DataFrame with municipality codes, names, and download links.
        """
        logger.info("Fetching and parsing municipality download links...")
        all_data = []
        for dataset, base_url in self.BASE_URLS.items():
            logger.info(f"Processing dataset: {dataset} from {base_url}")
            dataset_feed = self._fetch_feed(base_url)
            if not dataset_feed:
                continue

            for terr_entry in dataset_feed.entries:
                territorial_info = self._parse_territorial_entry(terr_entry)
                if not territorial_info:
                    continue

                terr_feed = self._fetch_feed(territorial_info["link"])
                if not terr_feed:
                    continue

                for mun_entry in terr_feed.entries:
                    municipality_info = self._parse_municipality_entry(mun_entry)
                    if not municipality_info:
                        continue

                    all_data.append(
                        {
                            "territorial_office_code": territorial_info[
                                "territorial_code"
                            ],
                            "territorial_office_name": territorial_info[
                                "territorial_name"
                            ],
                            "catastro_municipality_code": municipality_info[
                                "municipality_code"
                            ],
                            "catastro_municipality_name": municipality_info[
                                "municipality_name"
                            ],
                            "datatype": dataset,
                            "zip_link": municipality_info["link"],
                        }
                    )
        if not all_data:
            logger.warning("Parsing finished, but no links were found.")
            return pd.DataFrame()

        logger.info(f"Finished parsing. Found {len(all_data)} links.")
        columns = [
            "territorial_office_code",
            "territorial_office_name",
            "catastro_municipality_code",
            "catastro_municipality_name",
            "datatype",
            "zip_link",
        ]
        return pd.DataFrame(all_data, columns=columns)

    def _extract_gml_from_zip(
            self, zip_content: bytes, datatype: str, subtype: Optional[str], zip_url: str
    ) -> Optional[BytesIO]:
        """
        Extracts the GML file from a zip archive content based on the datatype and subtype.

        :param zip_content: Downloaded zip file content from _download_zip_content().
        :param datatype: The type of data to extract (e.g., "Buildings", "CadastralParcels").
        :param subtype: The subtype of data to extract (e.g., "Buildings", "Building_Parts").
        :param zip_url: The URL from which the zip file was downloaded.
        :return: BytesIO object containing the GML file content or None if not found.
        """
        try:
            with zipfile.ZipFile(BytesIO(zip_content), "r") as zip_ref:
                suffix_config = self.FILE_SUFFIXES.get(datatype)
                if not suffix_config:
                    raise ValueError(
                        f"Internal error: Invalid datatype '{datatype}' for suffix lookup."
                    )

                target_suffix = ""
                if isinstance(suffix_config, dict):
                    if subtype:
                        if subtype not in suffix_config:
                            raise ValueError(
                                f"Invalid subtype '{subtype}' for '{datatype}'. Available: {list(suffix_config.keys())}"
                            )
                        target_suffix = suffix_config[subtype]
                    else:
                        target_suffix = next(iter(suffix_config.values()))
                elif isinstance(suffix_config, list):
                    target_suffix = suffix_config[0]
                else:
                    raise TypeError(
                        f"Unexpected suffix config for datatype '{datatype}'."
                    )

                gml_filename = None
                for filename in zip_ref.namelist():
                    if filename.lower().endswith(target_suffix.lower()):
                        gml_filename = filename
                        break

                if gml_filename:
                    logger.debug(f"Extracting '{gml_filename}' from archive.")
                    return BytesIO(zip_ref.read(gml_filename))
                else:
                    raise FileNotFoundError(
                        f"No file with suffix '{target_suffix}' found in zip from {zip_url}. Files: {zip_ref.namelist()}"
                    )

        except zipfile.BadZipFile:
            logger.error(f"Downloaded file from {zip_url} is not a valid zip archive.")
            raise
        except Exception as e:
            logger.error(f"Error processing zip file from {zip_url}: {e}")
            raise

    def extract_zip_to_folder(self, zip_content: bytes, output_folder: str) -> bool:
        """
        Extracts zip file content to the specified output folder.
        :param zip_content: Bytes of the zip file.
        :param output_folder: Path to the output directory.
        :return: True if extraction succeeds, False otherwise.
        """
        try:
            os.makedirs(output_folder, exist_ok=True)
            with zipfile.ZipFile(BytesIO(zip_content)) as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.lower().endswith(".gml"):
                        zip_ref.extract(zip_info, output_folder)
                        extracted_any = True
                        logger.debug(
                            f"Extracted {zip_info.filename} to {output_folder}"
                        )

            logger.debug(f"Extracted zip content to folder: {output_folder}")
            return True
        except zipfile.BadZipFile:
            logger.error("Failed to extract: Bad zip file format.")
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
        return False

    def _download_zip_content(self, url: str) -> Optional[bytes]:
        """
        Downloads the zip file from the given URL and returns its content.
        :param url: URL of the zip file to download.
        :return: Content of the zip file as bytes or None if download fails.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.debug(f"Successfully downloaded content from: {url}")
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading from URL: {url} - {e}")
            return None

    def _download_and_extract_zip(self, url: str, output_folder: str) -> bool:
        """
        Combines downloading and extracting a zip file from a URL.
        :param url: URL of the zip file.
        :param output_folder: Directory where files should be extracted.
        :return: True if successful, False otherwise.
        """
        zip_content = self._download_zip_content(url)
        if zip_content is None:
            logger.error("No zip content downloaded.")
            return False
        return self.extract_zip_to_folder(zip_content, output_folder)

    def _ensure_index_loaded(self) -> None:
        """
        Ensures the municipality index is loaded. Raises an error if not.
        :return: None
        """
        if not self._index_loaded:
            raise RuntimeError(
                "Municipality index could not be loaded during initialization. Check logs for errors."
            )

    def _find_link(self, municipality_code: str, datatype: str) -> Optional[str]:
        """
        Finds the download link for a given municipality code and datatype.

        :param municipality_code: Municipality code to search for.
        :param datatype: Datatype to search for (e.g., "Buildings", "CadastralParcels").
        :return: Download link or None if not found.
        """
        filtered = self.municipalities_links[
            (self.municipalities_links["datatype"] == datatype)
            & (
                self.municipalities_links["catastro_municipality_code"]
                == municipality_code
            )
        ]
        if filtered.empty:
            return None
        link = filtered.iloc[0]["zip_link"]
        return link if pd.notna(link) else None

    def _get_single_municipality_gdf(
        self, municipality_code: str, datatype: str, subtype: Optional[str]
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Retrieves a single municipality's GeoDataFrame for the specified datatype and subtype.
        :param municipality_code: Municipality code to retrieve data for.
        :param datatype: Datatype to retrieve (e.g., "Buildings", "CadastralParcels").
        :param subtype: Optional subtype for the datatype (e.g., "Buildings", "Building_Parts").
        :return: GeoDataFrame containing the data or None if not found.
        """
        logger.info(
            f"Retrieving {datatype} ({subtype or 'default'}) for municipality {municipality_code}"
        )
        zip_link = self._find_link(municipality_code, datatype)
        if not zip_link:
            raise ValueError(
                f"No index entry found for Municipality {municipality_code}, Datatype {datatype}."
            )

        zip_content = self._download_zip_content(zip_link)
        if not zip_content:
            raise ConnectionError(f"Failed to download zip file from {zip_link}")

        try:
            gml_file_buffer = self._extract_gml_from_zip(
                zip_content, datatype, subtype, zip_link
            )
            if gml_file_buffer:
                # Ensure the buffer is reset before reading otherwise it can give problems
                gml_file_buffer.seek(0)
                # Adresses files seem to have several layers, the important one is "Address" however, there's "ThoroughfareName" that contains the details about the street name and the codification which can be linked to the "Address" layer by the gml_id
                if datatype == "Addresses":
                    gdf = gpd.read_file(gml_file_buffer, layer="Address")
                else:
                    gdf = gpd.read_file(gml_file_buffer)
                logger.info(
                    f"Loaded GDF for {municipality_code} - {datatype}. Found {len(gdf)} features."
                )
                return gdf
            else:
                return None
        except (FileNotFoundError, zipfile.BadZipFile, ValueError) as e:
            logger.error(
                f"Failed to extract/load GML for {municipality_code} - {datatype}: {e}"
            )
            raise
        except Exception as e:
            # Catch potential geopandas/fiona read errors
            logger.critical(
                f"Unexpected error loading GDF for {municipality_code} - {datatype}: {e}"
            )
            raise

    def available_municipalities(self, datatype: str) -> pd.DataFrame:
        """
        Retrieves a list of municipalities available for a specific datatype.

        This method filters the municipality index for the specified datatype and returns a DataFrame
        containing municipality codes and names.

        :param datatype: The type of data to filter by (e.g., "Buildings", "CadastralParcels").
        :type datatype: str
        :return: A DataFrame with municipality codes and names for the specified datatype.
        :rtype: pd.DataFrame
        """
        self._ensure_index_loaded()
        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available: {self.available_datatypes}"
            )

        df = (
            self.municipalities_links[
                self.municipalities_links["datatype"] == datatype
            ][["catastro_municipality_code", "catastro_municipality_name"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        logger.info(f"Found {len(df)} municipalities for datatype '{datatype}'.")
        return df

    def get_data(
        self,
        municipality_codes: Union[str, List[str]],
        datatype: str,
        subtype: Optional[str] = None,
        target_crs: Optional[str] = "EPSG:4326",
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Fetches cadastral data for one or more municipalities and returns it as a GeoDataFrame.

        This method retrieves data for the specified municipality codes and datatype, optionally reprojecting
        the data to a target CRS. It supports fetching data for multiple municipalities and concatenates the
        results into a single GeoDataFrame.

        :param municipality_codes: A single municipality code or a list of municipality codes to retrieve data for.
        :type municipality_codes: Union[str, List[str]]
        :param datatype: The type of data to retrieve (e.g., "Buildings", "CadastralParcels").
        :type datatype: str
        :param subtype: Optional subtype for the datatype (e.g., "Buildings", "Building_Parts").
        :type subtype: Optional[str]
        :param target_crs: The target coordinate reference system (CRS) for the GeoDataFrame. Default is "EPSG:4326".
        :type target_crs: Optional[str]
        :return: A GeoDataFrame containing the requested data or None if no data is found.
        :rtype: Optional[gpd.GeoDataFrame]

        Usage
        --------
        Get addresses data for a specific municipality:

        >>> from mango.clients.catastro import CatastroData
        >>> catastro = CatastroData()
        >>> addresses_data = catastro.get_data("28900", "Addresses")

        Get buildings data for the same municipality:

        >>> buildings_data = catastro.get_data("28900", "Buildings")
        """
        self._ensure_index_loaded()

        if isinstance(municipality_codes, str):
            codes_list = [municipality_codes]
        elif isinstance(municipality_codes, list):
            codes_list = municipality_codes
        else:
            raise TypeError(
                "municipality_codes must be a list of strings or a single string."
            )

        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available: {self.available_datatypes}"
            )

        gdfs = []
        processed_codes = []
        failed_codes = {}

        for code in codes_list:
            sleep(self.request_interval)
            code_str = str(code).zfill(5)
            logger.info(f"Processing {code_str} - {datatype} ({subtype or 'default'})")
            try:
                gdf = self._get_single_municipality_gdf(code_str, datatype, subtype)
                if gdf is not None and not gdf.empty:
                    initial_crs = gdf.crs
                    if target_crs and initial_crs != target_crs:
                        logger.info(
                            f"Reprojecting {code_str} from {initial_crs} to {target_crs}"
                        )
                        gdf = gdf.to_crs(target_crs)
                    elif not initial_crs and target_crs:
                        logger.warning(
                            f"CRS missing for {code_str}. Assuming '{target_crs}' for concatenation."
                        )
                        gdf.crs = target_crs

                    gdf["catastro_municipality_code"] = code_str
                    gdfs.append(gdf)
                    processed_codes.append(code_str)
                elif gdf is not None and gdf.empty:
                    logger.warning(f"No features found for {code_str} - {datatype}.")
                    processed_codes.append(code_str)
                else:
                    failed_codes[code_str] = "Data retrieval failed (check logs)"

            except Exception as e:
                logger.error(f"Failed processing {code_str} - {datatype}: {e}")
                failed_codes[code_str] = str(e)
                continue

        if not gdfs:
            logger.warning("No dataframes collected for concatenation.")
            if failed_codes:
                logger.warning(f"Failures occurred for: {failed_codes}")
            return None

        logger.info(f"Successfully processed {len(processed_codes)} municipalities.")
        if failed_codes:
            logger.warning(
                f"Failed to process {len(failed_codes)} municipalities: {failed_codes}"
            )

        logger.info("Concatenating GeoDataFrames...")
        final_crs = target_crs if target_crs else (gdfs[0].crs if gdfs else None)
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs=final_crs
        )

        logger.info(
            f"Final GeoDataFrame created with {len(combined_gdf)} features. CRS: {combined_gdf.crs}"
        )
        return combined_gdf

    def _perform_entrance_linkage(
            self, addresses_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Link entrances to buildings based on the localId_address and localId_building columns.
        :param addresses_gdf: GeoDataFrame containing address data.
        :param buildings_gdf: GeoDataFrame containing building data.
        :return: GeoDataFrame with linked entrances and buildings or None if no matches found.
        """

        addr_gdf = addresses_gdf.copy()
        bldg_gdf = buildings_gdf.copy()

        addr_gdf = addr_gdf.add_suffix("_address")
        bldg_gdf = bldg_gdf.add_suffix("_building")

        addr_orig_geom_col = addresses_gdf._geometry_column_name
        bldg_orig_geom_col = buildings_gdf._geometry_column_name

        if (
            "geometry_address" not in addr_gdf.columns
            and f"{addr_orig_geom_col}_address" in addr_gdf.columns
        ):
            addr_gdf = addr_gdf.rename_geometry("geometry_address")
        if (
            "geometry_building" not in bldg_gdf.columns
            and f"{bldg_orig_geom_col}_building" in bldg_gdf.columns
        ):
            bldg_gdf = bldg_gdf.rename_geometry("geometry_building")

        if "localId_address" not in addr_gdf.columns:
            logger.error(
                "Required column 'localId_address' not found in Addresses data."
            )
            return None
        if "localId_building" not in bldg_gdf.columns:
            logger.error(
                "Required column 'localId_building' not found in Buildings data."
            )
            return None

        addr_gdf["merge_id_address"] = (
            addr_gdf["localId_address"].astype(str).str.rsplit(".", n=1).str[-1]
        )

        if addr_gdf.empty:
            logger.warning("No 'Entrance' features found in Addresses data.")
            return None

        entrance_counts = addr_gdf["merge_id_address"].value_counts()
        addr_gdf["entrance_count_per_building"] = addr_gdf["merge_id_address"].map(
            entrance_counts
        )

        merged_gdf = addr_gdf.merge(
            bldg_gdf,
            left_on="merge_id_address",
            right_on="localId_building",
            how="left",
        )

        merged_gdf.drop(columns=["merge_id_address"], inplace=True, errors="ignore")

        if "geometry_address" in merged_gdf.columns:
            merged_gdf = merged_gdf.set_geometry("geometry_address")
        else:
            logger.warning("Could not set final geometry for merged data.")

        return merged_gdf

    def get_matched_entrance_with_buildings(
        self, municipality_code: str, target_crs: Optional[str] = "EPSG:4326"
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Retrieves and links entrances to buildings for a specific municipality.

        This method fetches address and building data for the specified municipality, links entrances
        to their corresponding buildings, and returns the result as a GeoDataFrame.

        :param municipality_code: The municipality code to retrieve and link data for.
        :type municipality_code: str
        :param target_crs: The target coordinate reference system (CRS) for the GeoDataFrame. Default is "EPSG:4326".
        :type target_crs: Optional[str]
        :return: A GeoDataFrame with matched entrances and buildings or None if no matches are found.
        :rtype: Optional[gpd.GeoDataFrame]

        Usage
        --------

        Get already matched entrances and buildings in one step:

        >>> from mango.clients.catastro import CatastroData
        >>> catastro = CatastroData()
        >>> merged_data_auto = catastro.get_matched_entrance_with_buildings("25252")
        """
        self._ensure_index_loaded()
        code_str = str(municipality_code).zfill(5)
        logger.info(f"Matching entrances and buildings for {code_str}")

        try:
            addresses_gdf = self.get_data(code_str, "Addresses", target_crs=target_crs)
            buildings_gdf = self.get_data(
                code_str, "Buildings", subtype="Buildings", target_crs=target_crs
            )
        except Exception as e:
            logger.error(
                f"Failed to retrieve base data for matching in {code_str}: {e}"
            )
            return None

        if addresses_gdf is None or addresses_gdf.empty:
            logger.warning(
                f"Cannot perform match: Addresses data missing or empty for {code_str}."
            )
            return None
        if buildings_gdf is None or buildings_gdf.empty:
            logger.warning(
                f"Cannot perform match: Buildings data missing or empty for {code_str}."
            )
            return None

        linked_gdf = self._perform_entrance_linkage(addresses_gdf, buildings_gdf)

        logger.info(
            f"Successfully matched {len(linked_gdf)} entrances to buildings for {code_str}."
        )
        return linked_gdf

    def link_entrances_to_buildings(
        self, addresses_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Links entrances to buildings based on their identifiers.

        This method takes GeoDataFrames for addresses and buildings, matches entrances to buildings
        using their local identifiers, and returns a GeoDataFrame with the linked data.

        :param addresses_gdf: A GeoDataFrame containing address data.
        :type addresses_gdf: gpd.GeoDataFrame
        :param buildings_gdf: A GeoDataFrame containing building data.
        :type buildings_gdf: gpd.GeoDataFrame
        :return: A GeoDataFrame with linked entrances and buildings or None if no matches are found.
        :rtype: Optional[gpd.GeoDataFrame]

        Usage
        --------

        Link entrances to buildings:

        >>> from mango.clients.catastro import CatastroData
        >>> catastro = CatastroData()

        >>> addresses_data = catastro.get_data("28900", "Addresses")
        >>> buildings_data = catastro.get_data("28900", "Buildings")

        >>> merged_data = catastro.link_entrances_to_buildings(addresses_data, buildings_data)
        """

        linked_gdf = self._perform_entrance_linkage(
            addresses_gdf=addresses_gdf.copy(), buildings_gdf=buildings_gdf.copy()
        )

        return linked_gdf

    def download_all_data(self, output_directory: str, sample: bool = False) -> None:
        """
        Downloads all available data from the Catastro API and saves it to the specified directory,
        structured by timestamp and datatype.

        :param output_directory: Directory where the downloaded data will be saved.
        :type output_directory: str
        :param sample: If True, only a sample of the data will be downloaded. Default is False.
        :type sample: bool
        :return: None

        Usage
        --------

        Download all data to a specified directory:

        >>> from mango.clients.catastro import CatastroData
        >>> catastro = CatastroData()
        >>> catastro.download_all_data("path/to/output_directory")

        Download a sample of the data:

        >>> catastro.download_all_data("path/to/output_directory", sample=True)
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_directory = os.path.join(output_directory, timestamp)
        os.makedirs(run_directory, exist_ok=True)

        for datatype in self.municipalities_links["datatype"].unique():
            logger.info(f"Downloading {datatype} data...")

            datatype_dir = os.path.join(run_directory, datatype)
            os.makedirs(datatype_dir, exist_ok=True)

            if not sample:
                datatype_links = self.municipalities_links[
                    self.municipalities_links["datatype"] == datatype
                ]["zip_link"]
            else:
                datatype_links = self.municipalities_links[
                    self.municipalities_links["datatype"] == datatype
                ]["zip_link"].sample(n=10)

            for link in datatype_links:
                success = self._download_and_extract_zip(link, datatype_dir)
                if success:
                    logger.info(
                        f"Successfully extracted zip from {link} to {datatype_dir}"
                    )
                else:
                    logger.error(f"Failed to process zip from {link}")
