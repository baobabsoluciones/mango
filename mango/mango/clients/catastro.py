import geopandas as gpd
import pandas as pd
import requests
import os
import zipfile
from io import BytesIO
from time import sleep
from bs4 import BeautifulSoup
import json


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

    def __init__(self, verbose=False, cache=False, request_timeout=30):
        """
        Initializes the CatastroData module.

        :param bool verbose: If True, prints detailed messages. Defaults to False.
        :param bool cache: If True, loads/saves the municipality index from/to cache. Defaults to False.
        :param int request_timeout: Timeout in seconds for network requests. Defaults to 30.
        """
        self.verbose = verbose
        self.cache = cache
        self.request_timeout = request_timeout
        self.municipalities_links = pd.DataFrame()
        self._index_loaded = False
        self.available_datatypes = list(self.BASE_URLS.keys())

    def _log(self, message) -> None:
        """Logs a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _fetch_content(self, url) -> bytes or None:
        """Fetches content from a URL with error handling."""
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            self._log(f"Successfully fetched content from: {url}")
            return response.content
        except requests.exceptions.RequestException as e:
            self._log(f"Error fetching {url}: {e}")
            return None

    def _get_soup(self, url) -> BeautifulSoup or None:
        """Fetches content and returns a BeautifulSoup object."""
        content = self._fetch_content(url)
        if content:
            try:
                soup = BeautifulSoup(content, features="xml")
                return soup
            except Exception as e:
                self._log(f"Error parsing XML from {url}: {e}")
                return None
        return None

    def _fetch_and_parse_links(self) -> pd.DataFrame:
        """
        Fetches and parses territorial and municipality links from the ATOM feeds.
        """
        self._log("Fetching and parsing municipality .zip links...")
        municipalities_data = []

        for dataset, base_url in self.BASE_URLS.items():
            self._log(f"Processing dataset: {dataset} from {base_url}")
            base_soup = self._get_soup(base_url)
            if not base_soup:
                self._log(
                    f"Skipping dataset {dataset} due to fetch/parse error on base URL."
                )
                continue

            territorial_entries = base_soup.find_all("entry")
            for terr_entry in territorial_entries:
                title_element = terr_entry.find("title")
                link_element = terr_entry.find("link", attrs={"href": True})

                if not (
                    title_element
                    and link_element
                    and len(title_element.get_text(strip=True)) >= 22
                ):
                    self._log(f"Skipping invalid territorial entry in {dataset}")
                    continue

                terr_name_full = title_element.get_text(strip=True)
                territorial_link = link_element.get("href")
                try:
                    # TODO: Review
                    territorial_code = str(terr_name_full[19:21]).zfill(2)
                    territorial_name = terr_name_full[22:].strip()
                except IndexError:
                    self._log(
                        f"Could not parse code/name from title: '{terr_name_full}' in {dataset}. Skipping."
                    )
                    continue

                self._log(
                    f"  Processing Territorial Office: {territorial_code} - {territorial_name}"
                )

                terr_soup = self._get_soup(territorial_link)
                if not terr_soup:
                    self._log(
                        f"  Skipping territorial office {territorial_name} due to fetch/parse error."
                    )
                    continue

                municipality_entries = terr_soup.find_all("entry")
                for mun_entry in municipality_entries:
                    link_tag = mun_entry.find(
                        "link", rel="enclosure", attrs={"href": True}
                    )
                    title_tag = mun_entry.find("title")

                    if not (link_tag and title_tag):
                        self._log("  Skipping entry with missing link or title.")
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
                        self._log(
                            f"  Could not parse code/name from title: '{municipality_info}'. Skipping."
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
                    self._log(
                        f"    Found: {municipality_code} - {municipality_name} ({dataset})"
                    )

        self._log(f"Finished parsing. Found {len(municipalities_data)} total links.")
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
        """
        if self._index_loaded:
            self._log("Municipality index already loaded.")
            return

        if self.cache and os.path.exists(self.CACHE_FILE):
            try:
                self._log(f"Loading municipalities index from cache: {self.CACHE_FILE}")
                self.municipalities_links = pd.read_json(self.CACHE_FILE, dtype=False)
                self._index_loaded = True
                self._log("Successfully loaded index from cache.")
                return
            except (json.JSONDecodeError, ValueError, KeyError, FileNotFoundError) as e:
                self._log(f"Failed to load from cache ({e}). Fetching fresh data.")
                if os.path.exists(self.CACHE_FILE):
                    try:
                        os.remove(self.CACHE_FILE)
                    except OSError as remove_err:
                        self._log(
                            f"Warning: Could not remove potentially corrupt cache file {self.CACHE_FILE}: {remove_err}"
                        )

        self._log("Fetching fresh municipalities index...")
        self.municipalities_links = self._fetch_and_parse_links()
        self._index_loaded = True

        if self.cache and not self.municipalities_links.empty:
            try:
                self._log(f"Saving municipalities index to cache: {self.CACHE_FILE}")
                self.municipalities_links.to_json(self.CACHE_FILE, indent=4)
            except IOError as e:
                self._log(f"Error saving index to cache: {e}")
        elif self.cache and self.municipalities_links.empty:
            self._log("Skipping cache save because fetched index is empty.")

    def _ensure_index_loaded(self) -> None:
        """Checks if the index is loaded and raises an error if not."""
        if not self._index_loaded or self.municipalities_links.empty:
            raise RuntimeError(
                "Municipality index not loaded. Call load_index() first."
            )

    def _download_and_extract(self, municipality_code, datatype):
        """Downloads and extracts the GML file from the zip archive."""
        self._ensure_index_loaded()

        self._log(
            f"Filtering for Municipality Code: {municipality_code}, Datatype: {datatype}"
        )

        municipality_code_str = str(municipality_code).zfill(5)

        filtered_links = self.municipalities_links[
            (self.municipalities_links["Datatype"] == datatype)
            & (self.municipalities_links["Municipality Code"] == municipality_code_str)
        ]

        if filtered_links.empty:
            raise ValueError(
                f"No index entry found for Municipality Code '{municipality_code_str}' and Datatype '{datatype}'. Check availability."
            )

        zip_link = filtered_links.iloc[0]["Zip Link"]

        if not zip_link or pd.isna(zip_link):
            raise ValueError(
                f"Index entry found, but no valid Zip Link for municipality '{municipality_code_str}' and Datatype '{datatype}'."
            )

        self._log(
            f"Downloading data for municipality {municipality_code_str} ({datatype}) from {zip_link}..."
        )
        zip_content = self._fetch_content(zip_link)

        if not zip_content:
            raise ConnectionError(f"Failed to download zip file from {zip_link}")

        try:
            with zipfile.ZipFile(BytesIO(zip_content), "r") as zip_ref:
                file_suffixes = {
                    "Addresses": ".gml",
                    "Buildings": ".building.gml",
                    "CadastralParcels": ".cadastralparcel.gml",
                }
                suffix = file_suffixes.get(datatype)
                if not suffix:

                    raise ValueError(
                        f"Internal error: Invalid datatype '{datatype}' specified for suffix lookup."
                    )

                gml_filename = None
                for filename in zip_ref.namelist():
                    if datatype == "Addresses":
                        if (
                            filename.upper().endswith(".GML")
                            and "_AD_" in filename.upper()
                        ):
                            gml_filename = filename
                            break
                    elif filename.lower().endswith(suffix.lower()):
                        gml_filename = filename
                        break

                if gml_filename:
                    self._log(f"Extracting '{gml_filename}' from zip archive.")
                    return zip_ref.open(gml_filename)
                else:
                    raise FileNotFoundError(
                        f"No file ending with expected pattern ('{suffix}' or specific Address GML) "
                        f"found in the zip archive from {zip_link} for datatype {datatype}. "
                        f"Files found: {zip_ref.namelist()}"
                    )
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                f"Downloaded file from {zip_link} is not a valid zip archive."
            )
        except Exception as e:
            self._log(f"An unexpected error occurred during zip extraction: {e}")
            raise

    def available_municipalities(self, datatype: str) -> pd.DataFrame:
        """
        Returns a DataFrame of available municipalities for a given datatype.
        Requires load_index() to be called first.

        :param str datatype: The type of data ("Buildings", "CadastralParcels", "Addresses").
        :return: pandas.DataFrame with names and codes of all available municipalities for the specified datatype
        """
        self._ensure_index_loaded()
        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available types: {self.available_datatypes}"
            )

        return (
            self.municipalities_links[
                self.municipalities_links["Datatype"] == datatype
            ][["Municipality Code", "Municipality Name"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def get_municipality_data(self, municipality_code, datatype) -> gpd.GeoDataFrame:
        """
        Main method of the class. Gets a GeoDataFrame for a single municipality and datatype.
        Requires load_index() to be called first.

        :param municipality_code: The 5-digit code of the municipality.
        :param datatype: The type of data ("Buildings", "CadastralParcels", "Addresses").
        :return: geopandas.GeoDataFrame with the loaded spatial data.
        """
        self._ensure_index_loaded()
        self._log(
            f"Attempting to get {datatype} data for municipality {municipality_code}..."
        )

        municipality_code_str = str(municipality_code).zfill(5)

        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available types: {self.available_datatypes}"
            )

        try:
            with self._download_and_extract(
                municipality_code_str, datatype
            ) as gml_file:
                gdf = gpd.read_file(gml_file)
                self._log(
                    f"Successfully loaded GeoDataFrame for {municipality_code_str} ({datatype})."
                )
                return gdf
        except (
            ValueError,
            FileNotFoundError,
            ConnectionError,
            zipfile.BadZipFile,
            RuntimeError,
        ) as e:
            self._log(
                f"Failed to get data for {municipality_code_str} ({datatype}): {e}"
            )
            raise
        except Exception as e:
            self._log(
                f"An unexpected error occurred loading GeoDataFrame for {municipality_code_str} ({datatype}): {e}"
            )
            raise

    def get_multiple_municipalities_data(
        self,
        municipality_codes: Union[str, list[str]],
        datatype,
        target_crs="EPSG:4326",
    ) -> gpd.GeoDataFrame or None:
        """
        Returns a combined GeoDataFrame for multiple municipalities of the same datatype.
        Requires `load_index()` to be called first. Optionally reprojects to the specified `target_crs`,
        or defaults to EPSG:4326 for merging all municipalities into one GeoDataFrame.

        :param municipality_codes: A list of 5-digit municipality codes (str or int).
        :param datatype: The type of data ("Buildings", "CadastralParcels", "Addresses").
        :param target_crs: The target Coordinate Reference System (default: "EPSG:4326").
        :returns: Combined data into a geopandas.DataFrame, or None if no data could be processed.
        """
        self._ensure_index_loaded()

        if not isinstance(municipality_codes, list) and not isinstance(
            municipality_codes, str
        ):
            raise TypeError(
                "municipality_codes must be a list of strings or a single string."
            )
        if datatype not in self.available_datatypes:
            raise ValueError(
                f"Invalid datatype '{datatype}'. Available types: {self.available_datatypes}"
            )

        gdfs = []
        if isinstance(municipality_codes, str):
            municipality_codes = [municipality_codes]
        for code in municipality_codes:
            sleep(0.1)
            code_str = str(code).zfill(5)
            try:
                self._log(f"Processing municipality: {code_str} ({datatype})")
                gdf = self.get_municipality_data(code_str, datatype)

                if gdf is not None and not gdf.empty:
                    if gdf.crs and gdf.crs != target_crs:
                        self._log(
                            f"Reprojecting {code_str} from {gdf.crs} to {target_crs}"
                        )
                        gdf = gdf.to_crs(target_crs)
                    elif not gdf.crs:
                        self._log(
                            f"Warning: CRS missing for {code_str}. Cannot reproject. Assuming {target_crs}."
                        )

                    gdf["municipality_code"] = code_str
                    gdfs.append(gdf)
                else:
                    self._log(
                        f"No valid data returned for municipality {code_str} ({datatype})."
                    )

            except Exception as e:
                self._log(
                    f"Error processing municipality {code_str} ({datatype}): {e}. Skipping."
                )
                continue

        if not gdfs:
            self._log("No data collected for any of the requested municipalities.")
            return None

        self._log(f"Concatenating data for {len(gdfs)} municipalities.")
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs=target_crs if gdfs else None
        )
        return combined_gdf
