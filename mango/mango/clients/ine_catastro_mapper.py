"""
Module for mapping between different coding systems used by Spanish official sources.

This module provides the CatastroINEMapper class for mapping between the coding systems
used by the Spanish Cadastre (Catastro) and the Spanish National Statistics Institute (INE).

"""

import logging
from io import BytesIO
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from mango.clients.catastro import CatastroData
from mango.clients.ine import INEData

logger = logging.getLogger(__name__)


class CatastroINEMapper:
    """
    Utility class to handle mapping between Catastro and INE municipality codes.

    This class integrates with both Catastro and INE modules to enrich municipality data.
    It processes raw relationship data obtained from:
    https://www.fega.gob.es/es/content/relacion-de-municipios-por-ccaa-con-equivalencias-entre-los-codigos-ine-y-catastro-2025.
    The mapping file url should be checked and updated regularly as the data may change.

    :param load_from_apis: Whether to load additional data from Catastro and INE. (Can take long as it has to fetch the catastro index)
    :type load_from_apis: bool
    :param save_processed: Whether to save the processed mapping to a file.
    :type save_processed: bool
    :param catastro_client: Optional pre-initialized CatastroData client. (RECOMENDED, with cached index)
    :type catastro_client: CatastroData
    :param ine_client: Optional pre-initialized INEData client.
    :type ine_client: INEData
    :param processed_file: Path to a pre-processed mapping file (if provided, mapping_file is ignored).
    :type processed_file: str

    Usage
    --------

    >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper
    >>> from mango.clients.catastro import CatastroData
    >>> from mango.clients.ine import INEData

    Initialize the mapper with INE data loading to get the accurate names from both INE and Catastro.
    It is recommended to pass the Catastro client with the cache_dir parameter set up to avoid downloading the data again. (Can take a while)

    >>> mapper = CatastroINEMapper(load_from_apis=True,
    ...                            ine_client=INEData(),
    ...                            catastro_client=CatastroData(cache=True, cache_file_path=r"catastro_cache.json", verbose=True))
    """

    PROCESSED_FILENAME = "processed_municipalities_mapping.csv"
    MAPPING_FILE_URL = "https://www.fega.gob.es/sites/default/files/files/document/RELACION_MUNICIPIOS_RUSECTOR_AGREGADO_ZONA_CAMPA%C3%91A_2025.xlsx"

    def __init__(
        self,
        load_from_apis: bool = False,
        save_processed: bool = False,
        catastro_client: Optional[CatastroData] = None,
        ine_client: Optional[INEData] = None,
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
        Enrich the mapping with names from Catastro and INE.

        This method fetches additional municipality names from the Catastro and INE.
        and updates the mapping table accordingly.
        """
        logger.info("Enriching municipality data with names from Catastro and INE.")

        if self.load_from_apis:
            if not self.catastro_client:
                logger.info("Initializing Catastro client")
                self.catastro_client = CatastroData(verbose=False, cache=False)

            if not self.ine_client:
                logger.info("Initializing INE client")
                self.ine_client = INEData(verbose=False)

            self._enrich_with_catastro_names()

            self._enrich_with_ine_names()

    def _enrich_with_catastro_names(self) -> None:
        """
        Get municipality names from Catastro data.

        This method retrieves municipality names from the Catastro and updates
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

        This method retrieves municipality names from the INE and updates
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

        >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper
        >>> mapper = CatastroINEMapper(load_from_apis=False)

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

        >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper
        >>> mapper = CatastroINEMapper(load_from_apis=False)

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

        >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper
        >>> mapper = CatastroINEMapper(load_from_apis=False)

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

        >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper
        >>> mapper = CatastroINEMapper(load_from_apis=False)

        >>> mapping_table = mapper.get_mapping_table()
        >>> print(mapping_table.head())
        """
        return self.mapping


def merge_catastro_census(
    cadastral_addresses_with_buildings: gpd.GeoDataFrame,
    census_data: gpd.GeoDataFrame,
    mapping: CatastroINEMapper,
) -> gpd.GeoDataFrame:
    """
    Merges cadastral address and building data with census data based on spatial relationships
    and municipality code matching, correcting mismatches by finding the nearest census unit.

    :param mapping: Mapping object for converting municipality codes.
    :type mapping: CatastroINEMapper
    :param cadastral_addresses_with_buildings: GeoDataFrame containing cadastral addresses and buildings.
    :type cadastral_addresses_with_buildings: gpd.GeoDataFrame
    :param census_data: GeoDataFrame containing census data.
    :type census_data: gpd.GeoDataFrame
    :return: GeoDataFrame with merged data, including population estimates at each address.
    :rtype: gpd.GeoDataFrame

    Usage
    --------

    >>> from mango.clients.catastro import CatastroData
    >>> from mango.clients.ine import INEData
    >>> from mango.clients.ine_catastro_mapper import CatastroINEMapper

    >>> catastro_client = CatastroData(cache=True, cache_file_path=r"catastro_cache.json", verbose=True)
    >>> ineDataSource = INEData()
    >>> mapper = CatastroINEMapper(load_from_apis=True,
    ...                            ine_client=ineDataSource,
    ...                            catastro_client=catastro_client)

    >>> census_df = ineDataSource.fetch_full_census()
    >>> census_with_geom = ineDataSource.enrich_with_geometry(census_df)
    >>> cadastral_addresses_with_buildings_df = catastro_client.get_matched_entrance_with_buildings("25252")

    >>> merged_data = merge_catastro_census(cadastral_addresses_with_buildings_df, census_with_geom, mapper)
    """

    # initial geospatial join
    address_with_census = gpd.sjoin(
        cadastral_addresses_with_buildings.to_crs(census_data.crs),
        census_data,
        how="left",
        predicate="within",
        rsuffix="_census",
    )

    # identify mismatched addresses
    mismatch_addresses = address_with_census[
        "catastro_municipality_code_building"
    ].astype(str) != address_with_census["ine_municipality_code"].astype(str).apply(
        mapping.ine_to_catastro_code
    )

    # reassign to the closest census unit excluding the previously assigned
    matched = address_with_census[~mismatch_addresses]
    mismatched = address_with_census[mismatch_addresses].copy()

    redone_matches = []

    if not mismatched.empty:
        for idx, row in mismatched.iterrows():
            single_building = gpd.GeoDataFrame(
                [row],
                columns=mismatched.columns[:43],
                geometry=[row.geometry_address],
                crs=mismatched.crs,
            )
            filtered_census = census_data[census_data["CUSEC"] != row["CUSEC"]]

            corrected = gpd.sjoin_nearest(single_building, filtered_census, how="left")

            redone_matches.append(corrected)

        redone_matches_gdf = pd.concat(redone_matches, ignore_index=True)

        final_address_with_census = pd.concat(
            [matched, redone_matches_gdf], ignore_index=True
        )
        final_address_with_census = gpd.GeoDataFrame(
            final_address_with_census, geometry="geometry_address", crs=census_data.crs
        )
        final_address_with_census = final_address_with_census.drop(
            columns=["geometry", "index_right"]
        )

    else:
        return matched

    return final_address_with_census


def distribute_population_by_dwellings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Estimate and assign population per building entrance based on the number of dwellings.

    This function distributes the total census tract population to each building entrance
    proportionally based on its share of the total dwellings in that tract.

    :param gdf: GeoDataFrame containing address-level data with building and census tract info.
    :type gdf: gpd.GeoDataFrame
    :return: GeoDataFrame with a new column 'population_per_entrance'.
    :rtype: gpd.GeoDataFrame
    """

    # number of dwellings assigned to each entrance
    gdf["dwellings_per_entrance"] = np.where(
        gdf["numberOfDwellings_building"] > 0,
        gdf["numberOfDwellings_building"] / gdf["entrance_count_per_building"],
        0,
    )

    # total number of dwellings in each census tract (CUSEC)
    total_dwellings_per_cusec = (
        gdf.groupby("CUSEC")["dwellings_per_entrance"]
        .sum()
        .reset_index(name="total_dwellings_in_cusec")
    )

    # merge the total dwellings per census tract back to the original dataframe
    gdf = gdf.merge(total_dwellings_per_cusec, on="CUSEC", how="left")

    # population per entrance by distributing the total population of the census tract
    # proportionally to the number of dwellings assigned to each entrance
    gdf["population_per_entrance"] = (
        gdf["ine_population"]
        / gdf["total_dwellings_in_cusec"]
        * gdf["dwellings_per_entrance"]
    )

    return gdf


def export_population_gdf(
    gdf: gpd.GeoDataFrame, path: str, lightweight: bool = False
) -> None:
    """
    Export the population data from a GeoDataFrame to a new GeoDataFrame.

    :param gdf: GeoDataFrame containing population data.
    :type gdf: gpd.GeoDataFrame
    :param path: Path to save the new GeoDataFrame.
    :type path: str
    :param lightweight: If True, exports a lightweight version of the GeoDataFrame.
    :return: None
    """

    # Validate input
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame.")
    if not isinstance(path, str):
        raise ValueError("Path must be a string.")
    if not path.endswith(".geojson"):
        raise ValueError("Path must end with .geojson")

    if lightweight:
        gdf[["gml_id_address", "geometry_address", "population_per_entrance"]].to_file(
            path, driver="GeoJSON"
        )
    else:
        gdf.to_file(path, driver="GeoJSON")

    print(f"Population data exported successfully to {path}")


def reload_population_gdf_from_file(
    path: str, lightweight: bool = False
) -> gpd.GeoDataFrame:
    """
    Reload a GeoDataFrame from a file.

    :param path: Path to the file.
    :type path: str
    :param lightweight: If True, it expects the lightweight version of the GeoDataFrame.
    :type lightweight: bool
    :return: GeoDataFrame loaded from the file.
    :rtype: gpd.GeoDataFrame
    """

    # Validate input
    if not isinstance(path, str):
        raise ValueError("Path must be a string.")
    if not path.endswith(".geojson"):
        raise ValueError("File extension must be .geojson")

    # Load from file
    gdf = gpd.read_file(path)

    gdf.rename(columns={"geometry": "geometry_address"}, inplace=True)
    gdf.set_geometry("geometry_address", inplace=True)

    if lightweight:
        return gdf

    col_to_move = "geometry_address"
    target_position = 11

    # Ensure we don't exceed the number of columns
    cols = gdf.columns.tolist()
    if col_to_move in cols:
        cols.remove(col_to_move)
        # Insert at desired position, pad if needed
        if target_position >= len(cols):
            cols.append(col_to_move)
        else:
            cols.insert(target_position, col_to_move)

        gdf = gdf[cols]

    return gdf