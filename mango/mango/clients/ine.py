"""
Module for obtaining census data from the Spanish National Statistics Institute (INE).

This module provides the INEAPIClient class for retrieving and processing
census data from the INE API.
"""

import requests
import pandas as pd
import os
from io import BytesIO
import geopandas as gpd
import logging
import json
import zipfile
import tempfile

GEOMETRY_DATA_URL = r"https://www.ine.es/prodyser/cartografia/seccionado_2024.zip"
INE_CODES_URL = "https://www.ine.es/daco/daco42/codmun/diccionario25.xlsx"
BASE_API_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/"
FULL_CENSUS_URL = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65031.csv"
PROVINCE_CODES = {
    "Albacete": "69095",
    "Alicante/Alacant": "69102",
    "Almería": "69106",
    "Araba/Álava": "65039",
    "Asturias": "69110",
    "Ávila": "69114",
    "Badajoz": "69118",
    "Balears, Illes": "69122",
    "Barcelona": "69126",
    "Bizkaia": "69130",
    "Burgos": "69134",
    "Cáceres": "69138",
    "Cádiz": "69142",
    "Cantabria": "69146",
    "Castellón/Castelló": "69150",
    "Ciudad Real": "69154",
    "Córdoba": "69158",
    "Coruña, A": "69162",
    "Cuenca": "69166",
    "Gipuzkoa": "69170",
    "Girona": "69174",
    "Granada": "69178",
    "Guadalajara": "69182",
    "Huelva": "69186",
    "Huesca": "69190",
    "Jaén": "69194",
    "León": "69198",
    "Lleida": "69202",
    "Lugo": "69206",
    "Madrid": "69210",
    "Málaga": "69214",
    "Murcia": "69218",
    "Navarra": "69222",
    "Ourense": "69226",
    "Palencia": "69230",
    "Palmas, Las": "69234",
    "Pontevedra": "69238",
    "Rioja, La": "69242",
    "Salamanca": "69246",
    "Santa Cruz de Tenerife": "69250",
    "Segovia": "69254",
    "Sevilla": "69258",
    "Soria": "69262",
    "Tarragona": "69266",
    "Teruel": "69343",
    "Toledo": "69270",
    "Valencia/València": "69274",
    "Valladolid": "69278",
    "Zamora": "69282",
    "Zaragoza": "69286",
    "Ceuta": "69290",
    "Melilla": "69294",
}

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _fetch_url(url: str) -> bytes:
    """
    Fetches content from a URL with error handling.

    :param url: URL to fetch
    :type url: str
    :return: Content of the response
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Error fetching data from {url}: {e}")
        raise


def _clean_full_census(df: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """
    Cleans the full census DataFrame by filtering out unwanted rows.

    :param df: DataFrame to clean
    :type df: pd.DataFrame
    :param year: Year to filter the data by, if None it will use the last year available in the data
    :type year: int
    :return: Cleaned DataFrame
    :rtype: pd.DataFrame
    """

    df = df[
        df["Provincias"].notna()
        & df["Municipios"].notna()
        & df["Secciones"].notna()
        & (df["Sexo"] == "Total")
        & (df["Lugar de nacimiento"] == "Total")
    ]

    # filter by specific year if provided or use the last year available
    year_filter = year if year else df["Periodo"].max()
    df = df[df["Periodo"] == year_filter]

    df["ine_province_code"] = df["Provincias"].str[:2].astype(str).str.zfill(2)
    df["ine_province_name"] = df["Provincias"].str[3:]
    df["ine_municipality_code"] = df["Municipios"].str[:5].astype(str).str.zfill(5)
    df["ine_municipality_name"] = df["Municipios"].str[5:]
    df["ine_census_tract_code"] = df["Secciones"].str[:10].astype(str).str.zfill(9)

    # Convert population to numeric where thousands separator is a dot ignoring NANs
    df["Total"] = df["Total"].str.replace(".", "", regex=False)
    df["Total"] = df["Total"].str.replace(",", ".", regex=False)
    df["ine_population"] = pd.to_numeric(df["Total"], errors="coerce")

    final_columns = [
        "ine_province_code",
        "ine_province_name",
        "ine_municipality_code",
        "ine_municipality_name",
        "ine_census_tract_code",
        "ine_population",
    ]

    df_final = df[final_columns]

    return df_final


def _valid_province_code(province_code: str) -> bool:
    """
    Validates the province code.

    :param province_code: Province code to validate
    :type province_code: str
    :return: True if valid, False otherwise
    :rtype: bool
    """
    if isinstance(province_code, int):
        province_code = str(province_code)
    if len(province_code) != 5:
        raise ValueError("Table code must be a 5-digit string/integer.")
    if province_code not in PROVINCE_CODES.values():
        return False

    return True


def save_geojson_file(
        df_geom: gpd.GeoDataFrame, folder_path: str, file_name: str
) -> None:
    """
    Saves a GeoDataFrame to a GeoJSON file.

    :param df_geom: GeoDataFrame to save
    :type df_geom: gpd.GeoDataFrame
    :param folder_path: Path to the folder where the file will be saved
    :type folder_path: str
    :param file_name: Name of the output file (without extension)
    :type file_name: str
    :return: None
    """

    if not isinstance(df_geom, gpd.GeoDataFrame):
        raise ValueError("df_geom must be a GeoDataFrame.")

    if df_geom.empty:
        raise ValueError("GeoDataFrame is empty. Nothing to save.")

    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError("Invalid folder_path. Must be a non-empty string.")

    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("Invalid file_name. Must be a non-empty string.")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    file_name = file_name if file_name.endswith(".geojson") else f"{file_name}.geojson"
    file_path = os.path.join(folder_path, file_name)

    try:
        df_geom.to_file(file_path, driver="GeoJSON")
    except Exception as e:
        logger.error(f"Could not save the GeoJSON to {file_path}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(f"GeoJSON data saved to {file_path}")


def list_table_codes() -> dict:
    """
    Lists all available table codes for provinces in the INE API.

    Each table code corresponds to a specific province and can be used to fetch census data.

    :return: A dictionary mapping province names to their respective table codes.
    :rtype: dict

    Usage
    --------

    Show all available table codes in the INE API:

    >>> table_codes = list_table_codes()
    >>> print(list_table_codes())
    """
    return PROVINCE_CODES


def fetch_full_census(year: int = None) -> pd.DataFrame:
    """
    Fetches the full census data from the INE API and cleans it.

    This function retrieves census data for all available provinces, filters it by the specified year
    (or the latest year available if not specified), and returns a cleaned DataFrame.

    :param year: The year to filter the data by. If None, the latest year available is used.
    :type year: int
    :return: A cleaned DataFrame containing census data with province, municipality, and census tract details.
    :rtype: pd.DataFrame

    Usage
    --------

    Fetch full census and add the geometry data:

    >>> full_census = fetch_full_census()
    >>> print(full_census.head())
    """

    content = _fetch_url(FULL_CENSUS_URL)

    df_raw = pd.read_csv(BytesIO(content), sep=";", dtype=str, thousands=".")
    df = _clean_full_census(df_raw, year)

    return df


def enrich_with_geometry(
        census_df: pd.DataFrame, geometry_path: str = None
) -> gpd.GeoDataFrame:
    """
    Enriches a census DataFrame with spatial geometry data.

    This function merges census data with geometrical data from a spatial file (e.g., shapefile or GeoJSON).
    If no file path is provided, it fetches the geometry data from a predefined URL.

    :param census_df: A cleaned census DataFrame to enrich with geometry.
    :type census_df: pd.DataFrame
    :param geometry_path: Path to the spatial file containing geometry data. If None, data is fetched from a URL.
    :type geometry_path: str
    :return: A GeoDataFrame containing both census and spatial data.
    :rtype: gpd.GeoDataFrame

    Usage
    --------

    Enrich previously fetched census data with the geometries assigned to the census tracts:

    >>> from mango.clients.ine import INEAPIClient, fetch_full_census, enrich_with_geometry
    >>> import geopandas as gpd
    >>> ine_api = INEAPIClient()
    >>> census_df = ine_api.fetch_census_by_section("69202")
    >>> geometry_df = enrich_with_geometry(census_df)
    >>> geometry_df.explore()
    """

    if geometry_path is None:
        try:
            response = requests.get(GEOMETRY_DATA_URL)
            response.raise_for_status()
            geo_bytes = response.content
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {GEOMETRY_DATA_URL}: {e}")
            raise

        with zipfile.ZipFile(BytesIO(geo_bytes)) as zf:
            shp_file = next((f for f in zf.namelist() if f.endswith(".shp")), None)
            if not shp_file:
                raise ValueError("No shapefile found in the zip archive")
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                geometry_data = gpd.read_file(os.path.join(tmpdir, shp_file))

    else:
        try:
            geometry_data = gpd.read_file(geometry_path)
        except Exception as e:
            logger.error(f"Could not read geometry file: {e}")
            raise

    geometry_with_census = geometry_data.merge(
        census_df, left_on="CUSEC", right_on="ine_census_tract_code"
    )

    # Move geometry column to the end
    geometry_with_census = geometry_with_census[
        [col for col in geometry_with_census.columns if col != "geometry"]
        + ["geometry"]
    ]

    return geometry_with_census


def download_full_census_with_geometry(
    folder_path: str,
    file_name: str = "full_census_with_geometry",
    geometry_path: str = None,
        split_by_province: bool = False,
) -> None:
    """
    Downloads the cleaned full census data and its geometry, and saves it to a specified path.

    :param folder_path: Path to the folder where the data will be saved.
    :type folder_path: str
    :param file_name: Name of the output file (without extension).
    :type file_name: str
    :param geometry_path: Path to the spatial file containing geometry data. If None, data is fetched from a URL.
    :type geometry_path: str
    :param split_by_province: If True, saves the data split by province. If False, saves the full dataset.
    :type split_by_province: bool
    :return: None


    Usage
    --------

    """

    # Check if folder path exists and is a string and create it if not
    if not folder_path or not isinstance(folder_path, str):
        raise ValueError("folder_path must be a non-empty string.")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    df = fetch_full_census()
    df_geom = enrich_with_geometry(df, geometry_path=geometry_path)

    if split_by_province:
        if "ine_province_code" not in df_geom.columns:
            raise ValueError("'ine_province_code' column is missing from the data.")

        for province_code in df_geom["ine_province_code"].unique():
            print(f"Processing province code: {province_code}")
            province_df = df_geom[df_geom["ine_province_code"] == province_code]
            province_file_name = f"{file_name}_province_{province_code}"
            save_geojson_file(province_df, folder_path, province_file_name)
    else:
        save_geojson_file(df_geom, folder_path, file_name)

        print(
            f"Full census data with geometry saved to {folder_path}/{file_name}.geojson"
        )


def read_geojson_file(file_path: str) -> gpd.GeoDataFrame:
    """
    Reads a GeoJSON file and returns it as a GeoDataFrame.

    :param file_path: Path to the GeoJSON file.
    :type file_path: str
    :return: A GeoDataFrame containing the data from the GeoJSON file.
    :rtype: gpd.GeoDataFrame

    Usage
    --------

    Read a GeoJSON file:

    >>> geo_df = read_geojson_file("path/to/your/file.geojson")
    >>> print(geo_df.head())
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    return gpd.read_file(file_path)


class INEAPIClient:
    """
    A client for retrieving and processing census data from the INE API.

    This class provides methods to fetch census data by section, clean it, and link it with municipality data.
    It also supports verbose logging for debugging purposes.

    :param verbose: If True, enables detailed logging for debugging.
    :type verbose: bool

    Usage
    --------

    Import the INEAPIClient class and create an instance:

    >>> from mango.clients.ine import INEAPIClient, fetch_full_census, list_table_codes
    >>> ine_api = INEAPIClient()
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.municipalities = self._get_municipalities()

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.WARNING)

    @staticmethod
    def _get_municipalities() -> pd.DataFrame:
        """
        Returns the municipalities DataFrame containing the name and the code as per INE.
        :return: DataFrame containing municipalities data
        :rtype: pd.DataFrame
        """
        content = _fetch_url(INE_CODES_URL)
        df = pd.read_excel(BytesIO(content), skiprows=1, header=0)
        df["ine_municipality_code"] = df["CPRO"].astype(str).str.zfill(2) + df[
            "CMUN"
        ].astype(str).str.zfill(3)
        df.rename({"NOMBRE": "ine_municipality_name"}, axis=1, inplace=True)

        df = df[
            [
                "CODAUTO",
                "CPRO",
                "CMUN",
                "ine_municipality_code",
                "ine_municipality_name",
            ]
        ]

        return df

    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by filtering out unwanted rows from the INE API response.

        :param df: DataFrame to clean
        :type df: pd.DataFrame
        :return: Cleaned DataFrame
        :rtype: pd.DataFrame
        """

        df = df.loc[
            df["Nombre"].str.contains("sección")
            & ~df["Nombre"].str.contains("Hombres|Mujeres|Española|Extranjera")
            & df["Nombre"].str.contains("Total")
            & df["Nombre"].str.contains("Todas las edades")
        ]

        matches = df["Nombre"].str.extract(r"^(.*?)\ssección\s(\d{5})", expand=True)
        df = df.assign(Name=matches[0], Code=matches[1])

        df["ine_population"] = df["Data"].apply(
            lambda x: x[0].get("Valor") if isinstance(x, list) and len(x) > 0 else None
        )

        df.rename(columns={"Name": "ine_municipality_name"}, inplace=True)

        df = df.merge(
            self.municipalities[["ine_municipality_code", "ine_municipality_name"]],
            how="left",
            left_on="ine_municipality_name",
            right_on="ine_municipality_name",
        )

        # Check if the merge was successful
        if df["ine_municipality_code"].isnull().any():
            # Print the municipalities that could not be matched
            unmatched_municipalities = df[df["ine_municipality_code"].isnull()][
                "ine_municipality_name"
            ].unique()

            raise ValueError(
                f"Some municipalities could not be matched to their INE code by name. Check the data. They are: {unmatched_municipalities}"
            )

        df.rename(
            columns={"Code": "ine_census_tract_code", "Provincia": "ine_province_name"},
            inplace=True,
        )

        df["ine_census_tract_code"] = df["ine_municipality_code"].astype(str) + df[
            "ine_census_tract_code"
        ].astype(str)
        df["ine_province_code"] = df["ine_municipality_code"].str[:2]
        df = df[
            [
                "ine_province_code",
                "ine_province_name",
                "ine_municipality_code",
                "ine_municipality_name",
                "ine_census_tract_code",
                "ine_population",
            ]
        ]

        return df

    def fetch_census_by_section(self, table_id: str | list[str]) -> pd.DataFrame:
        """
        WARNING: This method is unreliable due to INE naming inconsistencies.
        The best way to fetch this data is to fetch the full census.

        Fetches census data for specific province(s) from the INE API.

        This method retrieves data for one or more provinces based on their table IDs, cleans the data,
        and returns it as a DataFrame. Each table ID corresponds to a province.

        :param table_id: A single table ID or a list of table IDs representing provinces.
        :type table_id: str or list of str
        :return: A cleaned DataFrame containing census data for the specified provinces.
        :rtype: pd.DataFrame

        Usage
        --------

        Fetch census data for a specific province:

        >>> census = ine_api.fetch_census_by_section("69202")
        """
        if isinstance(table_id, str):
            table_id = [table_id]

        all_data = []
        for tid in table_id:
            if not _valid_province_code(tid):
                raise ValueError(
                    f"Invalid province code: {tid}. Call list_table_codes() for valid codes."
                )

            url = f"{BASE_API_URL}{tid}?nult=1"
            content = _fetch_url(url)

            data = json.loads(content)
            df = pd.json_normalize(data)

            province_name = next(
                (k for k, v in PROVINCE_CODES.items() if v == tid), None
            )
            df["Provincia"] = province_name

            df.to_json(f"data_{tid}.json", index=False)

            df = self._clean_table(df)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        return combined_df
