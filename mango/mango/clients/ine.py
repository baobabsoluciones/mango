"""
Module for obtaining census data from the Spanish National Statistics Institute (INE).

This module provides the INEAPIClient class for retrieving and processing
census data from the INE API.

Examples
--------
Import the INEAPIClient class and create an instance:
>>> from mango.clients.ine import INEAPIClient, fetch_full_census, list_table_codes
>>> ine_api = INEAPIClient()

Show all available table codes in the INE API:
>>> print(list_table_codes())

Fetch census data for a specific province:
>>> census_69202 = ine_api.fetch_census_by_section("69202")

Fetch full census and add the geometry data:
>>> full_census = fetch_full_census()
>>> geometry_data = enrich_with_geometry(census_69202)
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
    df["ine_census_tract_name"] = df["Secciones"].str[10:]

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
        "ine_census_tract_name",
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


def list_table_codes() -> dict:
    """
    List all the table codes related to each province.
    :return: dict
    """
    return PROVINCE_CODES


def fetch_full_census(year: int = None) -> pd.DataFrame:
    """
    Fetch all census data from the INE API.

    :param year: Year to filter the data by, if None, it will use the last year available in the data
    :type year: int
    :return: DataFrame containing all census data
    :rtype: pd.DataFrame
    """

    content = _fetch_url(FULL_CENSUS_URL)

    df_raw = pd.read_csv(BytesIO(content), sep=";", dtype=str, thousands=".")
    df = _clean_full_census(df_raw, year)

    return df


def enrich_with_geometry(census_df: pd.DataFrame, geometry_path: str = None) -> gpd.GeoDataFrame:
    """
    Enrich census DataFrame with geometrical data from a spatial file.

    :param census_df: Cleaned census data from INEAPIClient
    :type census_df: pd.DataFrame
    :param geometry_path: Path to the geometry file (e.g., GeoJSON, SHP)
    :type geometry_path: str
    :return: GeoDataFrame containing both census and spatial data
    :rtype: gpd.GeoDataFrame
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
            shp_file = next((f for f in zf.namelist() if f.endswith('.shp')), None)
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
        census_df, left_on="CUSEC", right_on="ine_census_tract"
    )

    return geometry_with_census


class INEAPIClient:
    """
    A client for fetching census data from the INE API.

    :param verbose: If True, print additional information (using logger.info)
    :type verbose: bool
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
        df["ine_municipality_code"] = df["CPRO"].astype(str).str.zfill(2) + df["CMUN"].astype(
            str
        ).str.zfill(3)
        df.rename({"NOMBRE": "ine_municipality_name"}, axis=1, inplace=True)

        df = df[["CODAUTO", "CPRO", "CMUN", "ine_municipality_code", "ine_municipality_name"]]

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
            unmatched_municipalities = df[df["ine_municipality_code"].isnull()]["ine_municipality_name"].unique()

            raise ValueError(
                f"Some municipalities could not be matched to their INE code by name. Check the data. They are: {unmatched_municipalities}"
            )

        df.rename(
            columns={
                "Code": "ine_census_tract",
                "Provincia": "province_name"},
            inplace=True
        )

        df["ine_census_tract"] = df["ine_municipality_code"].astype(str) + df["ine_census_tract"].astype(str)
        df = df[["province_name", "ine_municipality_code", "ine_municipality_name", "ine_census_tract", "ine_population"]]

        return df

    def fetch_census_by_section(self, table_id: str | list[str]) -> pd.DataFrame:
        """
        Get the data from a specific table or multiple tables in the INE API. Each table corresponds to a province.

        :param table_id: ID(s) of the table(s) to retrieve
        :type table_id: str or list of str
        :return: DataFrame containing the data from the table(s)
        :rtype: pd.DataFrame
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
