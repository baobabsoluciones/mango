import json
import logging
import os
import tempfile
import geopandas as gpd
import pandas as pd
from io import StringIO, BytesIO

import polars as pl
import requests
import zipfile

from mango.logging import get_configured_logger

NATIONAL_DATA_URLS = {
    "poblacion_sexo_pais_nacimiento_esp_ext": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65031.csv?nocab=1",
    "poblacion_sexo_nacionalidad_esp_ext": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65032.csv?nocab=1",
    "poblacion_sexo_pais_nacionalidad_principales": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65036.csv?nocab=1",
    "poblacion_sexo_pais_nacimiento_principales": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65035.csv?nocab=1",
    "poblacion_sexo_relacion_nacimiento_residencia": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65033.csv?nocab=1",
    "poblacion_sexo_edad_quinquenales": "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65034.csv?nocab=1",
}

API_BASE_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/"
DEFAULT_TABLE_CODES_PATH = "ine_table_codes.json"

GEOMETRY_DATA_URL = r"https://www.ine.es/prodyser/cartografia/seccionado_2024.zip"
INE_CODES_URL = "https://www.ine.es/daco/daco42/codmun/diccionario25.xlsx"

# configure logging level
logger = get_configured_logger(
    logger_type=__name__,
    log_console_level=logging.INFO,
    mango_color=True,
)


class INEData:

    def __init__(
            self,
            table_codes_json_path: str = DEFAULT_TABLE_CODES_PATH,
            verbose: bool = False,
    ) -> None:
        """
        Initializes the INEData with specified configurations.

        :param table_codes_json_path: Path to the JSON file containing table codes for provinces.
        :type table_codes_json_path: str
        :param verbose: If True, sets the logger to DEBUG level.
        :type verbose: bool
        """
        if verbose:
            logger.setLevel(logging.DEBUG)

        self.provincial_codes = self._load_provincial_codes(table_codes_json_path)
        self.municipalities = self._process_municipalities_dictionary()

        if not self.provincial_codes:
            logger.warning(
                f"Provincial table codes could not be loaded from {table_codes_json_path}. "
                "Fetching by province name and dataset title will not work if the file is missing or invalid."
            )

    def _load_provincial_codes(self, json_path):
        """
        Loads provincial table codes from a JSON file.

        :param json_path: Path to the JSON file containing table codes.
        :type json_path: str
        :return: Dictionary of provincial codes if successful, None otherwise.
        :rtype: dict or None
        """
        if not os.path.exists(json_path):
            logger.warning(f"Table codes file not found at {json_path}")
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {json_path}.")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading {json_path}: {e}")
            return None

    def _process_municipalities_dictionary(self) -> pd.DataFrame:
        """
        Fetches and processes the INE municipalities dictionary.

        :return: DataFrame containing municipalities data with codes and names.
        :rtype: pd.DataFrame
        """
        logger.info("Fetching municipalities data.")
        content = self._make_request(INE_CODES_URL).content
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

        logger.info("Municipalities data successfully fetched and processed.")
        return df

    def _make_request(self, url):
        """
        Makes a GET request to the given URL.

        :param url: The URL to make the request to.
        :type url: str
        :return: The response object if successful, None otherwise.
        :rtype: requests.Response or None
        """
        try:
            response = requests.get(url, timeout=30, allow_redirects=False)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for URL: {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for URL {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from URL {url}: {e}")
            return None

    def _save_raw_content(self, content, destination_folder, filename):
        """
        Saves raw content to a file.

        :param content: The content to save.
        :type content: str or bytes
        :param destination_folder: The folder to save the content to.
        :type destination_folder: str
        :param filename: The name of the file to save the content as.
        :type filename: str
        """
        if not destination_folder:
            return
        try:
            os.makedirs(destination_folder, exist_ok=True)
            file_path = os.path.join(destination_folder, filename)
            mode = "w" if isinstance(content, str) else "wb"
            encoding = "utf-8" if isinstance(content, str) else None
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            logger.info(f"Raw data saved to {file_path}")
        except IOError as e:
            logger.error(f"Error saving file {filename} to {destination_folder}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving {filename}: {e}")

    def _clean_csv_to_dataframe(self, csv_content_string, dataset_key, year=None):
        """
        Cleans the CSV content from INE into a polars DataFrame.

        :param csv_content_string: The CSV content as a string.
        :type csv_content_string: str
        :param dataset_key: The key identifying the dataset type (influences cleaning logic).
        :type dataset_key: str
        :param year: The year to filter the data for. If None, no filtering is applied.
        :type year: int, optional
        :return: Cleaned data as a polars DataFrame.
        :rtype: polars.DataFrame
        """
        try:
            df = None
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pl.read_csv(
                        StringIO(csv_content_string), separator=";", encoding=enc
                    )
                    logger.info(f"Successfully read CSV with encoding: {enc}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to parse CSV with encoding {enc}: {e}")

            if df is None:
                logger.error(
                    f"Could not parse CSV for {dataset_key} with attempted encodings."
                )
                return pl.DataFrame()

            df = df.drop_nulls()

            if df.columns[0] == "Total Nacional":
                df = df.drop(df.columns[0])

            df = df.filter(
                ~pl.col(df.columns[3]).str.contains("Total")
                & ~pl.col(df.columns[4]).str.contains("Total")
            )

            if year is not None:
                try:
                    year = int(year)
                    df = df.filter(pl.col("Periodo") == year)
                    if df.is_empty():
                        logger.warning(
                            f"No data found for year {year} in {dataset_key}."
                        )
                except ValueError:
                    logger.error(f"Year '{year}' is not a valid integer")
                    raise

            if "Total" in df.columns:
                df = df.with_columns(
                    pl.col("Total")
                    .str.replace_all(r"\.", "")
                    .str.replace_all(r"^$", "0")
                    .cast(pl.Int64)
                    .alias("population_count")
                )

            if "Periodo" in df.columns:
                df = df.with_columns(pl.col("Periodo").cast(pl.Int64))

            df = df.with_columns(
                pl.col("Provincias")
                .str.extract(r"(\d{2})", 1)
                .alias("ine_province_code"),
                pl.col("Provincias").str.slice(3).alias("ine_province_name"),
            )
            df = df.with_columns(
                pl.col("Municipios")
                .str.extract(r"(\d{5})", 1)
                .alias("ine_municipality_code"),
                pl.col("Municipios").str.slice(6).alias("ine_municipality_name"),
            )
            df = df.with_columns(
                pl.col("Secciones")
                .str.extract(r"(\d{10})", 1)
                .alias("ine_census_tract_code")
            )

            df = df.with_columns(pl.col("Sexo").alias("sex"))

            group_map = {
                "Lugar de nacimiento": "birth_country",
                "Nacionalidad": "nationality",
                "País de nacionalidad": "nationality",
                "País de nacimiento": "birth_country",
                "Relación entre lugar de nacimiento y lugar de residencia": "birth_residence_relation",
                "Edad": "age_group",
            }

            group_column = None
            for key, value in group_map.items():
                if key in df.columns:
                    group_column = key
                    break

            df = df.rename({group_column: group_map[group_column], "Periodo": "year"})

            df = df.join(
                pl.from_pandas(
                    self.municipalities[
                        ["ine_municipality_code", "ine_municipality_name"]
                    ]
                ),
                how="left",
                left_on="ine_municipality_name",
                right_on="ine_municipality_name",
            )

            df = df.select(
                [
                    "ine_province_code",
                    "ine_province_name",
                    "ine_municipality_code",
                    "ine_municipality_name",
                    "ine_census_tract_code",
                    "year",
                    "sex",
                    group_map[group_column],
                    "population_count",
                ]
            )

            if df.is_empty():
                logger.warning(
                    f"DataFrame is empty after initial load and drop_nulls for {dataset_key}."
                )
                return df

            return df

        except Exception as e:
            logger.error(
                f"An unexpected error occurred while cleaning CSV for {dataset_key}: {e}"
            )
            return pl.DataFrame()

    def _is_valid_table_code(self, table_code):
        """
        Validates the table code by checking if it exists in the provincial codes.

        :param table_code: The table code to validate.
        :type table_code: str
        :return: True if the table code is valid, False otherwise.
        :rtype: bool
        """
        if not isinstance(table_code, str) or not table_code.isdigit():
            logger.error(f"Invalid table code format: {table_code}")
            return False

        for province, datasets in self.provincial_codes.items():
            for dataset in datasets:
                if dataset["code"] == table_code:
                    logger.info(
                        f"Valid table code: {table_code} found in province '{province}' "
                        f"for dataset '{dataset['title']}'"
                    )
                    return True

        logger.error(f"Table code {table_code} not found in provincial codes.")
        return False

    def _get_title_by_code(self, target_code: str) -> str | None:
        """
        Retrieves the title of a dataset by its code.

        :param target_code: The code to look up.
        :type target_code: str
        :return: The title corresponding to the code, or None if not found.
        :rtype: str or None
        """
        for province, items in self.provincial_codes.items():
            for item in items:
                if item.get("code") == target_code:
                    return item.get("title")
        return None

    def _clean_api_data(self, raw_data, dataset_key):
        """
        Cleans the raw data retrieved from the API.

        :param raw_data: The raw data from the API.
        :type raw_data: dict
        :param dataset_key: The key identifying the dataset type.
        :type dataset_key: str
        :return: Cleaned data as a polars DataFrame.
        :rtype: polars.DataFrame
        """
        df = (
            pl.DataFrame(raw_data)
            .explode("Data")
            .filter(
                pl.col("Nombre").str.contains(r"\b\d{5}\b")
                & ~pl.col("Nombre").str.contains(
                    "Todas las edades"
                    if dataset_key == "Población por sexo y edad (grupos quinquenales)"
                    else "Total"
                )
            )
            .with_columns(
                [
                    pl.col("Nombre").str.split(by=". ").alias("split"),
                    pl.col("Data").struct.field("Valor").alias("population_count"),
                    pl.col("Data").struct.field("Anyo").alias("year"),
                ]
            )
            .with_columns(
                [
                    pl.col("population_count").cast(pl.Int64),
                    pl.col("split").list.get(0).alias("Section Name"),
                    pl.col("split").list.get(1).alias("sex"),
                ]
            )
            .with_columns(
                [
                    pl.col("Section Name")
                    .str.split_exact(" sección ", n=1)
                    .struct.field("field_0")
                    .alias("ine_municipality_name"),
                    pl.col("Section Name")
                    .str.split_exact(" sección ", n=1)
                    .struct.field("field_1")
                    .alias("ine_census_tract_code"),
                ]
            )
        )

        df = df.join(
            pl.from_pandas(
                self.municipalities[["ine_municipality_code", "ine_municipality_name"]]
            ),
            how="left",
            on="ine_municipality_name",
        )

        df = df.with_columns(
            pl.col("ine_municipality_code")
            .str.replace(" ", "")
            .alias("ine_municipality_code")
        )

        if dataset_key == "Población por sexo y nacionalidad (española/extranjera)":
            df = df.with_columns(
                [
                    pl.col("split")
                    .list.slice(2, 2)
                    .list.eval(
                        pl.element().filter(
                            pl.element().is_in(["Española", "Extranjera"])
                        )
                    )
                    .list.first()
                    .alias("nationality"),
                ]
            )

            desired_columns_in_order = [
                "ine_municipality_name",
                "ine_census_tract_code",
                "year",
                "sex",
                "nationality",
                "population_count",
            ]
            df = df.select(desired_columns_in_order)

        elif (
                dataset_key == "Población por sexo y país de nacimiento (España/extranjero)"
        ):
            df = df.with_columns(
                [
                    pl.col("split")
                    .list.slice(2, 2)
                    .list.eval(
                        pl.element().filter(
                            pl.element().is_in(["España", "Extranjera"])
                        )
                    )
                    .list.first()
                    .alias("birth_country"),
                ]
            )

            desired_columns_in_order = [
                "ine_municipality_name",
                "ine_census_tract_code",
                "year",
                "sex",
                "birth_country",
                "population_count",
            ]
            df = df.select(desired_columns_in_order)

        elif (
                dataset_key
                == "Población por sexo y país de nacionalidad (principales países)"
        ):
            df = df.with_columns(
                [
                    pl.col("split")
                    .list.slice(2, 2)
                    .list.eval(pl.element().filter(pl.element() != "Todas las edades"))
                    .list.first()
                    .alias("nationality"),
                ]
            )

            desired_columns_in_order = [
                "ine_municipality_name",
                "ine_census_tract_code",
                "year",
                "sex",
                "nationality",
                "population_count",
            ]
            df = df.select(desired_columns_in_order)

        elif (
                dataset_key
                == "Población por sexo y país de nacimiento (principales países)"
        ):
            df = df.with_columns(
                [
                    pl.col("split")
                    .list.slice(2, 2)
                    .list.eval(pl.element().filter(pl.element() != "Todas las edades"))
                    .list.first()
                    .alias("birth_country"),
                ]
            )

            desired_columns_in_order = [
                "ine_municipality_name",
                "ine_census_tract_code",
                "year",
                "sex",
                "birth_country",
                "population_count",
            ]
            df = df.select(desired_columns_in_order)

        elif (
                dataset_key
                == "Población por sexo y relación entre lugar de nacimiento y lugar de residencia"
        ):
            df = df.with_columns(
                pl.col("split").list.get(3).alias("birth_residence_relation"),
            )

            desired_columns_in_order = [
                "ine_municipality_name",
                "ine_census_tract_code",
                "year",
                "sex",
                "birth_residence_relation",
                "population_count",
            ]
            df = df.select(desired_columns_in_order)

        elif dataset_key == "Población por sexo y edad (grupos quinquenales)":
            df = df.with_columns(
                pl.col("split").list.get(2).alias("age_group"),
            )

            desired_columns_in_order = [
                "ine_municipality_name",
                "ine_census_tract_code",
                "year",
                "sex",
                "age_group",
                "population_count",
            ]
            df = df.select(desired_columns_in_order)

        else:
            logger.error(f"Unknown dataset key: {dataset_key}")

        return df

    def get_national_data(
            self, dataset_key, year=None, clean=True, download_folder=None
    ):
        """
        Fetches national data from a pre-defined list of CSV URLs.

        :param dataset_key: A key identifying the national dataset. Available keys include:
                          - poblacion_sexo_pais_nacimiento_esp_ext
                          - poblacion_sexo_nacionalidad_esp_ext
                          - poblacion_sexo_pais_nacionalidad_principales
                          - poblacion_sexo_pais_nacimiento_principales
                          - poblacion_sexo_relacion_nacimiento_residencia
                          - poblacion_sexo_edad_quinquenales
        :type dataset_key: str
        :param year: The year to filter the data by. If None, no filtering is applied.
        :type year: int, optional
        :param clean: If True, returns a cleaned polars DataFrame. If False, returns the raw CSV content as a string.
        :type clean: bool
        :param download_folder: If provided, saves the raw CSV to this folder with filename {dataset_key}.csv.
        :type download_folder: str, optional
        :return: Cleaned data as a polars DataFrame, raw CSV content as a string, or None on error.
        :rtype: polars.DataFrame or str or None

        Usage
        --------

        Fetch national data for a specific dataset key, e.g., "poblacion_sexo_pais_nacimiento_esp_ext":

        >>> from mango.clients.ine import INEData
        >>> ine = INEData()
        >>> df = ine.get_national_data("poblacion_sexo_pais_nacimiento_esp_ext", year=2023, clean=True)
        """
        url = NATIONAL_DATA_URLS.get(dataset_key)
        if not url:
            logger.error(
                f"Dataset key '{dataset_key}' not found in national URLS. Available keys: {list(NATIONAL_DATA_URLS.keys())}"
            )
            return None

        response = self._make_request(url)
        if not response:
            return None

        raw_csv_content = None
        encodings_to_try = [response.apparent_encoding, "utf-8", "latin-1", "cp1252"]

        for enc in encodings_to_try:
            if enc:
                try:
                    raw_csv_content = response.content.decode(enc)
                    logger.info(
                        f"Successfully decoded CSV for {dataset_key} with encoding: {enc}"
                    )
                    break
                except (UnicodeDecodeError, AttributeError) as e:
                    logger.warning(
                        f"Failed to decode CSV for {dataset_key} with {enc}: {e}"
                    )

        if raw_csv_content is None:
            logger.error(
                f"Could not decode CSV content for {dataset_key} from {url} with attempted encodings."
            )
            if download_folder:
                self._save_raw_content(
                    response.content,
                    download_folder,
                    f"{dataset_key}_error_raw_bytes.csv",
                )
            return None

        if download_folder:
            self._save_raw_content(
                raw_csv_content, download_folder, f"{dataset_key}.csv"
            )

        if clean:
            return self._clean_csv_to_dataframe(raw_csv_content, dataset_key, year)
        else:
            return raw_csv_content

    def clean_local_csv(self, file_path, dataset_key, year=None):
        """
        Cleans a locally stored CSV file using the same logic as for national data.

        The CSV must be the raw file from the INE website.

        :param file_path: The full path to the local CSV file.
        :type file_path: str
        :param dataset_key: A key to identify the dataset type (influences cleaning logic if specific).
                          For example: "poblacion_sexo_pais_nacimiento_esp_ext"
        :type dataset_key: str
        :param year: The year to filter the data for. If None, no filtering is applied.
        :type year: int, optional
        :return: Cleaned data as a Polars DataFrame, raw content as a string or bytes, or None on error.
        :rtype: polars.DataFrame or str or bytes or None

        Usage
        --------

        Clean a local CSV file that contains INE data, e.g., "poblacion_sexo_pais_nacimiento_esp_ext.csv":

        >>> from mango.clients.ine import INEData
        >>> ine = INEData()
        >>> df = ine.clean_local_csv("path/to/poblacion_sexo_pais_nacimiento_esp_ext.csv", "poblacion_sexo_pais_nacimiento_esp_ext", year=2023)
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None

        try:
            with open(file_path, "rb") as f:
                raw_csv_bytes = f.read()
                return self._clean_csv_to_dataframe(
                    raw_csv_bytes, dataset_key, year=year
                )
        except Exception as e:
            print(f"Error reading local file {file_path} as bytes: {e}")
            return None

    def list_table_codes(self):
        """
        Lists all available table codes and their titles.

        :return: A dictionary with province names as keys and lists of table code dictionaries as values.
                Each table code dictionary contains 'code' and 'title' keys.
        :rtype: dict

        Usage
        --------
        List all available table codes and their titles:

        >>> from mango.clients.ine import INEData
        >>> ine = INEData()
        >>> table_codes = ine.list_table_codes()
        >>> print(table_codes)

        """
        if not self.provincial_codes:
            logger.error("No provincial codes loaded. Cannot list table codes.")
            return {}

        return self.provincial_codes

    def get_data_by_table_code(self, table_code, clean=True, nlast=1):
        """
        Fetches data from the API using the specified table code.

        :param table_code: The table code to fetch data for. Use list_table_codes() to see available codes.
        :type table_code: str
        :param clean: If True, returns cleaned data as a polars DataFrame. If False, returns raw data as a dictionary.
        :type clean: bool
        :param nlast: Returns the nlast last periods/years.
        :type nlast: int
        :return: Cleaned data as a polars DataFrame, raw data as a dictionary, or None if an error occurs.
        :rtype: polars.DataFrame or dict or None

        Usage
        --------

        Fetch data for a specific table code, e.g., "69202":

        >>> from mango.clients.ine import INEData
        >>> ine = INEData()
        >>> data = ine.get_data_by_table_code("69202", clean=True, nlast=1)
        >>> print(data)


        """
        if not self._is_valid_table_code(table_code):
            logger.error(f"Invalid table code: {table_code}")
            return None

        url = f"{API_BASE_URL}{table_code}?nult={nlast}"
        response = self._make_request(url)
        if not response:
            return None

        try:
            raw_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON response for table code {table_code}: {e}"
            )
            return None

        if clean:
            return self._clean_api_data(raw_data, self._get_title_by_code(table_code))
        else:
            return raw_data

    def enrich_with_geometry(
            self, census_df: pd.DataFrame, geometry_path: str = None
    ) -> gpd.GeoDataFrame:
        """
        Enriches a census DataFrame with spatial geometry data.

        This function merges census data with geometrical data from a spatial file (e.g., shapefile or GeoJSON).
        If no file path is provided, it fetches the geometry data from a predefined URL.

        :param census_df: A cleaned census DataFrame to enrich with geometry. Will be converted to pandas if it's a polars DataFrame.
        :type census_df: pd.DataFrame or polars.DataFrame
        :param geometry_path: Path to the spatial file containing geometry data. If None, data is fetched from GEOMETRY_DATA_URL.
        :type geometry_path: str, optional
        :return: A GeoDataFrame containing both census and spatial data.
        :rtype: gpd.GeoDataFrame

        Usage
        --------

        Enrich previously fetched census data with the geometries assigned to the census tracts:

        >>> from mango.clients.ine import INEData
        >>> import geopandas as gpd

        >>> ine = INEData()
        >>> census_data = ine.get_national_data("poblacion_sexo_pais_nacimiento_esp_ext", year=2023, clean=True)
        >>> geometry_gdf = ine.enrich_with_geometry(census_df)

        """
        logger.info("Enriching census data with geometry.")
        if geometry_path:
            logger.debug(f"Using geometry file from path: {geometry_path}")
        else:
            logger.debug("Fetching geometry data from predefined URL.")

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

        if isinstance(census_df, pl.DataFrame):
            census_df = census_df.to_pandas()

        geometry_with_census = geometry_data.merge(
            census_df, left_on="CUSEC", right_on="ine_census_tract_code"
        )

        geometry_with_census = geometry_with_census[
            [col for col in geometry_with_census.columns if col != "geometry"]
            + ["geometry"]
        ]

        logger.info("Census data successfully enrich with geometry.")
        return geometry_with_census


# --- Example Usage ---
if __name__ == "__main__":
    ine = INEData(
        table_codes_json_path=r"../../sandbox/data/processed/ine_table_codes.json"
    )

    print(ine.get_data_by_table_code("69095", clean=True, nlast=1))
    print(
        ine.get_national_data(
            "poblacion_sexo_pais_nacimiento_esp_ext", year=2023, clean=True
        )
    )
