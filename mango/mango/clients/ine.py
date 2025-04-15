import requests
import pandas as pd
import os
from io import BytesIO
import geopandas as gpd
import logging
import json

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)
logger = logging.getLogger(__name__)


class INEAPIClient:
    """
    A client for fetching census data from the INE API.
    """

    pd.options.mode.copy_on_write = True

    INE_CODES_URL = "https://www.ine.es/daco/daco42/codmun/diccionario25.xlsx"
    BASE_API_URL = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA/"
    FULL_CENSUS_URL = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/65031.csv"
    CACHE_FILENAME = "census_data.csv"
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

    def __init__(self, verbose: bool = False, debug: bool = False):
        """
        Initialize the INEAPIClient.
        :param verbose: If True, print additional information (using logger.info)
        :type verbose: bool
        :param debug: If True, set the logging level to DEBUG for more detailed output
        :type debug: bool
        """
        self.verbose = verbose
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled: Logging level set to DEBUG.")
        else:
            logger.debug("Debug mode disabled.")
        logger.debug(f"Initializing INEAPIClient with verbose={verbose} and debug={debug}")
        self.municipalities = self._get_municipalities()
        logger.debug(f"Municipalities data loaded successfully. Shape: {self.municipalities.shape}")

    def _log(self, message: str) -> None:
        """
        Logs a message based on the verbose setting (now using logger directly).

        :param message: Message to log
        :type message: str
        """
        if self.verbose:
            logger.info(message)

    def _fetch_url(self, url: str) -> bytes:
        """
        Fetches content from a URL with error handling.

        :param url: URL to fetch
        :type url: str
        :return: Content of the response
        :rtype: bytes
        """
        self._log(f"Fetching data from URL: {url}")
        logger.debug(f"Fetching data from URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            logger.debug(f"Successfully fetched data from URL: {url}. Status code: {response.status_code}")
            return response.content
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {url}: {e}")
            logger.debug(f"Error details: {e}")
            raise

    def _get_municipalities(self) -> pd.DataFrame:
        """
        Returns the municipalities DataFrame containing the name and the code as per INE.
        :return: DataFrame containing municipalities data
        :rtype: pd.DataFrame
        """
        self._log("Fetching municipalities data from INE...")
        logger.debug(f"Fetching municipalities data from URL: {self.INE_CODES_URL}")
        content = self._fetch_url(self.INE_CODES_URL)

        self._log("Municipalities data fetched successfully.")
        logger.debug("Municipalities data fetched successfully.")
        self._log("Reading Excel file...")
        logger.debug("Reading Excel file into pandas DataFrame...")
        df = pd.read_excel(BytesIO(content), skiprows=1, header=0)
        logger.debug(f"Initial municipalities DataFrame shape: {df.shape}")
        df["ine_municipality_code"] = df["CPRO"].astype(str).str.zfill(2) + df["CMUN"].astype(
            str
        ).str.zfill(3)
        logger.debug("Created 'ine_municipality_code' column.")
        df.rename({"NOMBRE": "ine_municipality_name"}, axis=1, inplace=True)
        logger.debug("Renamed 'NOMBRE' column to 'ine_municipality_name'.")

        df = df[["CODAUTO", "CPRO", "CMUN", "ine_municipality_code", "ine_municipality_name"]]
        logger.debug(f"Selected final columns for municipalities DataFrame. Shape: {df.shape}")

        return df

    def list_table_codes(self) -> dict:
        """
        List all the table codes related to each province.
        :return: dict
        """
        logger.info("Listing available province codes.")
        return self.PROVINCE_CODES

    def _valid_province_code(self, province_code: str) -> bool:
        """
        Validates the province code.

        :param province_code: Province code to validate
        :type province_code: str
        :return: True if valid, False otherwise
        :rtype: bool
        """
        logger.debug(f"Validating province code: {province_code}")
        if isinstance(province_code, int):
            province_code = str(province_code)
            logger.debug(f"Province code converted to string: {province_code}")
        if len(province_code) != 5:
            raise ValueError("Table code must be a 5-digit string/integer.")
        if province_code not in self.PROVINCE_CODES.values():
            logger.debug(f"Province code {province_code} is not in the list of valid codes.")
            return False

        province_name = list(self.PROVINCE_CODES.keys())[list(self.PROVINCE_CODES.values()).index(province_code)]
        self._log(
            f"Province code {province_code} is valid and corresponds to {province_name}."
        )
        logger.debug(f"Province code {province_code} is valid and corresponds to {province_name}.")
        return True

    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by filtering out unwanted rows.

        :param df: DataFrame to clean
        :type df: pd.DataFrame
        :return: Cleaned DataFrame
        :rtype: pd.DataFrame
        """
        logger.info("Filtering data excluding aggregated values...")
        logger.debug(f"Initial shape of DataFrame before cleaning: {df.shape}")
        df = df.loc[
            df["Nombre"].str.contains("sección")
            & ~df["Nombre"].str.contains("Hombres|Mujeres|Española|Extranjera")
            & df["Nombre"].str.contains("Total")
            & df["Nombre"].str.contains("Todas las edades")
        ]
        logger.info("Extracting section names and codes...")
        logger.debug(f"Shape of DataFrame after initial filtering: {df.shape}")
        matches = df["Nombre"].str.extract(r"^(.*?)\ssección\s(\d{5})", expand=True)
        df = df.assign(Name=matches[0], Code=matches[1])
        logger.debug("Extracted section names and codes.")

        logger.info("Extracting census data from data dict...")
        logger.debug("Applying lambda function to extract 'Valor' from 'Data' column.")
        df["Total"] = df["Data"].apply(
            lambda x: x[0].get("Valor") if isinstance(x, list) and len(x) > 0 else None
        )
        logger.debug("Extracted total population.")

        logger.debug("Changing column name of name to ine_municipality_name...")
        df.rename(columns={"Name": "ine_municipality_name"}, inplace=True)

        logger.info("Adding ine code...")
        logger.debug("Merging with municipalities DataFrame...")
        df = df.merge(
            self.municipalities[["ine_municipality_code", "ine_municipality_name"]],
            how="left",
            left_on="ine_municipality_name",
            right_on="ine_municipality_name",
        )
        logger.debug(f"Shape of DataFrame after merging with municipalities: {df.shape}")

        # Check if the merge was successful
        if df["ine_municipality_code"].isnull().any():
            # Print the municipalities that could not be matched
            unmatched_municipalities = df[df["ine_municipality_code"].isnull()]["ine_municipality_name"].unique()
            logger.warning(
                f"Some municipalities could not be matched to their INE code by name. Check the data. They are: {unmatched_municipalities}"
            )
            logger.debug(f"Unmatched municipalities: {unmatched_municipalities}")
            raise ValueError(
                f"Some municipalities could not be matched to their INE code by name. Check the data. They are: {unmatched_municipalities}"
            )
        else:
            logger.debug("All municipalities matched successfully.")

        logger.info("Reordering columns...")
        df.rename(columns={"Code": "ine_census_tract"}, inplace=True)
        logger.debug("Renamed 'Code' column to 'ine_census_tract'.")
        df["ine_census_tract"] = df["ine_municipality_code"].astype(str) + df["ine_census_tract"].astype(str)
        logger.debug("Concatenated 'ine_municipality_code' and 'ine_census_tract'.")
        df = df[["Provincia", "ine_municipality_code", "ine_municipality_name", "ine_census_tract", "Total"]]
        logger.debug(f"Selected final columns for cleaned DataFrame. Shape: {df.shape}")

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
            logger.debug(f"Fetching census data for single table ID: {table_id}")
        else:
            logger.debug(f"Fetching census data for multiple table IDs: {table_id}")

        all_data = []
        for tid in table_id:
            logger.info(f"Fetching data for table ID: {tid}")
            if not self._valid_province_code(tid):
                raise ValueError(
                    f"Invalid province code: {tid}. Call list_table_codes() for valid codes."
                )

            url = f"{self.BASE_API_URL}{tid}?nult=1"
            logger.debug(f"Fetching data from URL: {url}")
            content = self._fetch_url(url)

            data = json.loads(content)
            df = pd.json_normalize(data)
            logger.info(f"Data fetched successfully for table ID {tid}.")
            logger.debug(f"Shape of raw data for table ID {tid}: {df.shape}")

            logger.info("Adding province name...")
            province_name = next(
                (k for k, v in self.PROVINCE_CODES.items() if v == tid), None
            )
            df["Provincia"] = province_name
            logger.debug(f"Added province name '{province_name}' to DataFrame.")

            df = self._clean_table(df)
            all_data.append(df)
            logger.debug(f"Cleaned data for table ID {tid}. Shape: {df.shape}")

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data for all requested tables. Shape: {combined_df.shape}")
        logger.debug(f"Combined DataFrame info:\n{combined_df.info()}")
        return combined_df

    def _clean_full_census(self, df: pd.DataFrame, year: int = None) -> pd.DataFrame:
        """
        Cleans the full census DataFrame by filtering out unwanted rows.

        :param df: DataFrame to clean
        :type df: pd.DataFrame
        :param year: Year to filter the data by, if None it will use the last year available in the data
        :type year: int
        :return: Cleaned DataFrame
        :rtype: pd.DataFrame
        """
        logger.info("Filtering full census data...")
        logger.debug(f"Initial shape of full census DataFrame: {df.shape}")
        df = df[
            df["Provincias"].notna()
            & df["Municipios"].notna()
            & df["Secciones"].notna()
            & (df["Sexo"] == "Total")
            & (df["Lugar de nacimiento"] == "Total")
        ]
        logger.debug(f"Shape after filtering non-null and 'Total' rows: {df.shape}")

        year_filter = year if year else df["Periodo"].max()
        logger.info(f"Filtering by year {year_filter}...")
        logger.debug(f"Filtering by year: {year_filter}")
        df = df[df["Periodo"] == year_filter]
        logger.debug(f"Shape after filtering by year {year_filter}: {df.shape}")

        logger.info("Extracting section names and codes...")
        df["Id_Provincia"] = df["Provincias"].str[:2]
        df["Provincias"] = df["Provincias"].str[3:]
        df["Id_Municipio"] = df["Municipios"].str[:5]
        df["Municipios"] = df["Municipios"].str[5:]
        df["Id_Seccion"] = df["Secciones"].str[:10]
        df["Secciones"] = df["Secciones"].str[10:]
        logger.debug("Extracted province, municipality, and section identifiers and names.")

        logger.info("Dropping unnecessary columns...")
        df.drop(columns=["Sexo", "Lugar de nacimiento", "Total Nacional"], inplace=True)
        logger.debug("Dropped unnecessary columns: 'Sexo', 'Lugar de nacimiento', 'Total Nacional'.")
        df = df[
            [
                "Id_Provincia",
                "Provincias",
                "Id_Municipio",
                "Municipios",
                "Id_Seccion",
                "Secciones",
                "Total",
            ]
        ]
        logger.debug(f"Selected final columns for full census DataFrame. Shape: {df.shape}")

        return df

    def fetch_full_census(self, cache: bool = False, year: int = None) -> pd.DataFrame:
        """
        Fetch all census data from the INE API.

        :param cache: If True, cache the data to a CSV file
        :type cache: bool
        :param year: Year to filter the data by, if None it will use the last year available in the data
        :type year: int
        :return: DataFrame containing all census data
        :rtype: pd.DataFrame
        """
        if cache and os.path.exists(self.CACHE_FILENAME):
            logger.info(f"Fetching data from cache at {self.CACHE_FILENAME}...")
            logger.debug(f"Checking for cache file at: {self.CACHE_FILENAME}")
            try:
                df = pd.read_csv(self.CACHE_FILENAME)
                logger.info("Data fetched from cache successfully.")
                logger.debug(f"Data loaded from cache. Shape: {df.shape}")
                return df
            except Exception as e:
                logger.error(f"Error reading cache file: {e}")
                logger.debug(f"Error details: {e}")
                logger.info("Cache file not found or invalid. Fetching data from the API.")
        else:
            logger.debug("Cache is disabled or file does not exist. Fetching from API.")

        logger.info("Fetching all census data from INE API...")
        logger.debug(f"Fetching full census data from URL: {self.FULL_CENSUS_URL}")
        content = self._fetch_url(self.FULL_CENSUS_URL)

        logger.info("All census data fetched successfully.")
        logger.debug("All census data fetched successfully.")
        df = pd.read_csv(BytesIO(content), sep=";")
        logger.debug(f"Raw full census data loaded. Shape: {df.shape}")
        df = self._clean_full_census(df, year)

        if cache:
            try:
                df.to_csv(self.CACHE_FILENAME, index=False)
                logger.info(f"Data cached successfully at {self.CACHE_FILENAME}.")
                logger.debug(f"Data saved to cache at: {self.CACHE_FILENAME}")
            except Exception as e:
                logger.error(f"Error caching data: {e}")
                logger.debug(f"Error details during caching: {e}")

        return df

    def _load_geometry(self, geometry_path: str) -> gpd.GeoDataFrame:
        """
        Loads the census section geometries from a spatial file. Requires the file to be downloaded.

        :param geometry_path: Path to the geometry file (GeoJSON, SHP, etc.)
        :type geometry_path: str
        :return: GeoDataFrame with geometries
        :rtype: gpd.GeoDataFrame
        """
        logger.info(f"Loading geometry from {geometry_path}...")
        logger.debug(f"Attempting to load geometry from path: {geometry_path}")
        try:
            gdf = gpd.read_file(geometry_path)
            logger.info("Geometry data loaded successfully.")
            logger.debug(f"Geometry data loaded successfully. CRS: {gdf.crs}")
            return gdf
        except Exception as e:
            logger.error(f"Error loading geometry data: {e}")
            logger.debug(f"Error details during geometry loading: {e}")
            raise

    def generate_geodataframe(self, table_id: str | list[str], geometry_path: str, year: int = None) -> gpd.GeoDataFrame:
        """
        Creates a density map by merging census data with geometries.

        :param table_id: ID(s) of the table(s) to retrieve
        :type table_id: str or list of str
        :param geometry_path: Path to the geometry file (GeoJSON, SHP, etc.)
        :type geometry_path: str
        :param year: Year to filter the data by, if None it will use the last year available in the data
        :type year: int
        :return: GeoDataFrame with density information
        :rtype: gpd.GeoDataFrame
        """
        logger.info("Fetching census data for density map...")
        census_data = self.fetch_census_by_section(table_id)
        logger.info("Census data fetched successfully.")
        logger.debug(f"Shape of fetched census data: {census_data.shape}")

        logger.info("Loading geometry data for density map...")
        geometry_data = self._load_geometry(geometry_path)
        logger.info("Geometry data loaded successfully.")
        logger.debug(f"Shape of loaded geometry data: {geometry_data.shape}")

        logger.info("Merging census and geometry data for density map...")
        logger.debug("Merging on 'CUSEC' from geometry and 'ine_census_tract' from census data.")
        merged_data = geometry_data.merge(census_data, left_on="CUSEC", right_on="ine_census_tract")
        logger.info("Data merged successfully.")
        logger.debug(f"Shape of merged data: {merged_data.shape}")


        return merged_data