import requests
import pandas as pd
import os
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
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

    def __init__(self, verbose: bool = False):
        """
        Initialize the INEAPIClient.
        :param verbose: If True, print additional information
        :type verbose: bool
        """
        self.verbose = verbose
        self.municipalities = self._get_municipalities()

    def _log(self, message: str) -> None:
        """
        Logs a message if verbose mode is enabled.

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
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {url}: {e}")
            raise

    def _get_municipalities(self) -> pd.DataFrame:
        """
        Returns the municipalities DataFrame containing the name and the code as per INE.
        :return: DataFrame containing municipalities data
        :rtype: pd.DataFrame
        """
        self._log("Fetching municipalities data from INE...")
        content = self._fetch_url(self.INE_CODES_URL)

        self._log("Municipalities data fetched successfully.")
        self._log("Reading Excel file...")
        df = pd.read_excel(BytesIO(content), skiprows=1, header=0)
        df["Codigo_INE"] = df["CPRO"].astype(str).str.zfill(2) + df["CMUN"].astype(
            str
        ).str.zfill(3)

        df = df[["CODAUTO", "CPRO", "CMUN", "Codigo_INE", "NOMBRE"]]

        return df

    def list_table_codes(self) -> dict:
        """
        List all the table codes related to each province.
        :return: dict
        """
        return self.PROVINCE_CODES

    def _valid_province_code(self, province_code: str) -> bool:
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
        if province_code not in self.PROVINCE_CODES.values():
            return False

        self._log(
            f"Province code {province_code} is valid and corresponds to {list(self.PROVINCE_CODES.keys())[list(self.PROVINCE_CODES.values()).index(province_code)]}."
        )
        return True

    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by filtering out unwanted rows.

        :param df: DataFrame to clean
        :type df: pd.DataFrame
        :return: Cleaned DataFrame
        :rtype: pd.DataFrame
        """
        self._log("Filtering data excluding aggregated values...")
        df = df.loc[
            df["Nombre"].str.contains("sección")
            & ~df["Nombre"].str.contains("Hombres|Mujeres|Española|Extranjera")
            & df["Nombre"].str.contains("Total")
            & df["Nombre"].str.contains("Todas las edades")
        ]
        self._log("Extracting section names and codes...")
        matches = df["Nombre"].str.extract(r"^(.*?)\ssección\s(\d{5})", expand=True)
        df = df.assign(Name=matches[0], Code=matches[1])

        self._log("Extracting census data from data dict...")
        df["Total"] = df["Data"].apply(
            lambda x: x[0].get("Valor") if isinstance(x, list) and len(x) > 0 else None
        )

        self._log("Adding ine code...")
        df = df.merge(
            self.municipalities[["Codigo_INE", "NOMBRE"]],
            how="left",
            left_on="Name",
            right_on="NOMBRE",
        )

        # Check if the merge was successful
        if df["Codigo_INE"].isnull().any():
            # Print the municipalities that could not be matched
            unmatched_municipalities = df[df["Codigo_INE"].isnull()]["Name"].unique()

            raise ValueError(
                f"Some municipalities could not be matched to their INE code by name. Check the data. They are: {unmatched_municipalities}"
            )

        self._log("Reordering columns...")
        df.rename(columns={"Code": "Seccion censal"}, inplace=True)
        df = df[["Provincia", "Codigo_INE", "Name", "Seccion censal", "Total"]]

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
            if not self._valid_province_code(tid):
                raise ValueError(
                    f"Invalid province code: {tid}. Call list_table_codes() for valid codes."
                )

            url = f"{self.BASE_API_URL}{tid}?nult=1"
            content = self._fetch_url(url)

            data = requests.get(url).json()
            df = pd.json_normalize(data)
            self._log(f"Data fetched successfully for table ID {tid}.")

            self._log("Adding province name...")
            df["Provincia"] = next(
                (k for k, v in self.PROVINCE_CODES.items() if v == tid), None
            )

            df = self._clean_table(df)
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)

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
        self._log("Filtering data excluding aggregated values...")
        df = df[
            df["Provincias"].notna()
            & df["Municipios"].notna()
            & df["Secciones"].notna()
            & (df["Sexo"] == "Total")
            & (df["Lugar de nacimiento"] == "Total")
        ]

        year_filter = year if year else df["Periodo"].max()
        self._log(f"Filtering by year {year_filter}...")
        df = df[df["Periodo"] == year_filter]

        self._log("Extracting section names and codes...")
        df["Id_Provincia"] = df["Provincias"].str[:2]
        df["Provincias"] = df["Provincias"].str[3:]
        df["Id_Municipio"] = df["Municipios"].str[:5]
        df["Municipios"] = df["Municipios"].str[5:]
        df["Id_Seccion"] = df["Secciones"].str[:10]
        df["Secciones"] = df["Secciones"].str[10:]

        self._log("Dropping unnecessary columns...")
        df.drop(columns=["Sexo", "Lugar de nacimiento", "Total Nacional"], inplace=True)
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
            self._log(f"Fetching data from cache at {self.CACHE_FILENAME}...")
            try:
                df = pd.read_csv(self.CACHE_FILENAME)
                self._log("Data fetched from cache successfully.")
                return df
            except Exception as e:
                logger.error(f"Error reading cache file: {e}")
                self._log("Cache file not found or invalid. Fetching data from the API.")

        content = self._fetch_url(self.FULL_CENSUS_URL)

        self._log("All census data fetched successfully.")
        df = pd.read_csv(BytesIO(content), sep=";")
        df = self._clean_full_census(df, year)

        if cache:
            try:
                df.to_csv(self.CACHE_FILENAME, index=False)
                self._log(f"Data cached successfully at {self.CACHE_FILENAME}.")
            except Exception as e:
                logger.error(f"Error caching data: {e}")

        return df
