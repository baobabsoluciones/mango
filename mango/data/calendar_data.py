from datetime import datetime

import holidays
import pandas as pd
import pycountry
import unidecode
from holidays import country_holidays


def get_calendar(
    country: str = "ES",
    start_year: int = 2010,
    end_year: int = datetime.now().year + 2,
    communities: bool = False,
    calendar_events: bool = False,
    name_transformations: bool = True,
):
    """
    The get_calendar function returns a pandas DataFrame with the following columns:
        - date: Date of the holiday.
        - name: Name of the holiday.
        - country_code: Country code (ISO 3166-2). Only if communities=True
        - community_code (optional): Autonomous Community code (ISO 3166-2) name. Only if communities=True.

    :param str country: Specify the country code
    :param int start_year: Set the first year of the calendar
    :param int end_year: Set the end year of the calendar
    :param bool communities: Include the holidays of each autonomous community in spain
    :param bool calendar_events: Add black friday to the calendar.
    :param bool name_transformations: Apply transformations to the names of the holidays
    :return: A pandas DataFrame with the calendar
    :doc-author: baobab soluciones
    """
    # Checks
    if start_year > end_year:
        raise ValueError("start_year must be lower than end_year")

    # List of years
    years = list(range(start_year, end_year))

    # Get national holidays
    country_holidays_dict = country_holidays(
        country=country,
        years=years,
        language="en",
    )

    # Dict to DataFrame
    df = pd.DataFrame.from_dict(country_holidays_dict, orient="index").reset_index()
    df["country_code"] = country
    # Rename columns
    df.columns = ["date", "name", "country_code"]

    # TODO: Add more dates that are not in the holidays package (e.g. 11-11 AliExpress, etc.)
    if calendar_events:
        # Add Black Friday
        usa_holidays_dict = country_holidays(
            country="US",
            years=years,
            language="es",
        )
        df_usa = pd.DataFrame.from_dict(usa_holidays_dict, orient="index").reset_index()
        df_usa["country_code"] = country
        # Rename columns
        df_usa.columns = ["date", "name", "country_code"]

        # Filter by "Thanksgiving"
        df_usa = df_usa[df_usa["name"].str.contains("Thanksgiving")]
        # Rename to Black Friday
        df_usa["name"] = "Black Friday"

        # Add to df
        df = pd.concat([df, df_usa])

    # Add jueves santo
    if country == "ES":
        # Get viernes santo
        df_viernes_santo = df[df["name"].str.contains("Viernes Santo")]
        # Rest one day
        df_viernes_santo["date"] = df_viernes_santo["date"] - pd.Timedelta(days=1)
        # Rename to Jueves Santo
        df_viernes_santo["name"] = "Jueves Santo"
        # Add to df
        df = pd.concat([df, df_viernes_santo])

    # Add communities holidays
    if communities:
        code_name_dict = _get_code_name_dict(country=country)
        list_com = []
        for community in holidays.ES.subdivisions:
            # Autonomous Community holidays
            com_holidays = country_holidays("ES", years=years, prov=community)
            # Dict to DataFrame
            df_com = pd.DataFrame.from_dict(com_holidays, orient="index").reset_index()
            df_com["country_code"] = country
            df_com["community_code"] = community
            df_com["community_name"] = code_name_dict[f"{country}-{community}"]
            list_com.append(df_com)
        df_com = pd.concat(list_com)

        # Rename columns
        df_com.columns = [
            "date",
            "name",
            "country_code",
            "community_code",
            "community_name",
        ]

        # Drop from df_com the holidays that are in df
        df_com = (
            df_com.merge(
                df, on=["date", "name", "country_code"], how="left", indicator=True
            )
            .query('_merge == "left_only"')
            .drop("_merge", axis=1)
        )

        # Concatenate dataframes
        df = pd.concat([df, df_com])

    # Sort by date
    df = df.sort_values(by="date").reset_index(drop=True)

    # Apply name transformations
    if name_transformations:
        df = _name_transformations(df)

    # Return dataframe
    return df


def _get_code_name_dict(country: str) -> dict:
    """
    The _get_code_name_dict function takes a country code as an argument and returns a dictionary of the state/province codes and names for that country.

    :param country: str: Specify the country that we want to get the subdivisions for
    :return: A dictionary that maps a subdivision code to its name
    :doc-author: baobab soluciones
    """
    return {
        subdivision.code: subdivision.name
        for subdivision in pycountry.subdivisions.get(country_code=country)
    }


def _name_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    The _name_transformations function performs the following transformations on the name column:
        1. Drop special characters in column name
        2. Replace characters with accents in columns Name
        3. Drop non-ascii characters
        4. Remove special characters in column name (except for whitespace)
        5. Drop extra whitespace in column name
        6. Remove "observed" and "trasladado" in column name

    :param pd.DataFrame df: Specify the dataframe that will be used in the function
    :return: A dataframe with the name column transformed
    :doc-author: baobab soluciones
    """

    # Drop special characters in column name
    df["name"] = df["name"].str.replace("\n", " ", regex=True)
    df["name"] = df["name"].str.replace("\r", " ", regex=True)
    df["name"] = df["name"].str.replace("\t", " ", regex=True)

    # Replace characters with accents in columns Name
    df["name"] = df["name"].apply(unidecode.unidecode)

    # Drop non-ascii characters
    df["name"] = df["name"].str.replace("[^\x00-\x7F]+", "", regex=True)

    # Remove special characters in column name
    df["name"] = df["name"].str.replace(r"[^\w\s]", "", regex=True)

    # Drop extra whitespace in column name
    df["name"] = df["name"].str.strip()

    # Remove consecutive whitespaces in column name
    df["name"] = df["name"].str.replace(" +", " ", regex=True)

    # Remove string "observed" and "trasladado" in column name
    df["name"] = df["name"].str.replace("observed", "")
    df["name"] = df["name"].str.replace("Observed", "")
    df["name"] = df["name"].str.replace("Trasladado", "")
    df["name"] = df["name"].str.replace("trasladado", "")

    # Drop extra whitespace in column name
    df["name"] = df["name"].str.strip()

    return df