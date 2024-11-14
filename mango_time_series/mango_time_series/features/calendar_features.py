import warnings
from datetime import datetime

import holidays
import numpy as np
import pandas as pd
import pycountry
import unidecode
from holidays import country_holidays

_WEIGHT_DICT = {
    "AN": 1,
    "AR": 1,
    "AS": 1,
    "CB": 1,
    "CE": 1,
    "CL": 1,
    "CM": 1,
    "CN": 1,
    "CT": 1,
    "EX": 1,
    "GA": 1,
    "IB": 1,
    "MC": 1,
    "MD": 1,
    "ML": 1,
    "NC": 1,
    "PV": 1,
    "RI": 1,
    "VC": 1,
}


def _calculate_distances(df, distances_config):
    """
    The _calculate_distances function calculates the distance of each date to the next holiday. It takes a
    dataframe and a dictionary with the configuration for the distance calculation as arguments.

    :param pd.DataFrame df: Specify the dataframe that will be used in the function
    :param dict distances_config: Specify the configuration for the distance calculation
    :return: A dataframe with the distance of each date to the next holiday
    """
    tmp = df.copy()
    back_days = distances_config["steps_back"]
    forward_days = distances_config["steps_forward"]
    days = [i for i in range(-back_days, forward_days + 1)]
    days_df = pd.DataFrame(days, columns=["distance"])
    tmp = tmp.merge(days_df, how="cross")
    tmp["date"] = tmp["date"] + pd.to_timedelta(tmp["distance"], unit="D")
    return tmp


def get_calendar(
    country: str = "ES",
    start_year: int = 2010,
    end_year: int = datetime.now().year + 2,
    communities: bool = False,
    weight_communities: dict = None,
    calendar_events: bool = False,
    return_weights: bool = None,
    return_distances: bool = False,
    distances_config: dict = None,
    name_transformations: bool = True,
    pivot: bool = False,
    pivot_keep_communities: bool = False,
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
    :param dict weight_communities: Dictionary with the weight of each community
    :param bool return_weights: Return the weight of each date
    :param bool return_distances: Return the distance of each date to the next holiday
    :param dict distances_config: Configuration for the distance calculation. Dictionary with the following keys:
        - steps_back: Number of steps back to calculate the distance
        - steps_forward: Number of steps forward to calculate the distance
    :param bool pivot: Pivot the calendar to get a column for each date
    :param bool pivot_keep_communities: Keep the communities column when pivoting
    :return: A pandas DataFrame with the calendar
    :doc-author: baobab soluciones
    """
    if weight_communities is None:
        weight_communities = _WEIGHT_DICT
    # Checks
    if start_year > end_year:
        raise ValueError("start_year must be lower than end_year")

    if return_weights is None:
        return_weights = communities

    if return_weights and not communities:
        return_weights = False
        warnings.warn(
            "return_weights is True but communities is False. Setting return_weights to False"
        )

    if (return_weights + return_distances == 0) and communities:
        raise ValueError(
            "At least one of return_weights or return_distances must be True when communities is True"
        )

    # List of years
    years = list(range(start_year, end_year))

    # Get national holidays
    country_holidays_dict = country_holidays(
        country=country,
        years=years,
        language="es",
    )

    # Dict to DataFrame
    df = pd.DataFrame.from_dict(country_holidays_dict, orient="index").reset_index()
    df["country_code"] = country
    # Rename columns
    df.columns = ["date", "name", "country_code"]

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

    # Add communities holidays
    if communities:
        code_name_dict = _get_code_name_dict(country=country)
        list_com = []
        for community in holidays.ES.subdivisions:
            # Autonomous Community holidays
            com_holidays = country_holidays(
                "ES", years=years, subdiv=community, language="es", observed=False
            )
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

        # Add weight column
        df_com["weight"] = df_com["community_code"].map(weight_communities)
        total_sum = sum(weight_communities.values())
        # Add new column with the sum of weights grouping by date and name
        df_com["weight"] = (
            df_com.groupby(["date", "name"])["weight"].transform("sum") / total_sum
        )

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

        # Fill na
        df["weight"] = df["weight"].fillna(1)

    # Sort by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # Apply name transformations
    if name_transformations:
        df = _name_transformations(df)

    if return_distances:
        if distances_config is None:
            raise ValueError(
                "distances_config must be provided when return_distances is True"
            )
        df = _calculate_distances(df, distances_config)

    # Filter to keep only the columns that are needed
    if not return_weights and "weight" in df.columns:
        df = df.drop(columns=["weight"])

    # Pivot
    if pivot:
        df = _pivot_calendar(
            df_calendar=df,
            pivot_keep_communities=pivot_keep_communities,
        )

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
    list_replaces = [
        "observed",
        "trasladado",
        "estimated",
        "estimado",
    ]
    for x in list_replaces:
        df["name"] = df["name"].str.replace(x, "", case=False)

    # Drop extra whitespace in column name
    df["name"] = df["name"].str.strip()

    return df


def _pivot_calendar(
    df_calendar: pd.DataFrame, pivot_keep_communities: bool = False
) -> pd.DataFrame:
    """
    The pivot_calendar function takes a dataframe of calendar events and pivots it to get a column for each date.

    :param pd.DataFrame df_calendar: Specify the dataframe that will be used in this function
    :param bool communities: Specify if the dataframe has a column with the weight of each date
    :return: A dataframe with a column for each date
    :doc-author: baobab soluciones
    """

    # Check weight and distance columns in df_calendar
    available_columns = list(
        set(df_calendar.columns).intersection(["weight", "distance"])
    )

    # Pivot df_calendar to get a column for each date
    for col in available_columns:
        df_calendar = pd.concat(
            [
                df_calendar,
                pd.get_dummies(df_calendar["name"], prefix=col, prefix_sep="_"),
            ],
            axis=1,
        )
    df_calendar.drop(["name"], axis=1, inplace=True)

    # Group by fecha max()
    if pivot_keep_communities:
        groupby_cols = ["date", "country_code", "community_name"]
        df_calendar["community_name"] = df_calendar["community_name"].fillna("Nacional")
    else:
        groupby_cols = ["date"]

    # Multiply by weight
    for prefix in available_columns:
        for col in df_calendar.columns:
            if col.startswith(f"{prefix}_"):
                df_calendar[col] = np.where(
                    df_calendar[col],
                    df_calendar[col] * df_calendar[prefix],
                    0 if prefix == "weight" else np.nan,
                )

        df_calendar.drop([prefix], axis=1, inplace=True)

    df_calendar = df_calendar.groupby(groupby_cols).max(numeric_only=True).reset_index()

    # Replace back to null values in Nacional
    if pivot_keep_communities:
        df_calendar["community_name"] = df_calendar["community_name"].replace(
            "Nacional", np.nan
        )

    if len(available_columns) == 1:
        # Remove prefix if only one column
        df_calendar.columns = df_calendar.columns.str.replace(
            f"{available_columns[0]}_", ""
        )

    return df_calendar
