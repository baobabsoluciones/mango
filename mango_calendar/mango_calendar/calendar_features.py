"""Calendar features."""

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


def _calculate_distances(df: pd.DataFrame, distances_config: dict) -> pd.DataFrame:
    """
    Calculate the distance of each date to the next holiday.

    This function takes a dataframe containing holiday dates and a configuration
    dictionary to calculate the distance (in days) from each date to the next holiday.
    It creates a cross join with a range of days around each holiday date.

    :param df: DataFrame containing holiday dates and information
    :type df: pd.DataFrame
    :param distances_config: Configuration dictionary with 'steps_back' and 'steps_forward' keys
    :type distances_config: dict
    :return: DataFrame with expanded date ranges and distance calculations
    :rtype: pd.DataFrame
    :raises KeyError: If required keys are missing from distances_config

    Example:
        >>> config = {'steps_back': 7, 'steps_forward': 7}
        >>> df = pd.DataFrame({'date': ['2023-12-25'], 'name': ['Christmas']})
        >>> result = _calculate_distances(df, config)
        >>> print(result['distance'].min(), result['distance'].max())
        -7 7
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
    weight_communities: dict | None = None,
    calendar_events: bool = False,
    return_weights: bool | None = None,
    return_distances: bool = False,
    distances_config: dict | None = None,
    name_transformations: bool = True,
    pivot: bool = False,
    pivot_keep_communities: bool = False,
) -> pd.DataFrame:
    """
    Get a comprehensive calendar DataFrame with holiday information.

    This function generates a pandas DataFrame containing holiday information for a
    specified country and date range. It can include national holidays, regional
    community holidays (for Spain), and custom calendar events like Black Friday.
    The function supports various transformations and output formats.

    The resulting DataFrame contains the following columns:
        - date: Date of the holiday
        - name: Name of the holiday
        - country_code: Country code (ISO 3166-2)
        - community_code: Autonomous Community code (ISO 3166-2), only if communities=True
        - community_name: Name of the autonomous community, only if communities=True
        - weight: Weight of the holiday, only if return_weights=True
        - distance: Distance to holiday in days, only if return_distances=True

    :param country: Country code (ISO 3166-2) for which to retrieve holidays
    :type country: str
    :param start_year: First year to include in the calendar
    :type start_year: int
    :param end_year: Last year to include in the calendar (exclusive)
    :type end_year: int
    :param communities: Whether to include autonomous community holidays (Spain only)
    :type communities: bool
    :param weight_communities: Dictionary mapping community codes to their weights
    :type weight_communities: dict | None
    :param calendar_events: Whether to add Black Friday to the calendar
    :type calendar_events: bool
    :param return_weights: Whether to return holiday weights (auto-set if communities=True)
    :type return_weights: bool | None
    :param return_distances: Whether to return distance calculations to holidays
    :type return_distances: bool
    :param distances_config: Configuration for distance calculations with 'steps_back' and 'steps_forward'
    :type distances_config: dict | None
    :param name_transformations: Whether to apply text transformations to holiday names
    :type name_transformations: bool
    :param pivot: Whether to pivot the calendar to have columns for each holiday
    :type pivot: bool
    :param pivot_keep_communities: Whether to keep community information when pivoting
    :type pivot_keep_communities: bool
    :return: DataFrame containing calendar information with holidays and metadata
    :rtype: pd.DataFrame
    :raises ValueError: If start_year > end_year or invalid parameter combinations
    :raises UserWarning: If return_weights is requested without communities=True

    Example:
        >>> # Get basic Spanish calendar for 2023
        >>> calendar = get_calendar(country="ES", start_year=2023, end_year=2024)
        >>> print(calendar.columns.tolist())
        ['date', 'name', 'country_code']
        >>>
        >>> # Get calendar with community holidays and weights
        >>> calendar_with_communities = get_calendar(
        ...     country="ES",
        ...     start_year=2023,
        ...     end_year=2024,
        ...     communities=True,
        ...     return_weights=True
        ... )
        >>> print('weight' in calendar_with_communities.columns)
        True
    """
    if weight_communities is None:
        weight_communities = _WEIGHT_DICT

    if start_year > end_year:
        raise ValueError("start_year must be lower than end_year")

    if return_weights is None:
        return_weights = communities

    if return_weights and not communities:
        return_weights = False
        warnings.warn(
            "return_weights requires communities=True. Setting return_weights to False"
        )

    if (return_weights + return_distances == 0) and communities:
        raise ValueError(
            "return_weights or return_distances required when communities=True"
        )

    years = list(range(start_year, end_year))

    country_holidays_dict = country_holidays(
        country=country,
        years=years,
        language="es",
    )

    df = pd.DataFrame.from_dict(country_holidays_dict, orient="index").reset_index()
    df["country_code"] = country
    df.columns = ["date", "name", "country_code"]

    if calendar_events:
        df = _add_black_friday(country, years, df)

    if communities:
        df = _add_communities_holidays(country, weight_communities, years, df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

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

    if pivot:
        df = _pivot_calendar(
            df_calendar=df,
            pivot_keep_communities=pivot_keep_communities,
        )

    return df


def _add_communities_holidays(
    country: str, weight_communities: dict, years: list[int], df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add autonomous community holidays to the calendar DataFrame.

    This function retrieves holidays for each autonomous community in Spain and
    adds them to the existing calendar DataFrame. It calculates weighted holidays
    based on community populations and filters out duplicates with national holidays.

    :param country: Country code for which to add community holidays (typically "ES")
    :type country: str
    :param weight_communities: Dictionary mapping community codes to their population weights
    :type weight_communities: dict
    :param years: List of years for which to retrieve community holidays
    :type years: list[int]
    :param df: Base calendar DataFrame to which community holidays will be added
    :type df: pd.DataFrame
    :return: DataFrame with community holidays added, including weight calculations
    :rtype: pd.DataFrame
    :raises KeyError: If community codes in weight_communities are not found

    Example:
        >>> base_df = pd.DataFrame({
        ...     'date': ['2023-01-01'],
        ...     'name': ['New Year'],
        ...     'country_code': ['ES']
        ... })
        >>> weights = {'CT': 1.0, 'MD': 1.0}
        >>> result = _add_communities_holidays('ES', weights, [2023], base_df)
        >>> print('community_code' in result.columns)
        True
    """
    code_name_dict = _get_code_name_dict(country=country)
    list_com = []
    for community in holidays.ES.subdivisions:  # type: ignore
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

    df = pd.concat([df, df_com])

    df["weight"] = df["weight"].fillna(1)

    return df


def _add_black_friday(country: str, years: list[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Black Friday dates to the calendar DataFrame.

    This function retrieves Thanksgiving dates from US holidays and adds Black Friday
    (the day after Thanksgiving) to the calendar. Black Friday is calculated as
    Thanksgiving + 1 day for each year specified.

    :param country: Country code for which to add Black Friday (will be set in country_code column)
    :type country: str
    :param years: List of years for which to calculate Black Friday dates
    :type years: list[int]
    :param df: Calendar DataFrame to which Black Friday will be added
    :type df: pd.DataFrame
    :return: DataFrame with Black Friday dates added
    :rtype: pd.DataFrame

    Example:
        >>> base_df = pd.DataFrame({
        ...     'date': ['2023-01-01'],
        ...     'name': ['New Year'],
        ...     'country_code': ['ES']
        ... })
        >>> result = _add_black_friday('ES', [2023], base_df)
        >>> black_friday = result[result['name'] == 'Black Friday']
        >>> print(len(black_friday) > 0)
        True
    """
    usa_holidays_dict = country_holidays(
        country="US",
        years=years,
        language="es",
    )
    df_usa = pd.DataFrame.from_dict(usa_holidays_dict, orient="index").reset_index()
    df_usa["country_code"] = country
    df_usa.columns = ["date", "name", "country_code"]

    # Filter by "Thanksgiving", add on day to all the dates
    # (black friday is the day after thanksgiving)
    df_usa = df_usa[df_usa["name"].str.contains("Thanksgiving")]
    df_usa["date"] = df_usa["date"] + pd.to_timedelta(1, unit="D")
    df_usa["name"] = "Black Friday"

    df = pd.concat([df, df_usa])
    return df


def _get_code_name_dict(country: str) -> dict:
    """
    Get a dictionary mapping subdivision codes to their names for a country.

    This function uses the pycountry library to retrieve administrative subdivisions
    (states, provinces, autonomous communities, etc.) for a given country and
    returns a dictionary mapping their codes to their names.

    :param country: ISO 3166-1 alpha-2 country code (e.g., 'ES', 'US', 'FR')
    :type country: str
    :return: Dictionary mapping subdivision codes to subdivision names
    :rtype: dict
    :raises LookupError: If the country code is not found in pycountry database

    Example:
        >>> code_dict = _get_code_name_dict('ES')
        >>> print('ES-CT' in code_dict)
        True
        >>> print(code_dict['ES-CT'])
        Cataluña
    """
    return {
        subdivision.code: subdivision.name
        for subdivision in pycountry.subdivisions.get(country_code=country)  # type: ignore
    }


def _name_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply comprehensive text transformations to holiday names.

    This function performs multiple text cleaning operations on the 'name' column
    to standardize holiday names. The transformations include:

    1. Replace newline, carriage return, and tab characters with spaces
    2. Remove accents and diacritical marks using unidecode
    3. Remove non-ASCII characters
    4. Remove special characters except alphanumeric and whitespace
    5. Strip leading and trailing whitespace
    6. Replace multiple consecutive spaces with single spaces
    7. Remove common holiday status words (observed, trasladado, estimated, estimado)

    :param df: DataFrame containing a 'name' column with holiday names to transform
    :type df: pd.DataFrame
    :return: DataFrame with transformed holiday names in the 'name' column
    :rtype: pd.DataFrame
    :raises KeyError: If the 'name' column is not present in the DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'name': ['Día de la Constitución (observed)', 'Navidad\n\r']
        ... })
        >>> result = _name_transformations(df)
        >>> print(result['name'].tolist())
        ['Dia de la Constitucion', 'Navidad']
    """
    # Drop special characters in column name
    df["name"] = df["name"].str.replace("\n", " ", regex=True)
    df["name"] = df["name"].str.replace("\r", " ", regex=True)
    df["name"] = df["name"].str.replace("\t", " ", regex=True)

    # Replace characters with accents in columns Name
    df["name"] = df["name"].apply(unidecode.unidecode)

    # Drop non-ascii characters
    df["name"] = df["name"].str.replace("[^\x00-\x7f]+", "", regex=True)

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
    Pivot the calendar DataFrame to create columns for each holiday type.

    This function transforms the long-format calendar DataFrame into a wide format
    where each holiday type becomes a separate column. It handles weight and distance
    columns by creating dummy variables and applying appropriate transformations.
    The function can optionally preserve community information during pivoting.

    :param df_calendar: Calendar DataFrame in long format with holiday names and metadata
    :type df_calendar: pd.DataFrame
    :param pivot_keep_communities: Whether to preserve community information in the pivoted result
    :type pivot_keep_communities: bool
    :return: Pivoted DataFrame with holiday types as columns and dates as rows
    :rtype: pd.DataFrame
    :raises KeyError: If required columns ('name', 'date') are missing from the DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2023-01-01', '2023-12-25'],
        ...     'name': ['New Year', 'Christmas'],
        ...     'weight': [1.0, 1.0]
        ... })
        >>> result = _pivot_calendar(df)
        >>> print('New Year' in result.columns)
        True
        >>> print('Christmas' in result.columns)
        True
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
