import os

import pandas as pd


def get_ts_dataset():
    """
    Function to get the time series dataset used in the tests
    :return: the time series dataset
    :rtype: pandas.DataFrame
    """
    this_dir, file = os.path.split(__file__)
    df = pd.read_pickle(f"{this_dir}/ts_dataset.pkl")
    return df


def load_energy_prices_dataset(
    frequency="hourly",
    add_features=None,
    dummy_features: bool = False,
    start_date=None,
    end_date=None,
    output_format="pandas",
):
    """
    Load energy prices dataset.
    This dataset contains the hourly energy prices for Spain between 2015 and 2020.

    :param frequency: type of dataset to load. It can be hourly, daily, weekly or monthly averages.
    :type frequency: str
    :param add_features: additional features to add to the dataset.
      Some of the features are incompatible with the frequency.
      The possible features to have are: hour, week (isocalendar), month, quarter, day of week or year.
    :type add_features: list
    :param dummy_features: if True, the function will transform the features added to a dummy (one hot) encoding. This is not implemented yet.
    :type dummy_features: bool
    :param start_date: start date to load the dataset
    :type start_date: str
    :param end_date: end date to load the dataset
    :type end_date: str
    :param output_format: format of the dataset. It can be pandas or numpy
    :type output_format: str
    """

    # TODO: extend this dataset with data up to the year 2024 or even previous years.
    # TODO: add dummy variables such as holidays (Spain, from this same library)
    # TODO: add dummy variables for COVID.
    # TODO: add dummy variables for Ukraine war (great effect on energy prices on european countries )
    # TODO: missing conversion to one-hot encoding.

    # load the dataset from the csv file
    this_dir, file = os.path.split(__file__)
    data = pd.read_csv(f"{this_dir}/energy_prices.csv")

    if start_date is not None:
        data = data[data["datetime"] >= start_date]

    if end_date is not None:
        data = data[data["datetime"] <= end_date]

    if add_features is not None:
        if "hour" in add_features:
            data["hour"] = data["datetime"].dt.hour
        if "week" in add_features:
            data["week"] = data["datetime"].dt.isocalendar().week
        if "month" in add_features:
            data["month"] = data["datetime"].dt.month
        if "quarter" in add_features:
            data["quarter"] = data["datetime"].dt.quarter
        if "dayofweek" in add_features:
            data["dayofweek"] = data["datetime"].dt.dayofweek
        if "year" in add_features:
            data["year"] = data["datetime"].dt.year

    if frequency == "hourly":
        pass
    elif frequency == "daily":
        # create date column to group by
        data["date"] = data["datetime"].dt.date
        indexes = ["date"]
        if "week" in add_features:
            indexes.append("week")
        if "month" in add_features:
            indexes.append("month")
        if "quarter" in add_features:
            indexes.append("quarter")
        if "dayofweek" in add_features:
            indexes.append("dayofweek")
        if "year" in add_features:
            indexes.append("year")
        return data.groupby(indexes).mean()
    elif frequency == "weekly":
        # create year-week column to group by
        data["week"] = data["datetime"].dt.strftime("%Y-%V")
        indexes = ["week"]
        if "month" in add_features:
            indexes.append("month")
        if "quarter" in add_features:
            indexes.append("quarter")
        if "year" in add_features:
            indexes.append("year")

        return data.groupby(indexes).mean()
    elif frequency == "monthly":
        # create year-month column to group by
        data["month"] = data["datetime"].dt.strftime("%Y-%m")
        indexes = ["month"]
        if "quarter" in add_features:
            indexes.append("quarter")
        if "year" in add_features:
            indexes.append("year")
        return data.groupby(indexes).mean()
    else:
        raise ValueError("Invalid type of dataset")

    if output_format == "pandas":
        return data
    elif output_format == "numpy":
        return data.to_numpy()
    else:
        raise ValueError("Invalid output format for the dataset")
