# Full imports
import pandas as pd
import plotly.express as px

# Partial imports
from typing import Union


def plotly_plot_lines(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: Union[list, str],
    output_path: str,
    hover_list: list = None,
    title: str = None,
):
    """
    The plot_lines function creates a plotly line chart from the given dataframe.
    It takes as input arguments:
        df (pandas DataFrame): DataFrame with the data to plot.
        x_axis (str): The column name of the column that will be used for the x-axis values.
        y_axis (list of strs or str): A string or list containing strings corresponding to columns in df that will be plotted on y-axis.

    :param pd.DataFrame df: Specify the dataframe that contains the data to be plotted
    :param str x_axis: Specify the column name of the dataframe that will be used as x axis
    :param Union[list, str] y_axis: Specify the columns of the dataframe that will be plotted
    :param str output_path: Specify the path where you want to save the plot. Without specifying the extension, the plot will be saved as a html file.
    :param list hover_list: Add the name of the column to be shown when hovering over a point in the plot
    :param str title: Set the title of the plot
    """
    # Create a list of traces
    fig = px.line(df, x=x_axis, y=y_axis, hover_data=hover_list)

    # add title
    fig.update_layout(title_text=title)

    # Save the plot as an html file
    fig.write_html(output_path + ".html")


def plotly_plot_scatter(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: Union[list, str],
    color_by: str = None,
    hover_list: list = None,
    output_path: str = None,
    title: str = None,
):
    """
    The plot_scatter function creates a scatter plot of the dataframe.
    It takes in a dataframe, x-axis column name, y-axis column name(s), and an optional color_by argument.
    The function will create a scatter plot with the x-axis as the inputted x_column and each y_column on
    the y axis.

    :param pd.DataFrame df: Pass in the dataframe that contains the data to be plotted
    :param str x_axis: Specify the column name of the dataframe that should be used as x-axis
    :param Union[list, str] y_axis: Allow the function to be used for both single and multiple y-axes
    :param str color_by: Specify which column in the dataframe should be used to color the points
    :param list hover_list:list: Add a list of columns that will be displayed when hovering over the data points
    :param str output_path: Specify the path where the html file will be saved
    :param str title: Set the title of the plot
    """
    # Create a list of traces
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, hover_data=hover_list)

    # add title
    fig.update_layout(title_text=title)

    # Save the plot as an html file
    fig.write_html(output_path + ".html")
