# Import from python base modules
from logging import log
from typing import Union

import pandas as pd

# Import from external modules
import plotly.express as px
from bs4 import BeautifulSoup


def plotly_plot_lines(
    df,
    x_axis: str,
    y_axis: Union[list, str],
    output_path: str = None,
    color_by: str = None,
    hover_list: list = None,
    show: bool = False,
    title: str = None,
    legend: bool = True,
    position: str = None,
    size: int = None,
    **kwargs_plotly,
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
    :param str output_path: Specify the path where you want to save the plot. Without specifying the extension, the plot will be saved as a html file
    :param str color_by: Specify the column name of the dataframe that will be used to color the lines
    :param list hover_list: Add the name of the column to be shown when hovering over a point in the plot
    :param bool show: Set to True if you want to show the plot
    :param str title: Set the title of the plot
    :param legend: bool: Show or hide the legend
    :param position: str: Specify the position of the legend in a plot
    :param size: int: Set the size of the legend
    :param kwargs_plotly: Additional arguments that will be passed to the plotly express line function.
        See https://plotly.com/python-api-reference/generated/plotly.express.line.html for more information
    """
    # Create a list of traces
    fig = px.line(
        df, x=x_axis, y=y_axis, hover_data=hover_list, color=color_by, **kwargs_plotly
    )

    position_dict = {
        "topright": {"xanchor": "right", "yanchor": "top"},
        "topleft": {"xanchor": "left", "yanchor": "top"},
        "bottomright": {"xanchor": "right", "yanchor": "bottom"},
        "bottomleft": {"xanchor": "left", "yanchor": "bottom"},
    }

    if legend and position is not None:
        try:
            # add title and legend inside the plot
            fig.update_layout(
                title_text=title,
                legend=dict(
                    title="",
                    yanchor=position_dict[position]["yanchor"],
                    y=0.99,
                    xanchor=position_dict[position]["xanchor"],
                    x=0.99,
                    font=dict(size=size),
                ),
            )
        except KeyError:
            print(
                f"{position} must be a one of the following {list(position_dict.keys())}"
            )
            fig.update_layout(title_text=title)
    else:
        # add title
        fig.update_layout(title_text=title)

    # Show
    if show == True:
        fig.show()

    # Save the plot as an html file
    if output_path:
        fig.write_html(
            output_path + ".html" if not output_path.endswith(".html") else output_path
        )
    else:
        return fig


def plotly_plot_scatter(
    df,
    x_axis: str,
    y_axis: Union[list, str],
    color_by: str = None,
    hover_list: list = None,
    output_path: str = None,
    show: bool = False,
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
    :param bool show: Set to True if you want to show the plot
    :param str title: Set the title of the plot
    """
    # Create a list of traces
    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, hover_data=hover_list)

    # add title
    fig.update_layout(title_text=title)

    # Show
    if show == True:
        fig.show()

    # Save the plot as an html file
    if output_path:
        fig.write_html(
            output_path + ".html" if not output_path.endswith(".html") else output_path
        )
    else:
        return fig


def join_html(
    htmls_path: list, button_list: list = None, encoding: str = None
) -> BeautifulSoup:
    """
    The join_html function takes a list of html files and joins them into one html file. In the new html file,
    the user can switch between the different plots by clicking on the buttons. The function takes as input
    the following arguments:
    :param list htmls_path: A list of paths to the html files generated by plotly that will be joined.
    :param list button_list: A list of strings that will be used as the button names.
            If not specified, the button names will be the same as the html file names.
    :param str encoding: The encoding of the html files. Default is "utf-8".
    """
    htmls = []
    if all(file.endswith(".html") for file in htmls_path):
        for file in htmls_path:
            with open(file, encoding=encoding) as f:
                htmls.append(BeautifulSoup(f.read(), "html.parser"))
    else:
        raise ValueError(
            "The input must be a list of html files or a list of BeautifulSoup objects"
        )
    divs = [html.find("div") for html in htmls]
    # Add id to each div
    for i, div in enumerate(divs):
        div["id"] = str(i)
        div["class"] = "custom_plot"
    # Create the function inside the script tag
    script = """
            <script>function showDiv(divId) {
            const divs = document.getElementsByClassName("custom_plot");
            for (let i = 0; i < divs.length; i++) {
            if (divs[i].id === divId) {
                divs[i].style.display = "block";
            } else {
                divs[i].style.display = "none";
            }
            }
            }
            </script>
        """
    # Add the script tag to the output html
    output_html_bs4 = BeautifulSoup(str(htmls[0].head), "html.parser")
    output_html_bs4.head.append(BeautifulSoup(script, "html.parser"))
    # Create the body of the output html
    output_html_bs4.body = output_html_bs4.new_tag("body")
    output_html_bs4.body["style"] = "margin: 0px;"
    output_html_bs4.body["onload"] = 'showDiv("0")'
    output_html_bs4.body["id"] = "body"
    output_html_bs4.insert(1, output_html_bs4.body)
    # Create nav bar with onclick function
    if button_list is None:
        button_list = [f"{html[:-4]}" for html in htmls_path]
    if len(button_list) != len(htmls_path):
        log.warning(
            "The length of the button list must be the same as the length of the htmls_path list"
        )
    for i, button in enumerate(button_list):
        output_html_bs4.body.append(
            output_html_bs4.new_tag(
                "button", onclick=f'showDiv("{i}")', id=f"button_{i}"
            )
        )
        # Change inner text of the button
        output_html_bs4.body.find("button", {"id": f"button_{i}"}).string = button
        # Append al the divs to the output html body
    for div in divs:
        output_html_bs4.body.append(div)
    return output_html_bs4
