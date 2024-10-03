UI_TEXT = {
    # General
    "page_title": "Visualization",
    "sidebar_title": "Visualizations",
    "upload_button_text": "Upload a file",
    "upload_instructions": "Drag and drop a file here or click to upload",
    "file_limits": "Maximum {} MB per file.",
    "no_files_uploaded": "No files uploaded.",

    # Series Selection
    "select_series": "Select series to analyze",
    "choose_column": "Choose {}:",
    "add_selected_series": "Add selected series",
    "series_already_added": "The selected series is already in the list",
    "remove_all_series": "Remove all selected series",
    "no_series_selected": "No series selected",
    "no_columns_to_filter": "No columns to filter. Only one series detected",

    # Plot Options
    "choose_plot": "Select the plot",
    "plot_options": [
        "Original series",
        "Series by year",
        "STL",
        "Lag analysis",
        "Seasonality boxplot",
    ],
    "choose_years": "Choose years to visualize",
    "select_frequency": "Select frequency",
    "frequency_options": ["Daily", "Weekly", "Monthly"],
    "boxplot_titles": {
        "daily": "Daily boxplot",
        "weekly": "Weekly boxplot",
        "monthly": "Monthly boxplot",
    },

    # Visualization
    "select_visualization": "Select visualization",
    "visualization_options": ["Exploration", "Forecast"],
    "select_temporal_grouping": "Select temporal grouping of data",
    "temporal_grouping_options": ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "yearly": "Yearly",

    # Errors and Warnings
    "stl_error": "STL decomposition cannot be performed for the selected series, try another level of temporal aggregation.",
    "boxplot_error": "An appropriate boxplot cannot be displayed for the selected granularity.",
    "temporal_analysis_error": "Temporal analysis cannot be performed",
    "datetime_warning": "The 'datetime' or 'forecast_origin' columns are not present to calculate 'h'.",
    "f_column_missing": "The 'f' column is not present, so errors will not be calculated.",
    "exploration_mode": "The current mode is 'Exploration', forecast columns and errors will not be calculated.",
    "visualization_not_implemented": "The visualization '{}' is not implemented. Please select '{}' or '{}'.",

    # Forecast
    "forecast_plot_title": "Forecast plot",
    "select_series_to_plot": "Select at least one series to plot the forecast",
    "choose_date": "Choose a date",

    # Error Visualization
    "error_visualization_title": "Error visualization",
    "select_date_range": "Select date range (datetime column) to visualize forecast errors",
    "top_10_errors": "Top 10 absolute percentage errors",
    "show_median_or_mean": "Show median or mean",
    "median_or_mean_options": ["Median", "Mean"],
    "median_option": "Median",
    "mean_option": "Mean",
    "mediana_mean_string_dict": {"Median": "median", "Mean": "mean"},
    "error_message": "The {} absolute percentage error of the series is {}",
    "select_plots": "Select plots to display",
    "plot_options_error": ["Box plot by horizon", "Box plot by datetime", "Scatter"],
    "horizon_boxplot_title": "Box plot by horizon",
    "horizon_warning": "The number of points per horizon is highly variable, review your forecast generation process. You should generate the same number of horizons for each datetime, making forecast_origin=forecast_origin-horizon with all the horizons you want to forecast.",
    "datetime_boxplot_title": "Box plot by datetime",
    "select_temporal_aggregation": "Select temporal aggregation for the boxplot",
    "temporal_aggregation_options": ["Daily", "Monthly"],

    # Labels
    "axis_labels": {
        "date": "Date",
        "value": "Value",
        "horizon": "Horizon",
        "error": "Absolute percentage error",
    },
    "series_names": {
        "real": "Real",
        "forecast": "Forecast",
    },
    "error_types": {
        "perc_abs_err": "Absolute percentage error",
        "perc_err": "Percentage error",
        "abs_err": "Absolute error",
        "err": "Error",
        "weekday": "Weekday",
    },

    # File Upload
    "preview_title": "Preview",
    "upload_file": "Upload a file",
    "file_title": "File:",
    "file_name": "File name",
    "separator": "Separator",
    "decimal": "Decimal",
    "thousands": "Thousands",
    "encoding": "Encoding",
    "date_format": "Date format",
    "decimal_help": "e.g., '.', ','",
    "thousands_help": "e.g., ',', '.', ' '",
    "encoding_help": "e.g., 'utf-8', 'latin1', 'ascii'",
    "date_format_help": "e.g., '%Y-%m-%d', '%d/%m/%Y'",
    "load_data": "Load data",
    "manage_files": "Manage uploaded files",
    "separator_help": "e.g., ',', ';', '|', '\\t'",
    "update_file": "Update file",
    "remove_file": "Remove file",

    # Time-related dictionaries
    "DAY_NAME_DICT": {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    },
    "MONTH_NAME_DICT": {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
    },
    "ALL_DICT": {
        "Daily": {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday",
        },
        "Monthly": {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
        },
    },
    "day": "Day",
    "month": "Month",

    # New entries
    "select_filter_type": "Select filter type:",
    "datetime_filter": "Filter by datetime",
    "forecast_origin_filter": "Filter by forecast origin",
    "both_filters": "Filter by both",
    "select_datetime_range": "Select datetime range:",
    "select_forecast_origin_range": "Select forecast origin range:",
    "add_new_file": "Add new file",
}