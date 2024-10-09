UI_TEXT = {
    "page_title": "Visualización",
    "upload_button_text": "Subir archivo",
    "upload_instructions": "Arrastra y suelta un archivo CSV aquí o haz clic para seleccionar un archivo.",
    "file_limits": "Máximo {} MB por archivo.",
    "select_series": "Selecciona serie a analizar",
    "choose_plot": "Selecciona el gráfico",
    "plot_options": [
        "Serie original",
        "Serie por año",
        "STL",
        "Análisis de lags",
        "Boxplot de estacionalidad",
    ],
    "choose_years": "Elige los años a visualizar",
    "stl_error": "No se puede realizar la descomposición STL para la serie seleccionada, prueba otro nivel de agregación temporal.",
    "boxplot_error": "No se puede visualizar un boxplot adecuado para la granularidad seleccionada.",
    "select_frequency": "Selecciona la frecuencia",
    "frequency_options": ["Diaria", "Semanal", "Mensual"],
    "boxplot_titles": {
        "daily": "Boxplot diario",
        "weekly": "Boxplot semanal",
        "monthly": "Boxplot mensual",
    },
    "sidebar_title": "Visualizaciones",
    "select_visualization": "Selecciona la visualización",
    "visualization_options": ["Exploración", "Pronóstico"],
    "select_temporal_grouping": "Selecciona la agrupación temporal de los datos",
    "no_columns_to_filter": "No hay columnas para filtrar. Solo una serie detectada",
    "temporal_analysis_error": "No se puede realizar el análisis temporal",
    "forecast_plot_title": "Gráfico de pronóstico",
    "select_series_to_plot": "Selecciona al menos una serie para graficar el pronóstico",
    "choose_date": "Elige una fecha",
    "error_visualization_title": "Visualización de errores",
    "select_date_range": "Seleccione el rango de fechas (columna datetime) para visualizar los errores de pronóstico",
    "top_10_errors": "Top 10 errores porcentuales absolutos",
    "show_median_or_mean": "Mostrar mediana o media",
    "median_or_mean_options": ["Mediana", "Media"],
    "median_option": "Mediana",
    "mean_option": "Media",
    "mediana_mean_string_dict": {"Mediana": "mediana", "Media": "media"},
    "error_message": "El error porcentual absoluto {} de la serie es de {}",
    "select_plots": "Selecciona los gráficos a mostrar",
    "plot_options_error": [
        "Box plot por horizonte",
        "Box plot por datetime",
        "Scatter",
    ],
    "horizon_boxplot_title": "Box plot por horizonte",
    "horizon_warning": "El número de puntos por horizonte es muy variable, revisa tu proceso de generación de pronósticos. Debes generar para cada datetime la misma cantidad de horizontes, haciendo forecast_origin=forecast_origin-horizonte con todos los horizontes que deseas pronosticar.",
    "datetime_boxplot_title": "Box plot por datetime",
    "select_temporal_aggregation": "Selecciona la agrupación temporal para el boxplot",
    "temporal_aggregation_options": ["Diario", "Mensual"],
    "axis_labels": {
        "date": "Fecha",
        "value": "Valor",
        "horizon": "Horizonte",
        "error": "Error porcentual absoluto",
    },
    "series_names": {
        "real": "Real",
        "forecast": "Pronóstico",
    },
    "error_types": {
        "perc_abs_err": "Error porcentual absoluto",
        "perc_err": "Error porcentual",
        "abs_err": "Error absoluto",
        "err": "Error",
        "weekday": "Día de la semana",
    },
    "preview_title": "Vista previa",
    "upload_file": "Subir un archivo",
    "file_title": "Archivo:",
    "file_name": "Nombre del archivo",
    "separator": "Separador",
    "decimal": "Decimal",
    "thousands": "Miles",
    "encoding": "Codificación",
    "date_format": "Formato de fecha",
    "decimal_help": "p. ej., '.', ','",
    "thousands_help": "p. ej., ',', '.', ' '",
    "encoding_help": "p. ej., 'utf-8', 'latin1', 'ascii'",
    "date_format_help": "p. ej., '%Y-%m-%d', '%d/%m/%Y'",
    "load_data": "Cargar datos",
    "manage_files": "Gestionar archivos subidos",
    "separator_help": "p. ej., ',', ';', '|', '\\t'",
    "update_file": "Actualizar archivo",
    "remove_file": "Eliminar archivo",
    "datetime_warning": "Las columnas 'datetime' o 'forecast_origin' no están presentes para calcular 'h'.",
    "f_column_missing": "La columna 'f' no está presente, por lo que no se calcularán los errores.",
    "exploration_mode": "El modo actual es 'Exploración', no se calcularán columnas de forecast ni errores.",
    "no_files_uploaded": "No se han subido archivos.",
    "choose_column": "Elige {}:",
    "add_selected_series": "Agregar serie seleccionada",
    "series_already_added": "La serie seleccionada ya está en la lista",
    "remove_all_series": "Eliminar todas las series seleccionadas",
    "no_series_selected": "No se han seleccionado series",
    "boxplot_by_horizon": "Box plot por horizonte",
    "boxplot_by_datetime": "Box plot por datetime",
    "scatter_plot": "Gráfico de dispersión",
    "temporal_grouping_options": [
        "Horario",
        "Diario",
        "Semanal",
        "Mensual",
        "Trimestral",
        "Anual",
    ],
    "hourly": "Horario",
    "daily": "Diario",
    "weekly": "Semanal",
    "monthly": "Mensual",
    "quarterly": "Trimestral",
    "yearly": "Anual",
    "visualization_not_implemented": "La visualización '{}' no está implementada. Por favor, seleccione '{}' o '{}'.",
    "DAY_NAME_DICT": {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    },
    "MONTH_NAME_DICT": {
        1: "Enero",
        2: "Febrero",
        3: "Marzo",
        4: "Abril",
        5: "Mayo",
        6: "Junio",
        7: "Julio",
        8: "Agosto",
        9: "Septiembre",
        10: "Octubre",
        11: "Noviembre",
        12: "Diciembre",
    },
    "ALL_DICT": {
        "Diario": {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo",
        },
        "Mensual": {
            1: "Enero",
            2: "Febrero",
            3: "Marzo",
            4: "Abril",
            5: "Mayo",
            6: "Junio",
            7: "Julio",
            8: "Agosto",
            9: "Septiembre",
            10: "Octubre",
            11: "Noviembre",
            12: "Diciembre",
        },
    },
    "day": "Día",
    "month": "Mes",
    "select_filter_type": "Selecciona el tipo de filtro:",
    "datetime_filter": "Filtrar por datetime",
    "forecast_origin_filter": "Filtrar por origen de pronóstico",
    "both_filters": "Filtrar por ambos",
    "select_datetime_range": "Selecciona el rango de datetime:",
    "select_forecast_origin_range": "Selecciona el rango de origen de pronóstico:",
    "add_new_file": "Añadir nuevo archivo",
    "aggregated_summary_title": "Resumen de errores",
    "select_top_10": "Selecciona el modelo para ver el top 10 de errores:",
    "best_error_message": "El mejor modelo es '{}', con un error porcentual absoluto de {}",
}
