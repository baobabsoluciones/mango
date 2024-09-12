import uuid

import streamlit as st
import requests
import pandas as pd
import os
from mango_time_series.mango.utils.processing import aggregate_to_input


@st.cache_data
def aggregate_data(data, select_agr_tmp):
    if select_agr_tmp == "hourly":
        freq = "h"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year', 'month', 'day', 'hour'],
                       'AGG_OPERATIONS': 'sum'}

        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]

    elif select_agr_tmp == "daily":
        freq = "D"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year', 'month', 'day'],
                        'AGG_OPERATIONS': 'mean'}
        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]
    elif select_agr_tmp == "weekly":
        freq = "W"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year', 'week'],
                        'AGG_OPERATIONS': 'mean'}
        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]
    elif select_agr_tmp == "monthly":
        freq = "M"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year', 'month'],
                        'AGG_OPERATIONS': 'mean'}
        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]
    elif select_agr_tmp == "yearly":
        freq = "Y"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year'],
                        'AGG_OPERATIONS': 'mean'}
        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]
    elif select_agr_tmp == "quarterly":
        freq = "Q"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year', 'quarter'],
                        'AGG_OPERATIONS': 'mean'}
        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]
    else:
        freq = "D"
        SERIES_CONF = {'KEY_COLS': ['store', 'product', 'year', 'month', 'day'],
                       'AGG_OPERATIONS': 'mean'}
        data = data[SERIES_CONF['KEY_COLS'] + ['h', 'y', 'f', 'err', 'abs_err', 'perc_err', 'perc_abs_err']]

    # Agreggate data by store, product, year and month and mean of h, y, f, err, abs_err, perc_err and perc_abs_err
    data = data.groupby(SERIES_CONF['KEY_COLS']).agg(SERIES_CONF['AGG_OPERATIONS']).reset_index()

    return data


def select_series(data, columns):
    st.sidebar.title("Selecciona serie a analizar")
    filtered_data = data.copy()

    for i, column in enumerate(columns):
        filter_values = filtered_data[column].unique()

        # Crear un selectbox para cada filtro
        select_series = st.sidebar.selectbox(
            f"Select {column} to filter:", filter_values, key=uuid.uuid4()
        )
        # count = count + 1

        # Filtrar el DataFrame por el valor seleccionado
        filtered_data = filtered_data[filtered_data[column] == select_series].copy()

    selected_item = ([str(select_series) for select_series in filtered_data.iloc[0]])[:len(columns)]
    selected_item = " - ".join(selected_item)

    while True:
        if st.sidebar.button("Añadir serie"):
            st.session_state['selected_series'].append(selected_item)
            break


def interface_visualization(data, logo_path: str = None, project_name: str = None):
    st.set_page_config(
        page_title="Visualizacion",
        page_icon=requests.get(logo_path).content,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title(project_name)

    # Create a sidebar for navigation
    st.sidebar.image(
        os.path.join(os.path.dirname(__file__), "..", "download-removebg-preview.png")
    )
    st.sidebar.title("Visualizaciones")

    # Ad country variable with values spain when store is A and France when store is B
    # data["country"] = data["store"].apply(
    #     lambda x: "Spain" if x == "Store_A" else "France"
    # )

    columns_id = [
        col
        for col in data.columns
        if col
        not in [
            "origin_date",
            "datetime",
            "h",
            "y",
            "f",
            "err",
            "abs_err",
            "perc_err",
            "perc_abs_err",
        ]
    ]
    # Convet origin_date to datetime
    df["origin_date"] = pd.to_datetime(df["origin_date"], format="%Y-%m-%d")

    st.sidebar.title("Selecciona la agrupación temporal de los datos")
    select_agr_tmp = st.sidebar.selectbox(
        f"Select the agrupation to analyze the series:",
        ["hourly", "daily", "weekly", "monthly", "yearly", "quarterly"],
    )

    # Extract hour from datetime
    df["hour"] = df["origin_date"].dt.hour
    # Extract year from datetime
    df["year"] = df["origin_date"].dt.year
    # Extract month from datetime
    df["month"] = df["origin_date"].dt.month
    # Extract day from datetime
    df["day"] = df["origin_date"].dt.day

    data = aggregate_data(df, select_agr_tmp)

    # Crear una lista en session_state para guardar las series seleccionadas
    if 'selected_series' not in st.session_state:
        st.session_state['selected_series'] = []

    select_series(data, columns_id)

    for idx, serie in enumerate(st.session_state['selected_series']):
        col1, col2 = st.sidebar.columns([8, 2])
        with col1:
            st.write(f"{idx + 1}. {serie}")
        with col2:
            if st.button("❌", key=f"remove_{idx}"):
                st.session_state['selected_series'].pop(idx)
                st.rerun()  # Refresh app to update the list of selected series

    # all dates and convert to datetime
    # all_dates = filtered_data["origin_date"].unique()
    # all_dates = pd.to_datetime(all_dates)
    #
    # # Create date input to select date
    # st.sidebar.title("Selecciona fecha de inicio predicción")
    # selected_date = st.sidebar.date_input(
    #     "Choose a date",
    #     min_value=min(all_dates),
    #     max_value=max(all_dates),
    #     value=min(all_dates),
    #     label_visibility="collapsed",
    # )
    #
    # st.write(f"Selected date: {selected_date}")
    #
    # # Draw de plot prediction
    # st.write("Prediction plot")



file = r"C:\Users\MontserratMuñoz\Documents\codigo\streamlit_code\streamlit_code\data\test_multi_index.csv"
logo_path = r"https://www.multiserviciosaeroportuarios.com/wp-content/uploads/2024/03/cropped-Logo-transparente-blanco-Multiservicios-Aeroportuarios-Maero-1-192x192.png"
df = pd.read_csv(file)
interface_visualization(data=df, logo_path=logo_path, project_name="Prueba")
