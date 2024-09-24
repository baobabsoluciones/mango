# Constants for time aggregation
SELECT_AGR_TMP_DICT = {
    "hourly": "H",
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "Q",
    "yearly": "YE",
}

# Day and month name dictionaries
DAY_NAME_DICT = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}

MONTH_NAME_DICT = {
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
}

ALL_DICT = {
    "Diario": DAY_NAME_DICT,
    "Mensual": MONTH_NAME_DICT,
}