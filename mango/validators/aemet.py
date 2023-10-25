from datetime import datetime, date
from typing import Optional, Literal, List

from pydantic import BaseModel, RootModel, field_validator, ValidationInfo


class UrlCallResponse(BaseModel):
    estado: int
    datos: str


class FetchStationsElement(BaseModel):
    latitud: str
    longitud: str
    provincia: str
    altitud: float
    indicativo: str
    nombre: str
    indsinop: str


class FetchStationsResponse(RootModel):
    root: List[FetchStationsElement]


class FetchHistoricElement(BaseModel):
    fecha: date
    indicativo: str
    nombre: str
    provincia: str
    altitud: float
    tmed: float
    prec: float
    tmin: float
    horatmin: str
    tmax: float
    horatmax: str
    dir: str
    velmedia: float
    racha: float
    horaracha: str
    sol: float
    presMax: float
    horaPresMax: str
    presMin: float
    horaPresMin: str

    @field_validator(
        "tmed",
        "prec",
        "tmin",
        "tmax",
        "velmedia",
        "racha",
        "sol",
        "presMax",
        "presMin",
        mode="before",
    )
    @classmethod
    def _parse_float(cls, v: str, info: ValidationInfo) -> float:
        try:
            return float(v.replace(",", "."))
        except ValueError:
            raise ValueError(f"{info.field_name}: Could not parse {v} to float")


class FetchHistoricResponse(RootModel):
    root: List[FetchHistoricElement]
