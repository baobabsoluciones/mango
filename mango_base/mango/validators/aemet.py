from datetime import date
from typing import List, Optional

from pydantic import BaseModel, RootModel, field_validator, ValidationInfo, ConfigDict


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


class FetchMunicipiosElement(BaseModel):
    id: str
    nombre: str
    longitud_dec: float
    latitud_dec: float


class FetchMunicipiosResponse(RootModel):
    root: List[FetchMunicipiosElement]


class FetchHistoricElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fecha: Optional[date] = None
    indicativo: Optional[str] = None
    nombre: Optional[str] = None
    provincia: Optional[str] = None
    altitud: Optional[float] = None
    tmed: Optional[float] = None
    prec: Optional[float] = None
    tmin: Optional[float] = None
    horatmin: Optional[str] = None
    tmax: Optional[float] = None
    horatmax: Optional[str] = None
    dir: Optional[str] = None
    velmedia: Optional[float] = None
    racha: Optional[float] = None
    horaracha: Optional[str] = None
    sol: Optional[float] = None
    presMax: Optional[float] = None
    horaPresMax: Optional[str] = None
    presMin: Optional[float] = None
    horaPresMin: Optional[str] = None

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
