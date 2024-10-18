from typing import List

from pydantic import BaseModel, RootModel


class Locations(BaseModel):
    name: str
    x: float
    y: float


class LocationsList(RootModel):
    root: List[Locations]
