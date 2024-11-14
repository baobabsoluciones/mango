from math import radians, cos, sin, asin, sqrt
from .decorators import pydantic_validation
from pydantic import BaseModel, confloat
from typing import Tuple


class HaversineArgs(BaseModel):
    point1: Tuple[confloat(ge=-90.0, le=90.0), confloat(ge=-180.0, le=180.0)]
    point2: Tuple[confloat(ge=-90.0, le=90.0), confloat(ge=-180.0, le=180.0)]


@pydantic_validation(HaversineArgs)
def haversine(point1, point2) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    If you need more precision use geopy implementation as it uses geodesics.
    :param point1: Tuple with latitude and longitude of the first point
    :param point2: Tuple with latitude and longitude of the second point
    """
    # convert decimal degrees to radians
    lat1, lon1 = point1
    lat2, lon2 = point2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
