from math import radians, cos, sin, asin, sqrt
from typing import Tuple

from pydantic import BaseModel, confloat

from .decorators import pydantic_validation


class HaversineArgs(BaseModel):
    """
    Pydantic model for validating Haversine distance calculation arguments.

    Validates that the provided coordinates are within valid latitude and
    longitude ranges for geographic calculations.

    :param point1: First geographic point as (latitude, longitude) tuple
    :type point1: Tuple[float, float]
    :param point2: Second geographic point as (latitude, longitude) tuple
    :type point2: Tuple[float, float]
    :raises ValidationError: If coordinates are outside valid ranges
    """

    point1: Tuple[confloat(ge=-90.0, le=90.0), confloat(ge=-180.0, le=180.0)]
    point2: Tuple[confloat(ge=-90.0, le=90.0), confloat(ge=-180.0, le=180.0)]


@pydantic_validation(HaversineArgs)
def haversine(point1, point2) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Computes the shortest distance between two points on the surface of a
    sphere (Earth) using the Haversine formula. The result is returned in
    kilometers. This implementation assumes Earth is a perfect sphere.

    For higher precision calculations, consider using geopy which implements
    geodesic calculations that account for Earth's ellipsoidal shape.

    :param point1: First geographic point as (latitude, longitude) tuple in decimal degrees
    :type point1: Tuple[float, float]
    :param point2: Second geographic point as (latitude, longitude) tuple in decimal degrees
    :type point2: Tuple[float, float]
    :return: Distance between the two points in kilometers
    :rtype: float
    :raises ValidationError: If coordinates are outside valid ranges (-90 to 90 for latitude, -180 to 180 for longitude)

    Example:
        >>> # Distance between Madrid and Barcelona
        >>> madrid = (40.4168, -3.7038)
        >>> barcelona = (41.3851, 2.1734)
        >>> distance = haversine(madrid, barcelona)
        >>> print(f"Distance: {distance:.2f} km")
        Distance: 504.47 km
        >>>
        >>> # Distance between New York and London
        >>> nyc = (40.7128, -74.0060)
        >>> london = (51.5074, -0.1278)
        >>> distance = haversine(nyc, london)
        >>> print(f"Distance: {distance:.2f} km")
        Distance: 5570.25 km
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
