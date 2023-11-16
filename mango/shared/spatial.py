from math import radians, cos, sin, asin, sqrt


def haversine(point1, point2) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    :param point1: Tuple with latitude and longitude of the first point
    :param point2: Tuple with latitude and longitude of the second point
    """
    # convert decimal degrees to radians
    lon1, lat1 = point1
    lon2, lat2 = point2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
