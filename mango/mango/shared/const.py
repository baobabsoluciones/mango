"""
ARCGIS CONSTANTS
"""

ARCGIS_TOKEN_URL = "https://www.arcgis.com/sharing/rest/oauth2/token"
ARCGIS_GEOCODE_URL = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
ARCGIS_ODMATRIX_DIRECT_URL = (
    "https://route.arcgis.com/arcgis/rest/services/World/OriginDestinationCostMatrix/NAServer/"
    "OriginDestinationCostMatrix_World/solveODCostMatrix"
)
ARCIS_ODMATRIX_JOB_URL = (
    "https://logistics.arcgis.com/arcgis/rest/services/World/OriginDestinationCostMatrix/GPServer/"
    "GenerateOriginDestinationCostMatrix/"
)

ARCGIS_CAR_TRAVEL_MODE = {}

ARCGIS_TRUCK_TRAVEL_MODE = {
    "attributeParameterValues": [
        {
            "attributeName": "Avoid Unpaved Roads",
            "parameterName": "Restriction Usage",
            "value": "AVOID_HIGH",
        },
        {
            "attributeName": "Avoid Private Roads",
            "parameterName": "Restriction Usage",
            "value": "AVOID_MEDIUM",
        },
        {
            "attributeName": "Through Traffic Prohibited",
            "parameterName": "Restriction Usage",
            "value": "AVOID_HIGH",
        },
        {
            "attributeName": "TravelTime",
            "parameterName": "Vehicle Maximum Speed (km/h)",
            "value": 90,
        },
        {
            "attributeName": "Roads Under Construction Prohibited",
            "parameterName": "Restriction Usage",
            "value": "PROHIBITED",
        },
        {
            "attributeName": "Avoid Gates",
            "parameterName": "Restriction Usage",
            "value": "AVOID_MEDIUM",
        },
        {
            "attributeName": "Driving a Bus",
            "parameterName": "Restriction Usage",
            "value": "AVOID_HIGH",
        },
    ],
    "description": "Models the movement of cars and other similar small automobiles,"
    " such as pickup trucks, and finds solutions that optimize travel time."
    " Travel obeys one-way roads, avoids illegal turns, and follows other rules that are"
    " specific to cars. When you specify a start time, dynamic travel speeds based"
    " on traffic are used where it is available.",
    "distanceAttributeName": "Kilometers",
    "id": "FEgifRtFndKNcJMJ",
    "impedanceAttributeName": "TravelTime",
    "name": "Driving Time",
    "restrictionAttributeNames": [
        "Avoid Unpaved Roads",
        "Avoid Private Roads",
        "Through Traffic Prohibited",
        "Roads Under Construction Prohibited",
        "Avoid Gates",
        "Driving a Bus",
    ],
    "simplificationTolerance": 10,
    "simplificationToleranceUnits": "esriMeters",
    "timeAttributeName": "TravelTime",
    "type": "AUTOMOBILE",
    "useHierarchy": True,
    "uturnAtJunctions": "Allowed only at dead ends",
}
