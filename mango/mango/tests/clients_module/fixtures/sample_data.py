"""
Sample data for CatastroData tests.

This module contains mock data, responses, and URLs used across
different test files to ensure consistency and maintainability.
"""

# Mock municipality data - estructura típica del DataFrame de municipios
MOCK_MUNICIPALITY_DATA = {
    "territorial_office_code": {"0": "02"},
    "territorial_office_name": {"0": "Albacete"},
    "catastro_municipality_code": {"0": "02001"},
    "catastro_municipality_name": {"0": "ABENGIBRE"},
    "datatype": {"0": "Buildings"},
    "zip_link": {
        "0": r"http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02001-ABENGIBRE/A.ES.SDGC.BU.02001.zip",
    },
}

# Datos adicionales para tests con múltiples municipios
MOCK_MULTIPLE_MUNICIPALITIES_DATA = {
    "territorial_office_code": {"0": "02", "1": "02", "2": "28"},
    "territorial_office_name": {"0": "Albacete", "1": "Albacete", "2": "Madrid"},
    "catastro_municipality_code": {"0": "02001", "1": "02002", "2": "28001"},
    "catastro_municipality_name": {"0": "ABENGIBRE", "1": "ALATOZ", "2": "MADRID"},
    "datatype": {"0": "Buildings", "1": "Buildings", "2": "Addresses"},
    "zip_link": {
        "0": r"http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02001-ABENGIBRE/A.ES.SDGC.BU.02001.zip",
        "1": r"http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02002-ALATOZ/A.ES.SDGC.BU.02002.zip",
        "2": r"http://www.catastro.hacienda.gob.es/INSPIRE/Addresses/28/28001-MADRID/A.ES.SDGC.AD.28001.zip",
    },
}

# Mock HTTP responses para diferentes escenarios
MOCK_SUCCESS_RESPONSE = {
    "status_code": 200,
    "content": b"fake-zip-data-content",
    "headers": {"content-type": "application/zip", "content-length": "1024"},
}

MOCK_ERROR_RESPONSES = {
    "timeout": "requests.exceptions.Timeout",
    "http_404": {"status_code": 404, "reason": "Not Found"},
    "http_500": {"status_code": 500, "reason": "Internal Server Error"},
    "connection_error": "requests.exceptions.ConnectionError",
    "invalid_response": {"status_code": 200, "content": b"invalid-zip-content"},
}

# URLs de prueba para integration tests
TEST_URLS = {
    "small_municipality": r"http://www.catastro.hacienda.gob.es/INSPIRE/Buildings/02/02001-ABENGIBRE/A.ES.SDGC.BU.02001.zip",
    "httpbin_test": "http://httpbin.org/status/200",
    "httpbin_timeout": "http://httpbin.org/delay/10",
    "httpbin_404": "http://httpbin.org/status/404",
}

# Códigos de municipios para tests
TEST_MUNICIPALITY_CODES = {
    "small": "02001",  # Abengibre (Albacete) - municipio pequeño
    "medium": "02003",  # Albacete capital - municipio mediano
    "nonexistent": "99999",  # Código que no existe
}

# Tipos de datos válidos e inválidos
VALID_DATATYPES = ["Buildings", "CadastralParcels", "Addresses"]
INVALID_DATATYPES = ["InvalidType", "buildings", "BUILDINGS", ""]

# Configuraciones de cache para tests
CACHE_CONFIGS = {
    "enabled": {"cache": True, "cache_file_path": "test_cache.json"},
    "disabled": {"cache": False},
    "custom_path": {"cache": True, "cache_file_path": "custom_test_cache.json"},
}

# Mock feed data para tests de parsing
MOCK_FEED_DATA = {
    "territorial_entry": {
        "title": "Territorial office 02 Albacete",
        "link": "http://test.catastro.es/territorial/02",
    },
    "municipality_entry": {
        "title": "02001-ABENGIBRE buildings",
        "link": "http://test.catastro.es/municipality/02001.zip",
    },
    "invalid_territorial": {
        "title": "Invalid title format",
        "link": "http://test.catastro.es/invalid",
    },
    "invalid_municipality": {
        "title": "Invalid format",
        "link": "http://test.catastro.es/invalid",
    },
}
