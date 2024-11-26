import os
# Set wait time to 0 to avoid waiting time between requests as it is not needed for testing
from datetime import datetime
from unittest import TestCase, mock

from mango.clients.aemet import AEMETClient
from mango.shared import ApiKeyError
from mango.validators.aemet import FetchHistoricResponse


class TestAemet(TestCase):
    """
    Test Aemet class
    """

    def assertDataframeEqual(self, df1, df2, *args, **kwargs):
        """
        Test if two dataframes are equal
        """
        try:
            import pandas.testing as pd_testing
        except ImportError:
            raise ImportError("pandas is required to test dataframes")

        return pd_testing.assert_frame_equal(df1, df2, *args, **kwargs)

    def set_enviroment_config(self):
        self._DEFAULT_WAIT_TIME = float(os.environ.get("AEMET_DEFAULT_WAIT_TIME", 1.25))
        self._FETCH_STATIONS_URL = os.environ.get(
            "AEMET_FETCH_STATIONS_URL",
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones",
        )
        self._FETCH_MUNICIPIOS_URL = os.environ.get(
            "AEMET_FETCH_MUNICIPIOS_URL",
            "https://opendata.aemet.es/opendata/api/maestro/municipios",
        )
        self._FETCH_HISTORIC_URL = os.environ.get(
            "AEMET_FETCH_HISTORIC_URL",
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini"
            "/{start_date_str}/fechafin/{end_date_str}"
            "/estacion/{station}",
        )
        self._FETCH_DAILY_URL = os.environ.get(
            "AEMET_FETCH_DAILY_URL",
            "https://opendata.aemet.es/opendata/api/observacion/convencional/datos/estacion/{station}",
        )
        self._FETCH_FORECAST_URL = os.environ.get(
            "AEMET_FETCH_FORECAST_URL",
            "https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{postal_code}",
        )

    def setUp(self) -> None:
        self.set_enviroment_config()
        self.fetch_url_response = {
            "datos": "https://opendata.aemet.es/data",
            "estado": 200,
        }
        self.forecast_data_response = [
            {
                "nombre": "PAMPLONA AEROPUERTO",
                "provincia": "NAVARRA",
                "elaborado": "2021-09-29T10:00:00",
                "prediccion": {},
            }
        ]
        self.invalid_response = {"exito": "negativo", "estado": 400}
        self.stations_response = [
            {
                "indicativo": "3194U",
                "latitud": "412342N",
                "longitud": "123123W",
                "indsinop": "3194U",
                "provincia": "NAVARRA",
                "altitud": 123,
                "nombre": "PAMPLONA AEROPUERTO",
            },
            {
                "indicativo": "3195",
                "latitud": "672342N",
                "longitud": "123123W",
                "indsinop": "3195",
                "provincia": "MADRID",
                "altitud": 123,
                "nombre": "MADRID, RETIRO",
            },
        ]
        self.municipios_response = [
            {
                "nombre": "Pamplona",
                "id": "id31001",
                "latitud_dec": 41.2342,
                "longitud_dec": -1.23123,
            },
            {
                "nombre": "Madrid",
                "id": "id28001",
                "latitud_dec": 41.2342,
                "longitud_dec": -1.23123,
            },
        ]
        # Account for postprocessing
        self.municipios_expected = []
        for m in self.municipios_response:
            m["codigo postal"] = m["id"].lstrip("id")
            self.municipios_expected.append(
                {
                    k: v
                    for k, v in m.items()
                    if k in ["codigo postal", "nombre", "longitud_dec", "latitud_dec"]
                }
            )
        self.live_data_response = [{}]
        self.historic_data_response = [
            {
                "fecha": "2020-01-01",
                "indicativo": "3194U",
                "nombre": "PAMPLONA AEROPUERTO",
                "provincia": "NAVARRA",
                "altitud": 123,
                "tmed": "17,3",
                "prec": "0,0",
                "tmin": "12,6",
                "horatmin": "23:59",
                "tmax": "21,9",
                "horatmax": "14:30",
                "dir": "27,0",
                "velmedia": "1,1",
                "racha": "7,8",
                "horaracha": "14:30",
                "sol": "10,3",
                "presMax": "1018,1",
                "horaPresMax": "Varias",
                "presMin": "1014,1",
                "horaPresMin": "00:00",
            },
            {
                "fecha": "2020-01-02",
                "indicativo": "3194U",
                "nombre": "PAMPLONA AEROPUERTO",
                "provincia": "NAVARRA",
                "altitud": 123,
                "tmed": "17,3",
                "prec": "0,0",
                "tmin": "12,6",
                "horatmin": "23:59",
                "tmax": "21,9",
                "horatmax": "14:30",
                "dir": "27,0",
                "velmedia": "1,1",
                "racha": "7,8",
                "horaracha": "14:30",
                "sol": "10,3",
                "presMax": "1018,1",
                "horaPresMax": "Varias",
                "presMin": "1014,1",
                "horaPresMin": "00:00",
            },
        ]

    def test_enviroment_config(self):
        client = AEMETClient(api_key="1234567890")
        # Default behaviour
        self.assertEqual(
            self._DEFAULT_WAIT_TIME,
            client._DEFAULT_WAIT_TIME,
        )
        self.assertEqual(
            self._FETCH_STATIONS_URL,
            client._FETCH_STATIONS_URL,
        )
        self.assertEqual(
            self._FETCH_MUNICIPIOS_URL,
            client._FETCH_MUNICIPIOS_URL,
        )
        self.assertEqual(
            self._FETCH_HISTORIC_URL,
            client._FETCH_HISTORIC_URL,
        )
        self.assertEqual(
            self._FETCH_DAILY_URL,
            client._FETCH_DAILY_URL,
        )
        self.assertEqual(
            self._FETCH_FORECAST_URL,
            client._FETCH_FORECAST_URL,
        )
        # Custom behaviour
        os.environ["AEMET_DEFAULT_WAIT_TIME"] = "1"
        os.environ["AEMET_FETCH_STATIONS_URL"] = "changed"
        os.environ["AEMET_FETCH_MUNICIPIOS_URL"] = "changed"
        os.environ["AEMET_FETCH_HISTORIC_URL"] = "changed"
        os.environ["AEMET_FETCH_DAILY_URL"] = "changed"
        os.environ["AEMET_FETCH_FORECAST_URL"] = "changed"
        client = AEMETClient(api_key="1")
        self.assertEqual(
            1,
            client._DEFAULT_WAIT_TIME,
        )
        self.assertEqual(
            "changed",
            client._FETCH_STATIONS_URL,
        )
        self.assertEqual(
            "changed",
            client._FETCH_MUNICIPIOS_URL,
        )
        self.assertEqual(
            "changed",
            client._FETCH_HISTORIC_URL,
        )
        self.assertEqual(
            "changed",
            client._FETCH_DAILY_URL,
        )
        self.assertEqual(
            "changed",
            client._FETCH_FORECAST_URL,
        )
        # Reset
        self.set_enviroment_config()
        # Set to 0 for faster testing
        os.environ["AEMET_DEFAULT_WAIT_TIME"] = "0"

    @mock.patch("mango.clients.rest_client.requests")
    def test_init_class(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
        ]

        client = AEMETClient(api_key="1234567890")
        client.connect()
        self.assertIsInstance(client, AEMETClient)
        self.assertEqual(
            client.all_stations,
            self.stations_response,
        )
        self.assertEqual(
            client.municipios,
            self.municipios_expected,
        )
        os.environ["AEMET_DEFAULT_WAIT_TIME"] = "0.5"
        # Test wait time smaller than DEFAULT_WAIT_TIME
        client = AEMETClient(api_key="1234567890", wait_time=0.1)
        client.connect()
        self.assertEqual(
            client._wait_time,
            0.1,
        )
        os.environ["AEMET_DEFAULT_WAIT_TIME"] = "0"

    @mock.patch("mango.clients.rest_client.requests")
    def test_init_class_invalid_api_key(self, mock_requests):
        mock_requests.get.return_value.raise_for_status.side_effect = ApiKeyError(
            "Error"
        )
        mock_requests.get.return_value.status_code = 401
        mock_requests.get.return_value.text = "Api key is invalid"
        with self.assertRaises(ApiKeyError):
            AEMETClient(api_key="").connect()

    @mock.patch("mango.clients.rest_client.requests")
    def test_get_filtered_stations(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
        ]
        client = AEMETClient(api_key="1234567890")
        client.connect()
        # Test with no parameters
        result = client._get_stations_filtered()
        self.assertEqual(result, [st["indicativo"] for st in self.stations_response])
        # Test with station code
        result = client._get_stations_filtered(station_code="3194U")
        self.assertEqual(result, ["3194U"])
        # Invalid station code
        with self.assertRaises(ValueError):
            client._get_stations_filtered(station_code="invalid")
        # Test with lat and long (should return station code of closest station)
        result = client._get_stations_filtered(lat=41.5, long=-1.5)
        self.assertEqual(result, ["3194U"])
        # With province only
        result = client._get_stations_filtered(province="NAVARRA")
        self.assertEqual(result, ["3194U"])
        # lower case
        result = client._get_stations_filtered(province="navarra")
        self.assertEqual(result, ["3194U"])
        # With invalid province
        result = client._get_stations_filtered(province="invalid")
        self.assertEqual(result, ["3194U", "3195"])
        # With invañid province but lat long
        result = client._get_stations_filtered(
            province="invalid", lat=41.2342, long=-1.23123
        )
        self.assertEqual(result, ["3194U"])
        # With province and lat long. Get madrid station lat long but navarra province
        result = client._get_stations_filtered(
            province="NAVARRA", lat=67.2342, long=-1.23123
        )
        self.assertEqual(result, ["3194U"])
        # Now with madrid province
        result = client._get_stations_filtered(
            province="MADRID", lat=67.2342, long=-1.23123
        )
        self.assertEqual(result, ["3195"])

    @mock.patch("mango.clients.rest_client.requests")
    def test_get_meteo_data(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
            self.fetch_url_response,
            self.live_data_response,
            self.fetch_url_response,
            self.historic_data_response,
            self.fetch_url_response,
            self.live_data_response,
        ]

        client = AEMETClient(api_key="1234567890")
        client.connect()

        # Test correct configuration
        result = client.get_meteo_data(station_code="3194U")
        self.assertEqual(result, self.live_data_response)
        # With start date
        result = client.get_meteo_data(
            station_code="3194U",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 2),
        )
        self.assertEqual(
            result, FetchHistoricResponse(self.historic_data_response).model_dump()
        )
        # No station code (lat,long) provided
        result = client.get_meteo_data(lat=41.2342, long=-1.23123)
        self.assertEqual(result, self.live_data_response)
        # Test error type of parameters
        self.assertRaises(TypeError, client.get_meteo_data, station_code=3194)
        self.assertRaises(TypeError, client.get_meteo_data, lat="41.2342")
        self.assertRaises(TypeError, client.get_meteo_data, long="41.2342")
        self.assertRaises(TypeError, client.get_meteo_data, province=1234)
        self.assertRaises(TypeError, client.get_meteo_data, start_date="2020-01-01")
        self.assertRaises(TypeError, client.get_meteo_data, end_date="2020-01-01")
        self.assertRaises(TypeError, client.get_meteo_data, end_date="2020-01-01")
        self.assertRaises(
            NotImplementedError, client.get_meteo_data, output_format="invalid_format"
        )
        # Test bad configuration
        # End date without start date
        self.assertRaises(
            ValueError,
            client.get_meteo_data,
            station_code="3194U",
            end_date=datetime(2020, 1, 1),
        )
        # Start date after end date
        self.assertRaises(
            ValueError,
            client.get_meteo_data,
            station_code="3194U",
            start_date=datetime(2020, 1, 2),
            end_date=datetime(2020, 1, 1),
        )
        # No station code (lat,long) provide only one
        self.assertRaises(ValueError, client.get_meteo_data, lat=41.2343)
        self.assertRaises(ValueError, client.get_meteo_data, long=-1.23123)
        # No station code (lat,long) provide both
        self.assertRaises(ValueError, client.get_meteo_data)

    @mock.patch("mango.clients.rest_client.requests")
    def test_error_response_aemet(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
            self.invalid_response,
            self.invalid_response,
        ]
        client = AEMETClient(api_key="123456789", wait_time=0)
        client.connect()
        # Test error response snd no data due to error
        with self.assertRaises(Exception):
            client.get_meteo_data(station_code="3194U")
        with self.assertRaises(Exception):
            client.get_meteo_data(station_code="3194U", start_date=datetime(2020, 1, 1))

    @mock.patch("mango.clients.rest_client.requests")
    def test_get_meteo_data_pandas(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
            self.fetch_url_response,
            self.historic_data_response,
            self.fetch_url_response,
            self.live_data_response,
        ]
        client = AEMETClient(api_key="1234567890")
        client.connect()
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.text = "OK"
        try:
            import pandas as pd

            result = client.get_meteo_data(
                station_code="3194U",
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 2),
                output_format="df",
            )
            self.assertDataframeEqual(
                result,
                pd.DataFrame(
                    FetchHistoricResponse(self.historic_data_response).model_dump()
                ),
            )
        except ImportError:
            pass
        # Force import error
        with mock.patch.dict("sys.modules", {"pandas": None}):
            self.assertRaises(
                ImportError,
                client.get_meteo_data,
                station_code="3194U",
                output_format="df",
            )

    @mock.patch("mango.clients.rest_client.requests")
    def test_custom_endpoint(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
            self.fetch_url_response,
            self.stations_response,
            self.invalid_response,
        ]
        client = AEMETClient(api_key="1234567890")
        client.connect()
        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.text = "OK"
        result = client.custom_endpoint(
            "/api/valores/climatologicos/inventarioestaciones/todasestaciones"
        )
        self.assertEqual(result, self.stations_response)
        # Invalid endpoint not starting with /api (full url)
        self.assertRaises(
            ValueError,
            client.custom_endpoint,
            "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones",
        )
        # Invalid endpoint not string
        self.assertRaises(TypeError, client.custom_endpoint, 1234)

        # Invalid response
        self.assertRaises(
            ValueError,
            client.custom_endpoint,
            "/api/valores/climatologicos/inventarioestaciones/todasestaciones",
        )

    @mock.patch("mango.clients.rest_client.requests")
    def test_get_forecast_data(self, mock_requests):
        mock_requests.get.return_value.json.side_effect = [
            self.fetch_url_response,
            self.stations_response,
            self.municipios_response,
            self.fetch_url_response,
            self.forecast_data_response,
            self.fetch_url_response,
            self.forecast_data_response,
            self.invalid_response,
        ]
        client = AEMETClient(api_key="1234567890", wait_time=0)
        client.connect()
        # Invalid type
        self.assertRaises(TypeError, client.get_forecast_data, postal_code=1234)
        # Valid postal code
        data = client.get_forecast_data(postal_code="31001")
        self.assertEqual(data, self.forecast_data_response)
        # Wihout postal code
        data = client.get_forecast_data()
        self.assertEqual(data, self.forecast_data_response)
        # Invalid postal code
        self.assertRaises(
            AssertionError, client.get_forecast_data, postal_code="invalid"
        )
        # INvalid response (will raise as no data is found for any postal code)
        self.assertRaises(Exception, client.get_forecast_data, postal_code="31001")
