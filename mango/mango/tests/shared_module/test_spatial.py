from unittest import TestCase

from mango.shared import haversine, ValidationError


class ValidationTests(TestCase):
    def setUp(self) -> None:
        self.distance = 504.2416415002708  # km

    def tearDown(self) -> None:
        pass

    def test_haversine(self):
        point1 = 40.4165, -3.70256
        point2 = 41.38879, 2.15899
        dist = haversine(point1=point1, point2=point2)
        self.assertEqual(dist, self.distance)

    def test_haversine_invalid_params(self):
        # lat1 invalid
        point1 = 100.0, 40.0  # Invalid
        point2 = 35.0, 35.0  # Valid
        with self.assertRaises(ValidationError):
            haversine(point1=point1, point2=point2)
        # lon1 invalid
        point1 = 35.0, 200.0  # Invalid
        point2 = 35.0, 35.0  # Valid
        with self.assertRaises(ValidationError):
            haversine(point1=point1, point2=point2)
        # lat2 invalid
        point1 = 35.0, 40.0  # Invalid
        point2 = 100.0, 35.0  # Valid
        with self.assertRaises(ValidationError):
            haversine(point1=point1, point2=point2)
        # lon2 invalid
        point1 = 35.0, 40.0  # Invalid
        point2 = 35.0, 200.0  # Valid
        with self.assertRaises(ValidationError):
            haversine(point1=point1, point2=point2)
