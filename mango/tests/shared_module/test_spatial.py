from unittest import TestCase
from mango.shared import haversine


class ValidationTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_haversine(self):
        point1 = 40.76, -73.984
        point2 = 40.76, -73.984
        dist = haversine(point1,point2)
        self.assertEqual(0,dist)

