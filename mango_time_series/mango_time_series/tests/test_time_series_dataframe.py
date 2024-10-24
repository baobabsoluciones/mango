from unittest import TestCase

from mango_time_series.time_series import TimeSeriesDataFrame


class TestTimeSeriesDataframe(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        instance = TimeSeriesDataFrame()
        text = instance.hellow_world()
        self.assertEqual(text, "Hello World")
