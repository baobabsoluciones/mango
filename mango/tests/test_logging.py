import os
import time
from unittest import TestCase
import logging as log

from mango.logging import log_time, Chrono
from mango.tests.const import normalize_path


class LoggingTests(TestCase):
    def setUp(self) -> None:
        self.root = log.getLogger("root")
        self.root.setLevel(log.DEBUG)
        self.handler = log.FileHandler(normalize_path("./data/temp.log"))
        self.handler.setLevel(log.DEBUG)
        formatter = log.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(formatter)
        self.root.addHandler(self.handler)

    def tearDown(self) -> None:
        self.root.removeHandler(self.handler)
        self.handler.close()
        os.remove(normalize_path("./data/temp.log"))

    def test_logging_decorator(self):
        @log_time
        def do_something():
            return 1 + 1

        result = do_something()
        self.assertEqual(result, 2)
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("do_something", lines[0])
            self.assertIn("INFO", lines[0])
            self.assertIn("2", lines[0])

    def test_chrono_class(self):
        chrono = Chrono("first test", silent=True)
        time.sleep(0.1)
        chrono.report_all()
        chrono.start("second test")
        time.sleep(0.1)
        chrono.stop_all()
        chrono.report_all()
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("first test", lines[0])
            self.assertIn("INFO", lines[0])
            self.assertIn("running", lines[0])

            self.assertIn("first test", lines[1])
            self.assertIn("INFO", lines[1])
            self.assertIn("took", lines[1])

            self.assertIn("second test", lines[2])
            self.assertIn("INFO", lines[2])
            self.assertIn("took", lines[2])

    def test_chrono_not_silent(self):
        chrono = Chrono("first test", silent=False)
        time.sleep(0.1)
        chrono.start("second test")
        time.sleep(0.1)
        chrono.stop_all()
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

            self.assertIn("first test", lines[0])
            self.assertIn("INFO", lines[0])
            self.assertIn("took", lines[0])

            self.assertIn("second test", lines[1])
            self.assertIn("INFO", lines[1])
            self.assertIn("took", lines[1])

    def test_chrono_callable(self):
        chrono = Chrono("first test", silent=True)
        chrono(time.sleep, 0.1)
        chrono.stop_all()
        chrono.report_all()
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("sleep", lines[0])
            self.assertIn("INFO", lines[0])
            self.assertIn("took", lines[0])

            self.assertIn("first test", lines[1])
            self.assertIn("INFO", lines[1])
            self.assertIn("took", lines[1])

            self.assertIn("sleep", lines[2])
            self.assertIn("INFO", lines[2])
            self.assertIn("took", lines[2])

    def test_chrono_report_custom(self):
        chrono = Chrono("first test", silent=True)
        time.sleep(0.1)
        chrono.report("first test", "custom message")
        time.sleep(0.1)
        chrono.stop_all()
        chrono.report_all()
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("first test", lines[0])
            self.assertIn("INFO", lines[0])
            self.assertIn("elapsed", lines[0])
            self.assertIn("custom message", lines[0])

            self.assertIn("first test", lines[1])
            self.assertIn("INFO", lines[1])
            self.assertIn("took", lines[1])
