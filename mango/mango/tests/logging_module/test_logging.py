import logging as log
import os
import time
from unittest import TestCase

from mango.logging import log_time, Chrono
from mango.logging.logger import get_basic_logger, get_configured_logger
from mango.tests.const import normalize_path


class LoggingTests(TestCase):
    def setUp(self) -> None:
        self.root = log.getLogger("tests")
        self.root.setLevel(log.DEBUG)
        self.handler = log.FileHandler(normalize_path("./data/temp.log"))
        self.handler.setLevel(log.DEBUG)
        formatter = log.Formatter("%(asctime)s: %(levelname)s - %(message)s")
        self.handler.setFormatter(formatter)
        self.root.addHandler(self.handler)

    def tearDown(self) -> None:
        self.root.removeHandler(self.handler)
        self.handler.close()
        os.remove(normalize_path("./data/temp.log"))

    def test_logging_decorator(self):
        @log_time("tests")
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

    def test_logging_decorator_root(self):
        logger = get_basic_logger()

        @log_time()
        def do_something():
            return 1 + 1

        result = do_something()
        self.assertEqual(result, 2)
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 0)

        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()

    def test_chrono_class(self):
        chrono = Chrono("first test", silent=True, logger="tests")
        time.sleep(0.1)
        chrono.report_all()
        chrono.start("second test")
        time.sleep(0.1)
        chrono.stop_all(False)
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

    def test_chrono_class_stop_report(self):
        chrono = Chrono("first test", silent=True, logger="tests")
        time.sleep(0.1)
        chrono.report_all()
        chrono.start("second test")
        time.sleep(0.1)
        chrono.stop_all(True)
        chrono.report_all()
        with open(normalize_path("./data/temp.log"), "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 5)
            self.assertIn("first test", lines[0])
            self.assertIn("INFO", lines[0])
            self.assertIn("running", lines[0])

            self.assertIn("first test", lines[1])
            self.assertIn("INFO", lines[1])
            self.assertIn("took", lines[1])

            self.assertIn("second test", lines[2])
            self.assertIn("INFO", lines[2])
            self.assertIn("took", lines[2])

            self.assertIn("first test", lines[3])
            self.assertIn("INFO", lines[3])
            self.assertIn("took", lines[3])

            self.assertIn("second test", lines[4])
            self.assertIn("INFO", lines[4])
            self.assertIn("took", lines[4])

    def test_chrono_not_silent(self):
        chrono = Chrono("first test", silent=False, logger="tests")
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
        chrono = Chrono("first test", silent=True, logger="tests")
        chrono(time.sleep, 0.1)
        chrono.stop_all(False)
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
        chrono = Chrono("first test", silent=True, logger="tests")
        time.sleep(0.1)
        chrono.report("first test", "custom message")
        time.sleep(0.1)
        chrono.stop_all(False)
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

    def test_get_root_logger(self):
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(normalize_path("./data/temp2.log")), exist_ok=True)

        # Create a file handler for logging
        log_path = normalize_path("./data/temp2.log")
        file_handler = log.FileHandler(log_path, mode="w")
        file_handler.setLevel(log.INFO)
        formatter = log.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s")
        file_handler.setFormatter(formatter)

        # Create a logger and add the file handler
        logger = log.getLogger("test_root_logger")
        logger.setLevel(log.INFO)
        logger.addHandler(file_handler)

        # Initialize Chrono with the logger
        chrono = Chrono("first test", silent=True, logger="test_root_logger")
        time.sleep(0.1)
        chrono.report("first test", "custom message")
        time.sleep(0.1)
        chrono.stop_all(False)
        chrono.report_all()

        # Give some time for file writing
        time.sleep(0.1)

        # Verify log file exists and has correct content
        self.assertTrue(os.path.exists(log_path), f"Log file {log_path} not created")

        with open(log_path, "r") as f:
            lines = f.readlines()
            self.assertGreaterEqual(
                len(lines), 2, f"Not enough log lines: {len(lines)}"
            )

            # Check first line
            first_line = lines[0]
            self.assertIn("first test", first_line)
            self.assertIn("INFO", first_line)
            self.assertIn("elapsed", first_line)
            self.assertIn("custom message", first_line)

            # Check second line
            second_line = lines[1]
            self.assertIn("first test", second_line)
            self.assertIn("INFO", second_line)
            self.assertIn("took", second_line)

        # Clean up
        logger.removeHandler(file_handler)
        file_handler.close()

        # Remove the log file
        os.remove(log_path)
