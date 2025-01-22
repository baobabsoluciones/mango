import logging as log
import os
import time
import json
import tempfile
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

        # Create a temporary directory for log files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        # Remove handlers from the root logger
        self.root.removeHandler(self.handler)
        self.handler.close()

        # Safely attempt to remove the temp log file
        try:
            temp_log_path = normalize_path("./data/temp.log")
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)
        except PermissionError:
            print(f"Could not remove {temp_log_path}. File may be in use.")
        except Exception as e:
            print(f"Error removing temp log file: {e}")

        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _get_temp_log_path(self, filename):
        """Helper method to get a temporary log file path"""
        return os.path.join(self.temp_dir, filename)

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

    def test_get_configured_logger_custom_console_format(self):
        """
        Test get_configured_logger with custom console format and date format
        """
        custom_format = "%(levelname)s - %(message)s"
        custom_datefmt = "%H:%M:%S"
        logger = get_configured_logger(
            logger_type="test_custom_format",
            log_console_format=custom_format,
            log_console_datefmt=custom_datefmt,
        )

        # Verify logger is created
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_custom_format")

    def test_get_configured_logger_colored_console(self):
        """
        Test get_configured_logger with colored console output
        """
        logger = get_configured_logger(
            logger_type="test_color_logger", mango_color=True
        )

        # Verify logger is created
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_color_logger")

    def test_get_configured_logger_file_logging(self):
        """
        Test get_configured_logger with file logging
        """
        log_path = self._get_temp_log_path("test_file_logging.log")

        logger = get_configured_logger(
            logger_type="test_file_logger",
            log_file_path=log_path,
            log_file_level=log.INFO,
            log_file_format="%(asctime)s | %(levelname)s: %(message)s",
        )

        # Log some messages
        logger.info("Test file logging")
        logger.warning("Warning message")

        # Verify log file was created and contains messages
        self.assertTrue(os.path.exists(log_path), f"Log file {log_path} not created")

        with open(log_path, "r") as f:
            lines = f.readlines()
            self.assertGreaterEqual(
                len(lines), 2, f"Not enough log lines: {len(lines)}"
            )

            # Check log messages
            self.assertIn("Test file logging", lines[0])
            self.assertIn("Warning message", lines[1])

    def test_get_configured_logger_json_logging(self):
        """
        Test get_configured_logger with JSON logging
        """
        log_path = self._get_temp_log_path("test_json_logging.json")

        logger = get_configured_logger(
            logger_type="test_json_logger",
            log_file_path=log_path,
            json_fields=["level", "message", "time"],
        )

        # Log some messages
        logger.info("JSON log test")
        logger.error("Error log test")

        # Verify log file was created
        self.assertTrue(
            os.path.exists(log_path), f"JSON log file {log_path} not created"
        )

        # Read and parse JSON log file
        with open(log_path, "r") as f:
            # Read the entire content
            content = f.read()
            print(f"JSON Log File Content: {content}")  # Debug print

            # If the content is a JSON array, parse it directly
            try:
                json_logs = json.loads(content)

                # Verify the logs
                self.assertEqual(
                    len(json_logs), 2, f"Unexpected number of JSON logs: {json_logs}"
                )

                for json_log in json_logs:
                    self.assertIn("level", json_log)
                    self.assertIn("message", json_log)
                    self.assertIn("time", json_log)
            except json.JSONDecodeError:
                # If not a valid JSON array, fall back to line-by-line parsing
                lines = [line.strip() for line in content.split("\n") if line.strip()]

                self.assertGreaterEqual(
                    len(lines), 2, f"Not enough JSON log lines: {lines}"
                )

                # Verify each line is a valid JSON
                for line in lines:
                    try:
                        # Remove leading/trailing brackets if present
                        line = line.strip("[]")
                        json_log = json.loads(line)
                        self.assertIn("level", json_log)
                        self.assertIn("message", json_log)
                        self.assertIn("time", json_log)
                    except json.JSONDecodeError as e:
                        self.fail(f"Invalid JSON line: {line}. Error: {e}")

    def test_get_configured_logger_custom_config_dict(self):
        """
        Test get_configured_logger with a custom configuration dictionary
        """
        custom_config = {
            "version": 1,
            "formatters": {
                "standard": {"format": "%(levelname)s - %(message)s"},
                "custom": {"format": "CUSTOM: %(levelname)s - %(message)s"},
            },
            "handlers": {
                "stream": {  # Add the missing 'stream' handler
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": "INFO",
                },
                "custom_stream": {
                    "class": "logging.StreamHandler",
                    "formatter": "custom",
                    "level": "DEBUG",
                },
            },
            "loggers": {
                "custom_config_logger": {
                    "handlers": ["custom_stream"],
                    "level": "DEBUG",
                    "propagate": False,
                }
            },
        }

        logger = get_configured_logger(
            logger_type="custom_config_logger", config_dict=custom_config
        )

        # Verify logger is created
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "custom_config_logger")
        self.assertEqual(logger.level, log.DEBUG)

    def test_log_time_decorator(self):
        """
        Comprehensive test for the log_time decorator

        This test verifies:
        1. The decorator correctly logs function execution time
        2. The log message contains the function name
        3. The log message contains the execution time
        4. The decorator preserves the original function's return value
        """
        # Prepare a custom logger for testing
        logger = log.getLogger("test_log_time")
        logger.setLevel(log.DEBUG)

        # Create a log file for this test
        log_path = self._get_temp_log_path("log_time_test.log")

        # Create a file handler
        file_handler = log.FileHandler(log_path, mode="w")
        file_handler.setLevel(log.DEBUG)
        formatter = log.Formatter("%(asctime)s: %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Define a test function with the log_time decorator
        @log_time("test_log_time")
        def sample_function(x, y):
            """A sample function to test logging"""
            time.sleep(0.1)  # Simulate some work
            return x + y

        # Call the decorated function
        result = sample_function(3, 4)

        # Verify the function returns the correct result
        self.assertEqual(result, 7)

        # Force flush and wait
        file_handler.flush()
        logger.handlers[0].flush()
        time.sleep(0.2)

        # Read and verify the log file
        with open(log_path, "r") as f:
            lines = f.readlines()

            # Print debug information
            print(f"Log file contents: {lines}")

            # Verify log entry
            self.assertEqual(len(lines), 1, f"Unexpected number of log lines: {lines}")

            # Check log line content
            log_line = lines[0]

            # Verify the log contains key information
            self.assertIn("sample_function", log_line)
            self.assertIn("took", log_line)
            self.assertIn("seconds", log_line)

        # Clean up
        logger.removeHandler(file_handler)
        file_handler.close()

    def test_log_time_decorator_multiple_calls(self):
        """
        Test log_time decorator with multiple function calls

        This test verifies:
        1. The decorator works correctly with multiple calls
        2. Each call is logged separately
        3. The function returns correct values for different inputs
        """
        # Prepare a custom logger for testing
        logger = log.getLogger("test_log_time_multiple")
        logger.setLevel(log.DEBUG)

        # Create a log file for this test
        log_path = self._get_temp_log_path("log_time_multiple_test.log")

        # Create a file handler
        file_handler = log.FileHandler(log_path, mode="w")
        file_handler.setLevel(log.DEBUG)
        formatter = log.Formatter("%(asctime)s: %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Define a test function with the log_time decorator
        @log_time("test_log_time_multiple")
        def multiply_numbers(x, y):
            """A sample function to test logging with multiple calls"""
            time.sleep(0.05)  # Simulate some work
            return x * y

        # Call the decorated function multiple times with different inputs
        results = [
            multiply_numbers(2, 3),
            multiply_numbers(4, 5),
            multiply_numbers(6, 7),
        ]

        # Verify the function returns correct results
        self.assertEqual(results, [6, 20, 42])

        # Force flush and wait
        file_handler.flush()
        logger.handlers[0].flush()
        time.sleep(0.3)

        # Read and verify the log file
        with open(log_path, "r") as f:
            lines = f.readlines()

            # Print debug information
            print(f"Log file contents: {lines}")

            # Verify log entries
            self.assertEqual(len(lines), 3, f"Unexpected number of log lines: {lines}")

            # Check each log line
            for line in lines:
                self.assertIn("multiply_numbers", line)
                self.assertIn("took", line)
                self.assertIn("seconds", line)

        # Clean up
        logger.removeHandler(file_handler)
        file_handler.close()

    def test_log_time_decorator_dynamic_log_level(self):
        """
        Test log_time decorator with different log levels
        """
        # Prepare a custom logger for testing
        logger = log.getLogger("test_log_time_levels")
        logger.setLevel(log.DEBUG)

        # Create a log file for this test
        log_path = self._get_temp_log_path("log_time_levels_test.log")

        # Create a file handler
        file_handler = log.FileHandler(log_path, mode="w")
        file_handler.setLevel(log.DEBUG)
        formatter = log.Formatter("%(asctime)s: %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Test different log levels
        log_levels = [
            ("DEBUG", log.DEBUG),
            ("INFO", log.INFO),
            ("WARNING", log.WARNING),
            ("ERROR", log.ERROR),
        ]

        for level_name, level_value in log_levels:
            # Reset the log file
            open(log_path, "w").close()

            # Define a test function with the log_time decorator using dynamic log level
            @log_time("test_log_time_levels", level=level_name)
            def sample_function():
                """A sample function to test logging with different levels"""
                time.sleep(0.1)  # Simulate some work
                return 42

            # Call the decorated function
            result = sample_function()

            # Verify the function returns the correct result
            self.assertEqual(result, 42)

            # Force flush and wait
            file_handler.flush()
            logger.handlers[0].flush()
            time.sleep(0.2)

            # Read and verify the log file
            with open(log_path, "r") as f:
                lines = f.readlines()

                # Print debug information
                print(f"Log file contents for {level_name} level: {lines}")

                # Verify log entry
                self.assertEqual(
                    len(lines),
                    1,
                    f"Unexpected number of log lines for {level_name} level: {lines}",
                )

                # Check log line content
                log_line = lines[0]

                # Verify the log contains key information and correct level
                self.assertIn("sample_function", log_line)
                self.assertIn("took", log_line)
                self.assertIn("seconds", log_line)
                self.assertIn(level_name, log_line)

        # Clean up
        logger.removeHandler(file_handler)
        file_handler.close()
