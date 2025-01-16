import os
from unittest import TestCase
from unittest import mock

from click.testing import CliRunner

from mango_time_series.cli import cli


class TestCli(TestCase):
    def setUp(self):
        # Get current status of envs
        self.ts_dashboard_name = os.getenv("TS_DASHBOARD_PROJECT_NAME")
        self.ts_dashboard_logo_url = os.getenv("TS_DASHBOARD_LOGO_URL")
        self.ts_dashboard_experimental_features = os.getenv(
            "TS_DASHBOARD_EXPERIMENTAL_FEATURES"
        )

        # Make sure all env variables are unset
        os.unsetenv("TS_DASHBOARD_PROJECT_NAME")
        os.unsetenv("TS_DASHBOARD_LOGO_URL")
        os.unsetenv("TS_DASHBOARD_EXPERIMENTAL_FEATURES")

    def tearDown(self):
        # Reset to value
        if self.ts_dashboard_name:
            os.environ["TS_DASHBOARD_PROJECT_NAME"] = self.ts_dashboard_name
        if self.ts_dashboard_logo_url:
            os.environ["TS_DASHBOARD_LOGO_URL"] = self.ts_dashboard_logo_url
        if self.ts_dashboard_experimental_features:
            os.environ["TS_DASHBOARD_EXPERIMENTAL_FEATURES"] = (
                self.ts_dashboard_experimental_features
            )

    @mock.patch("os.system")
    def test_cli_default(self, mock_system):
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(cli, ["dashboard", "time_series"])

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the command contains streamlit and the correct file
        command = mock_system.call_args[0][0]
        self.assertIn("streamlit run", command)
        self.assertIn("time_series_app.py", command)

        # Assert that the environment variable were set to defaults
        self.assertEqual(os.getenv("TS_DASHBOARD_PROJECT_NAME"), "Project")
        self.assertEqual(os.getenv("TS_DASHBOARD_LOGO_URL"), "")
        self.assertEqual(os.getenv("TS_DASHBOARD_EXPERIMENTAL_FEATURES"), "0")

    @mock.patch("os.system")
    def test_cli_with_args(self, mock_system):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "dashboard",
                "time_series",
                "--experimental_features",
                "1",
                "--logo_url",
                "file:///test.png",
                "--project_name",
                "test",
            ],
        )

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the environment variable was set
        # Assert that the environment variable were set to defaults
        self.assertEqual(os.getenv("TS_DASHBOARD_PROJECT_NAME"), "test")
        self.assertEqual(os.getenv("TS_DASHBOARD_LOGO_URL"), "file:///test.png")
        self.assertEqual(os.getenv("TS_DASHBOARD_EXPERIMENTAL_FEATURES"), "1")
