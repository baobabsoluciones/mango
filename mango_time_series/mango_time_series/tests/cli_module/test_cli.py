from unittest import TestCase
from unittest import mock
from click.testing import CliRunner
from mango_time_series.cli import cli
import os


class TestCli(TestCase):
    @mock.patch("os.system")
    def test_cli(self, mock_system):
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "time_series"])

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the command contains streamlit and the correct file
        command = mock_system.call_args[0][0]
        assert "streamlit run" in command
        assert "time_series_app.py" in command

        # Assert that the environment variable was not set or was set to 0
        assert (
            "ENABLE_EXPERIMENTAL_FEATURES" not in os.environ
            or os.environ["ENABLE_EXPERIMENTAL_FEATURES"] == "0"
        )

    @mock.patch("os.system")
    def test_cli_with_environment(self, mock_system):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["dashboard", "time_series", "--experimental_features", "1"]
        )

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the environment variable was set
        assert os.environ["ENABLE_EXPERIMENTAL_FEATURES"] == "1"
