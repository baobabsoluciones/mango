import os
from unittest import TestCase
from unittest import mock

from click.testing import CliRunner
from mango_dashboard.file_explorer.cli import cli


class TestFileExplorerCli(TestCase):
    def setUp(self):
        # Create a temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_data")
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # Create a test file
        self.test_file = os.path.join(self.test_dir, "test.csv")
        with open(self.test_file, "w") as f:
            f.write("col1,col2\n1,2\n3,4")

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    @mock.patch("os.system")
    def test_cli_file_explorer_default(self, mock_system):
        """Test the file explorer CLI with default parameters"""
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "file_explorer"])

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the command contains streamlit and the correct file
        command = mock_system.call_args[0][0]
        self.assertIn("streamlit run", command)
        self.assertIn("file_explorer_app.py", command)

        # Assert default path is current directory
        self.assertIn("--path", command)
        self.assertIn(os.getcwd(), command)

    @mock.patch("os.system")
    def test_cli_file_explorer_with_path(self, mock_system):
        """Test the file explorer CLI with custom path"""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboard", "file_explorer", "--path", self.test_dir],
        )

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the command contains the custom path
        command = mock_system.call_args[0][0]
        self.assertIn("streamlit run", command)
        self.assertIn("file_explorer_app.py", command)
        self.assertIn(f'--path "{self.test_dir}"', command)

    @mock.patch("os.system")
    def test_cli_file_explorer_with_all_options(self, mock_system):
        """Test the file explorer CLI with all available options"""
        config_path = os.path.join(self.test_dir, "config.json")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "dashboard",
                "file_explorer",
                "--path",
                self.test_dir,
                "--editable",
                "1",
                "--config_path",
                config_path,
            ],
        )

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert that os.system was called
        mock_system.assert_called_once()

        # Assert that the command contains all options
        command = mock_system.call_args[0][0]
        self.assertIn("streamlit run", command)
        self.assertIn("file_explorer_app.py", command)
        self.assertIn(f'--path "{self.test_dir}"', command)
        self.assertIn("--editable 1", command)
        self.assertIn(f'--config_path "{config_path}"', command)

    def test_cli_help(self):
        """Test that help command works"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert help text contains expected information
        self.assertIn("mango_file_explorer", result.output)
        self.assertIn("Commands in the mango_file_explorer cli", result.output)

    def test_cli_dashboard_help(self):
        """Test that dashboard help command works"""
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "--help"])

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert help text contains expected information
        self.assertIn("Commands in the dashboard", result.output)

    def test_cli_file_explorer_help(self):
        """Test that file_explorer help command works"""
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "file_explorer", "--help"])

        # Assert the command executed successfully
        assert result.exit_code == 0

        # Assert help text contains expected information
        self.assertIn("Creates the data folder explorer dashboard", result.output)
        self.assertIn("--path", result.output)
        self.assertIn("--editable", result.output)
        self.assertIn("--config_path", result.output)
