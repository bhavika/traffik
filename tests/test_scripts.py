from unittest import TestCase
from click.testing import CliRunner
from traffik.scripts import cli


class TestConsole(TestCase):
    def test_basic(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--city", "berlin"])
        assert result.exit_code == 0
