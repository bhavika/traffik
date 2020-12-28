from unittest import TestCase, mock
import pytest
from click.testing import CliRunner
import click
from traffik.scripts import process, validate_cityname


class TestConsole(TestCase):
    @mock.patch('traffik.config.CITIES', ['mumbai', 'dc'])
    def test_basic(self):
        runner = CliRunner()
        result = runner.invoke(process, ["--city", "mumbai"], catch_exceptions=True)
        assert result.exit_code == 1 # FileNotFound

    def test_validate_cityname(self):
        with pytest.raises(click.BadParameter):
            validate_cityname(None, None, "dc")
