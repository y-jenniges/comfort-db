"""Tests for the paths module."""
from pathlib import Path
import pytest

from comfort.util.paths import resolve_path


class TestResolvePath:
    # A given path is returned unchanged, independent of the environment
    def test_argument_beats_environment_variable(self, monkeypatch):
        monkeypatch.setenv("COMFORT_TEST_PATH", "/from/environment")
        assert resolve_path("/from/argument", "COMFORT_TEST_PATH", "test file") == "/from/argument"

    # Path objects are returned as they are
    def test_path_object_is_returned(self):
        path = Path("some") / "file.nc"
        assert resolve_path(path, "COMFORT_TEST_PATH", "test file") is path

    # Without an argument (None or empty), the environment variable is used
    @pytest.mark.parametrize("path", [None, ""])
    def test_environment_variable(self, monkeypatch, path):
        monkeypatch.setenv("COMFORT_TEST_PATH", "/from/environment")
        assert resolve_path(path, "COMFORT_TEST_PATH", "test file") == "/from/environment"

    # The variable comes from the .env file in the working directory, not from one next to the library code
    # (a different file than the repo's own .env, so this fails if the wrong one is read)
    def test_dotenv_file_in_working_directory(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text('COMFORT_TEST_PATH="/from/dotenv"\n')
        monkeypatch.chdir(tmp_path)
        # load_dotenv sets the variable directly, so register it with monkeypatch to remove it after the test
        monkeypatch.setenv("COMFORT_TEST_PATH", "placeholder")
        monkeypatch.delenv("COMFORT_TEST_PATH")
        assert resolve_path(None, "COMFORT_TEST_PATH", "test file") == "/from/dotenv"

    # A variable that is already set wins over the .env file
    def test_environment_variable_beats_dotenv_file(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text('COMFORT_TEST_PATH="/from/dotenv"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COMFORT_TEST_PATH", "/from/environment")
        assert resolve_path(None, "COMFORT_TEST_PATH", "test file") == "/from/environment"

    # Without an argument and without the variable, the error names both the variable and the file
    def test_missing_path_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("COMFORT_TEST_PATH", raising=False)
        with pytest.raises(EnvironmentError, match="COMFORT_TEST_PATH") as error:
            resolve_path(None, "COMFORT_TEST_PATH", "test file")
        assert "test file" in str(error.value)
