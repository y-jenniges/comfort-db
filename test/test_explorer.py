"""Tests for comfort.app.explorer."""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

# Skip tests if required imports are not installed
pytest.importorskip("dash", reason="dash not installed ([dash] extra)")
pytest.importorskip("dash_bootstrap_components", reason="dash-bootstrap-components not installed ([dash] extra)")
pytest.importorskip("plotly", reason="plotly not installed ([dash] extra)")

from dash import Dash

from comfort.app import explorer

# Number of @callback functions registered in explorer.py
N_CALLBACKS = 6


@pytest.fixture
def small_db(tmp_path):
    """Create a minimal COMFORT-like SQLite database with one parameter table."""
    db_path = str(tmp_path / "explorer.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE station (ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)")
    conn.execute("""
        CREATE TABLE P_NITRATE (
            ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL,
            PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER,
            BOTTLE_NUMBER INTEGER, PROFILE_NUMBER INTEGER,
            PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER
        )
    """)
    conn.commit()
    conn.close()
    return db_path


class TestImport:
    # Importing the module must not need a database or COMFORT_DB_PATH (e.g. for the docs build)
    def test_import_without_database_path(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "COMFORT_DB_PATH"}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-c", "import comfort.app.explorer"],
            env=env, cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


class TestCreateApp:
    # Without an explicit path or COMFORT_DB_PATH an error is raised when creating the app
    def test_raises_without_path(self, monkeypatch):
        monkeypatch.delenv("COMFORT_DB_PATH", raising=False)
        monkeypatch.setattr(explorer, "load_dotenv", None)
        with pytest.raises(EnvironmentError, match="COMFORT_DB_PATH"):
            explorer.create_app()

    # The path can come from the environment variable instead of the argument
    def test_reads_environment_variable(self, monkeypatch, small_db):
        monkeypatch.setenv("COMFORT_DB_PATH", small_db)
        assert isinstance(explorer.create_app(), Dash)

    # The layout lists the parameters found in the database
    def test_layout_contains_parameters(self, small_db):
        app = explorer.create_app(small_db)
        assert isinstance(app, Dash)
        response = app.server.test_client().get("/_dash-layout")
        assert response.status_code == 200
        assert b"NITRATE" in response.data

    # All callbacks are attached to the created app
    def test_callbacks_registered(self, small_db):
        app = explorer.create_app(small_db)
        response = app.server.test_client().get("/_dash-dependencies")
        assert response.status_code == 200
        assert len(response.get_json()) == N_CALLBACKS

    # Every app created in one process gets the callbacks
    def test_callbacks_registered_on_each_app(self, small_db):
        for _ in range(2):
            app = explorer.create_app(small_db)
            response = app.server.test_client().get("/_dash-dependencies")
            assert len(response.get_json()) == N_CALLBACKS


class TestGetConn:
    # Querying the database before create_app() fails with a runtime error
    def test_raises_when_path_not_set(self, monkeypatch):
        monkeypatch.setattr(explorer, "_db_path", None)
        with pytest.raises(RuntimeError, match="create_app"):
            explorer._get_conn()
