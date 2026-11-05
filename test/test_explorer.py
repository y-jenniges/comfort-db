"""Tests for comfort.app.explorer."""
import base64
import logging
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
import pytest

# Skip tests if required imports are not installed
pytest.importorskip("dash", reason="dash not installed ([dash] extra)")
pytest.importorskip("dash_bootstrap_components", reason="dash-bootstrap-components not installed ([dash] extra)")
pytest.importorskip("plotly", reason="plotly not installed ([dash] extra)")

from dash import Dash, Output

from comfort.app import explorer

def _upload(text: str | bytes, media_type: str = "text/csv") -> str:
    """Encode *text* as the data URL that dcc.Upload hands to the callback."""
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return f"data:{media_type};base64,{base64.b64encode(raw).decode()}"


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
        assert len(response.get_json()) == len(explorer._CALLBACKS)

    # Every app created in one process gets the callbacks
    def test_callbacks_registered_on_each_app(self, small_db):
        for _ in range(2):
            app = explorer.create_app(small_db)
            response = app.server.test_client().get("/_dash-dependencies")
            assert len(response.get_json()) == len(explorer._CALLBACKS)

    # A mistyped path must not create an empty database file
    def test_missing_database_file_raises(self, tmp_path):
        missing = tmp_path / "typo.sqlite"
        with pytest.raises(FileNotFoundError, match="typo.sqlite"):
            explorer.create_app(missing)
        assert not missing.exists()


class TestGetConn:
    # Querying the database before create_app() fails with a runtime error
    def test_raises_when_path_not_set(self, monkeypatch):
        monkeypatch.setattr(explorer, "_db_path", None)
        with pytest.raises(RuntimeError, match="create_app"):
            explorer._get_conn()


class TestScatterCallback:
    # The connection is closed even when the query raises
    def test_connection_closed_when_query_fails(self, monkeypatch):
        class TrackedConn:
            closed = False

            def close(self):
                self.closed = True

        conn = TrackedConn()
        monkeypatch.setattr(explorer, "_get_conn", lambda: conn)

        def failing_read(*args, **kwargs):
            raise sqlite3.OperationalError("boom")

        monkeypatch.setattr(explorer, "read_parameter", failing_read)
        monkeypatch.setattr(explorer, "ctx", types.SimpleNamespace(triggered_id="btn-scatter"))
        result = explorer._scatter_callback(
            1, [0, 100], [2000, 2020], "map", [], "depth", "OXYGEN", -180, 180, -90, 90, [],
            None, None, None, None,
        )
        assert conn.closed
        assert "boom" in result[-2]



class TestLoadCsv:
    GOOD = (
        "LONGITUDE,LATITUDE,LEV_M,DATEANDTIME,P_OXYGEN\n"
        "-20.5,50.5,5,2000-01-01 00:00:00,250.0\n"
        "-20.5,50.5,15,2000-01-01 00:00:00,240.0\n"
    )

    @pytest.fixture(autouse=True)
    def clean_cache(self):
        explorer._clear_csv_cache()
        yield
        explorer._clear_csv_cache()

    # A valid gridded table fills the cache and offers its parameter column
    def test_valid_file(self):
        result = explorer._load_csv(_upload(self.GOOD), "grid.csv")
        assert result[0] == "Loaded: grid.csv (2 rows)"
        assert [o["value"] for o in result[1]] == ["P_OXYGEN"]
        assert explorer._csv_cache["df"] is not None
        assert explorer._csv_cache["depths"] == [5.0, 15.0]

    # A file that is not UTF-8 text is reported in the label and the old cache is dropped
    def test_binary_file_is_reported(self):
        explorer._csv_cache["df"] = "stale"
        result = explorer._load_csv(_upload(b"\xff\xfe\x00\x01", "application/octet-stream"), "bad.bin")
        assert result[0].startswith("Could not read bad.bin")
        assert result[1] == []
        assert explorer._csv_cache["df"] is None

    # A table without the grid coordinates is rejected with the names of the missing columns
    def test_missing_columns_are_named(self):
        result = explorer._load_csv(_upload("LONGITUDE,P_OXYGEN\n1,2\n"), "no_depth.csv")
        assert "LATITUDE" in result[0] and "LEV_M" in result[0]

    # Unparsable dates are reported instead of raising inside the callback
    def test_bad_dates_are_reported(self):
        text = self.GOOD.replace("2000-01-01 00:00:00", "not a date")
        result = explorer._load_csv(_upload(text), "dates.csv")
        assert result[0].startswith("Could not read dates.csv")

    # An empty upload does not crash
    def test_empty_file_is_reported(self):
        result = explorer._load_csv(_upload(""), "empty.csv")
        assert result[0].startswith("Could not read empty.csv")

    # Success and failure results both have one entry per Output declared on the callback
    def test_error_result_matches_output_count(self):
        # Count the Output objects among the arguments the callback was registered with
        n_outputs = next(
            sum(isinstance(a, Output) for a in args)
            for args, _, func in explorer._CALLBACKS
            if func is explorer._load_csv
        )
        ok = explorer._load_csv(_upload(self.GOOD), "grid.csv")
        bad = explorer._load_csv(_upload(""), "empty.csv")
        assert len(ok) == len(bad) == n_outputs
