"""
Pytest configuration.

Tests that require a real bathymetry NetCDF file are marked with
@pytest.mark.needs_bathymetry. They are skipped unless the environment
variable BATHYMETRY_PATH is set and points to an existing file.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def _bathymetry_skip_reason() -> str | None:
    """Return why the bathymetry tests cannot run. If they can, return ``None``."""
    path = os.environ.get("BATHYMETRY_PATH")
    if not path:
        # No bathymetry path
        return "BATHYMETRY_PATH not set (needs_bathymetry test)"
    if not os.path.isfile(path):
        # Bathymetry file does not exist
        return f"BATHYMETRY_PATH is not an existing file: {path} (needs_bathymetry test)"
    return None


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with pytest.mark.needs_bathymetry."""
    reason = _bathymetry_skip_reason()
    if reason is None:
        return
    skip_needs_bathymetry = pytest.mark.skip(reason=reason)
    for item in items:
        if "needs_bathymetry" in item.keywords:
            item.add_marker(skip_needs_bathymetry)
