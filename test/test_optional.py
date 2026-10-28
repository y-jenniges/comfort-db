"""Tests for comfort.util.optional and the install hints of the optional extras."""
import sys
import pytest

from comfort.io import load_comfort
from comfort.util.optional import import_optional


class TestImportOptional:
    # An installed module is returned
    def test_returns_module(self):
        assert import_optional("json", "none", "testing") is sys.modules["json"]

    # A submodule path is imported as that submodule
    def test_submodule(self):
        assert import_optional("os.path", "none", "testing") is sys.modules["os.path"]

    # A missing module names the module, the feature and the extra
    def test_missing_module_message(self):
        with pytest.raises(ImportError, match=r"dummy_module is required for the test; .*comfort-db\[grid\]"):
            import_optional("dummy_module.sub", "grid", "the test")


class TestInstallHints:
    # load_comfort fails on missing xarray before it touches the database
    def test_load_comfort_without_xarray(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(ImportError, match=r"xarray is required for load_comfort.*comfort-db\[grid\]"):
            load_comfort("does_not_exist.sqlite")

    # The dict output does not need xarray
    def test_load_comfort_dict_without_xarray(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "xarray", None)
        assert load_comfort(tmp_path / "empty.sqlite", output="dict") == {}

    # Region parsing names the geo extra when shapely is missing
    def test_regions_without_shapely(self, monkeypatch):
        from comfort.geo import _parse_regions
        monkeypatch.setitem(sys.modules, "shapely.geometry", None)
        with pytest.raises(ImportError, match=r"shapely is required.*comfort-db\[geo\]"):
            _parse_regions({"box": [(0, 0), (1, 0), (1, 1)]}, "name")
