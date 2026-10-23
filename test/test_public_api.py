"""Smoke tests for the public API: Every name in a module's ``__all__`` exists."""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    "comfort", "comfort.io", "comfort.qc", "comfort.gridding", "comfort.geo", "comfort.physics",
    "comfort.profile_analysis", "comfort.units", "comfort.scaling", "comfort.sections",
    "comfort.database", "comfort.database.information", "comfort.database.structure",
    "comfort.util", "comfort.util.sqlite_utils",
]


# Check if __all__ names exists (in each module)
@pytest.mark.parametrize("module_name", MODULES)
def test_all_names_exist(module_name):
    module = importlib.import_module(module_name)
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.{name}"


# Star import works
def test_star_import_of_root_works():
    exec("from comfort import *", {})


# Bounds, filters and options are keyword-only (to avoid order-induced errors)
@pytest.mark.parametrize("call", [
    "comfort.io.load_comfort(None, ['NITRATE'], None)",
    "comfort.io.read_parameter(None, 'NITRATE', None)",
    "comfort.io.subset_region(None, 10.0)",
    "comfort.gridding.Grid(-90)",
    "comfort.gridding.GridManager.create_grid(None, 0, 10, 1, 0, 10, 1)",
    "comfort.gridding.GridManager.get_grid_id_from_params(None, 0, 10, 1, 0, 10, 1)",
])
def test_bounds_are_keyword_only(call):
    import comfort
    import comfort.gridding
    import comfort.io
    with pytest.raises(TypeError, match="positional"):
        eval(call, {"comfort": comfort})
