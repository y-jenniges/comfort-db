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
