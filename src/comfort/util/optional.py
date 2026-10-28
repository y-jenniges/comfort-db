"""Import helper for optional dependencies."""
from __future__ import annotations

import importlib
from types import ModuleType


def import_optional(module: str, install_extra: str, required_for: str) -> ModuleType:
    """Import an optional dependency or raise an ``ImportError`` with install instructions.

    Args:
        module (str): Module to import, e.g. ``"xarray"`` or ``"cartopy.crs"``.
        install_extra (str): Name of the ``comfort-db`` extra that installs it, e.g. ``"grid"``.
        required_for (str): What needs the module, e.g. ``"load_comfort"``.
    Returns:
        types.ModuleType: The imported module.
    Raises:
        ImportError: If *module* is not installed.
    """
    try:
        # Import the module
        return importlib.import_module(module)
    except ImportError as err:
        # Raise error with install instructions, keeping the original cause
        raise ImportError(
            f"{module.split('.')[0]} is required for {required_for}; "
            f'install it with: pip install "comfort-db[{install_extra}]"'
        ) from err
