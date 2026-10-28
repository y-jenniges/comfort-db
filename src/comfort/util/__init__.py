"""Utility helpers (SQL safety)."""
from __future__ import annotations

from .sqlite_utils import validate_identifier

__all__ = ["validate_identifier"]
