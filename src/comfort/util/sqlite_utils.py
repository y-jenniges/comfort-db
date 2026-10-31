"""SQL safety helpers and VACUUM."""
from __future__ import annotations

__all__ = [
    "validate_identifier",
    "vacuum",
]

import logging
import re
import sqlite3

logger = logging.getLogger(__name__)

# Only allow safe SQL identifiers (letters, digits, underscores)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def vacuum(conn: sqlite3.Connection) -> None:
    """Run VACUUM on the database. Logs a warning if it fails (e.g. insufficient disk space)."""
    try:
        conn.execute("VACUUM;")
    except sqlite3.OperationalError as e:
        logger.warning("VACUUM failed: %s", e)


def validate_identifier(identifier: str) -> None:
    """Raise ValueError if *identifier* is not a safe SQL identifier.

    Accepts only ASCII letters, digits, and underscores (must start with
    a letter or underscore). Use this on any externally-supplied table or
    column name before interpolating it into a SQL string.
    """
    if not isinstance(identifier, str) or not _IDENTIFIER_RE.match(identifier):
        raise ValueError(
            f"Invalid SQL identifier: {identifier!r}. "
            "Only letters, digits, and underscores are allowed."
        )
