"""SQL safety helpers."""
from __future__ import annotations

__all__ = [
    "validate_identifier",
]

import re

# Only allow safe SQL identifiers (letters, digits, underscores)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
