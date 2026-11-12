"""Path helpers."""
from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv


def resolve_path(file_path: str | os.PathLike | None, env_var: str, description: str) -> str | os.PathLike:
    """Return *file_path* or the value of the environment variable *env_var* if *file_path* is not given.

    Priority follows this order:

    1. The given *file_path*.
    2. An environment variable that is already set.
    3. The variable from a ``.env`` file, searched upwards from the working directory.

    Args:
        file_path: Path given by the caller. ``None`` or an empty string means "not given".
        env_var: Name of the environment variable that holds the fallback path.
        description: What the path points to, e.g. ``"COMFORT database"``. Used in the error message.
    Returns:
        str | os.PathLike: The path to use.
    Raises:
        EnvironmentError: If no path is given and *env_var* is not set.
    """
    # Use the file path if given
    if file_path:
        return file_path

    # Pick up the variable from a .env file, searched upwards from the user's working directory
    load_dotenv(find_dotenv(usecwd=True))
    env_path = os.environ.get(env_var)
    if not env_path:
        raise EnvironmentError(
            f"No {description} path given and the {env_var} environment variable is not set.\n"
            f"Pass a path or set {env_var} (environment variable or .env file), e.g.\n"
            f"    export {env_var}=/path/to/file"
        )
    return env_path
