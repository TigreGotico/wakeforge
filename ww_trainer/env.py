"""Load environment variables from .env file.

Walks up from the caller's location to find a .env file, then populates
os.environ with any keys not already set. Call load_env() early in any
training script before importing mlflow or other authenticated clients.
"""
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def env_default(key: str, default: T, cast: Optional[Callable[[str], T]] = None) -> T:
    """Return the env-var *key* cast to *cast*, or *default* if unset/invalid.

    Call this **after** :func:`load_env` so that .env values are already in
    ``os.environ``.

    Args:
        key:     Environment variable name (e.g. ``"WW_EPOCHS"``).
        default: Value to use when the variable is absent or unparseable.
        cast:    Callable that converts the string value (e.g. ``int``,
                 ``float``, ``str``). Omit for plain string variables.

    Example::

        from ww_trainer.env import load_env, env_default
        load_env()
        parser.add_argument("--epochs", type=int,
                            default=env_default("WW_EPOCHS", 30, int))
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    if cast is None:
        return raw  # type: ignore[return-value]
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return default


def load_env(env_file: str | Path | None = None) -> None:
    """Load key=value pairs from *env_file* (or auto-discovered .env) into os.environ.

    Keys already present in the environment are not overwritten, so
    shell-level overrides still take precedence.

    Args:
        env_file: Explicit path to .env file. If None, searches upward from
                  this module's directory for a file named ``.env``.
    """
    if env_file is None:
        # Walk up from project root (this file lives in ww_trainer/)
        candidate = Path(__file__).parent.parent / ".env"
        if not candidate.exists():
            return
        env_file = candidate

    env_file = Path(env_file)
    if not env_file.exists():
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
