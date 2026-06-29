"""Structured, rich-backed logging for the aqi_india package.

A single :func:`get_logger` factory configures a process-wide
:class:`rich.logging.RichHandler` (falling back to the stdlib handler if rich is
unavailable) so every module logs with consistent timestamps, levels and colour.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_DEFAULT_LEVEL = os.environ.get("AQI_LOG_LEVEL", "INFO").upper()


def _configure_root(level: str) -> None:
    """Attach a single rich handler to the package root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("aqi_india")
    root.setLevel(level)
    root.propagate = False
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=True
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except Exception:  # pragma: no cover - rich is a hard dep but stay safe
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str | None = None, level: str | None = None) -> logging.Logger:
    """Return a configured logger under the ``aqi_india`` namespace.

    Args:
        name: Dotted suffix (e.g. ``"ingest.s5p"``). ``None`` returns the package
            root logger.
        level: Optional level override (e.g. ``"DEBUG"``).

    Returns:
        A :class:`logging.Logger` with the shared rich handler attached.
    """
    _configure_root(level or _DEFAULT_LEVEL)
    logger_name = "aqi_india" if not name else f"aqi_india.{name}"
    logger = logging.getLogger(logger_name)
    if level:
        logger.setLevel(level.upper())
    return logger


__all__ = ["get_logger"]
