#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers shared by the Gradium services."""

from typing import Any, Literal

from loguru import logger

GradiumLogLevel = Literal["debug", "info", "warning", "error"]
"""Lowest level of the server's log messages a session can ask for."""


def log_server_message(msg: dict[str, Any]) -> None:
    """Log a Gradium ``{"type": "log"}`` message at the level the server gave it.

    The server sends these, for example a warning about an unknown or
    deprecated ``json_config`` key, to a session whose setup message asks for
    them with ``log_level``. A level this function does not know is logged as
    a warning.

    Args:
        msg: The decoded server message.
    """
    level = str(msg.get("level", "")).upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        level = "WARNING"
    logger.log(level, f"Gradium {level.lower()}: {msg.get('message')}")
