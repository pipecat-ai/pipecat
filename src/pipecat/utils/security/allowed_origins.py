#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Origin validation utilities."""

import os


def default_allowed_origins() -> list[str]:
    """Return allowed origins from the ``PIPECAT_ALLOWED_ORIGINS`` env var.

    Parses a comma-separated list of origin strings. Returns an empty list
    (allow all) when the variable is unset or empty.
    """
    val = os.getenv("PIPECAT_ALLOWED_ORIGINS", "")
    return [o.strip() for o in val.split(",") if o.strip()]


def is_origin_allowed(origin: str, allowed_origins: list[str]) -> bool:
    """Return whether ``origin`` is permitted by ``allowed_origins``.

    Args:
        origin: The value of the ``Origin`` header, or an empty string if absent.
        allowed_origins: List of allowed origin strings. An empty list allows
            all origins. When non-empty, a missing or disallowed origin is
            rejected.
    """
    if not allowed_origins:
        return True
    return origin.lower() in {o.lower() for o in allowed_origins}
