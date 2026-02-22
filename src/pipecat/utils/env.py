#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Environment variable helpers.

This module provides small, centralized parsing helpers for environment variables.
"""

from __future__ import annotations

import os


class InvalidEnvVarValueError(ValueError):
    """Raised when an environment variable value cannot be parsed."""

    def __init__(self, name: str, value: str, expected: str):
        """Initialize an InvalidEnvVarValueError."""
        super().__init__(f"Invalid value for env var {name!r}: {value!r}. Expected {expected}.")
        self.name = name
        self.value = value
        self.expected = expected


def env_truthy(name: str, default: bool = False) -> bool:
    """Interpret an environment variable as a boolean.

    - If the variable is **not set**, returns `default`.
    - If the variable is set to a recognized boolean string, returns the parsed value.
    - Otherwise, raises `InvalidEnvVarValueError`.

    Recognized values (case-insensitive, whitespace ignored):
    - Truthy: "1", "true", "yes", "y", "on"
    - Falsy:  "0", "false", "no", "n", "off", ""
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off", ""}:
        return False

    raise InvalidEnvVarValueError(
        name=name,
        value=raw,
        expected="true or false",
    )
