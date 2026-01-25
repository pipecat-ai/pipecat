#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""API key validation utilities for AI services."""

from typing import Optional


class APIKeyError(ValueError):
    """Exception raised when an API key is missing or invalid."""

    pass


def validate_api_key(
    api_key: Optional[str],
    service_name: str,
    allow_none: bool = False,
    env_var_name: Optional[str] = None,
) -> None:
    """Validate that an API key is not None or blank.

    This function raises an exception if the API key is missing or empty,
    improving developer experience by catching configuration errors early.

    Args:
        api_key: The API key to validate.
        service_name: Name of the service (e.g., "OpenAI", "Anthropic").
        allow_none: If True, allows None values (for services that use environment variables).
        env_var_name: Optional name of the environment variable to suggest in error message.

    Raises:
        APIKeyError: If the API key is None or blank (unless allow_none is True).

    Example:
        >>> validate_api_key(api_key, "OpenAI", env_var_name="OPENAI_API_KEY")
        >>> # Raises APIKeyError if api_key is None or empty
    """
    # Allow None if the service supports environment variable fallback
    if allow_none and api_key is None:
        return

    # Check for None or empty/whitespace-only strings
    if api_key is None or (isinstance(api_key, str) and not api_key.strip()):
        env_hint = f" Set the {env_var_name} environment variable or pass it explicitly." if env_var_name else ""
        raise APIKeyError(
            f"API key for {service_name} is missing or empty.{env_hint} "
            f"Please provide a valid API key to use this service."
        )
