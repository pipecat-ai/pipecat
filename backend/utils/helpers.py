"""Helper utility functions"""

import uuid
import re
from typing import Optional


def generate_session_id() -> str:
    """
    Generate a unique session ID

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format

    Args:
        api_key: API key string

    Returns:
        True if valid format, False otherwise
    """
    # API keys should start with "pk_" and contain only alphanumeric and -_
    pattern = r"^pk_[A-Za-z0-9\-_]{32,}$"
    return bool(re.match(pattern, api_key))


def sanitize_phone_number(phone: str) -> str:
    """
    Sanitize phone number to E.164 format

    Args:
        phone: Phone number string

    Returns:
        Sanitized phone number
    """
    # Remove all non-digit characters except leading +
    sanitized = re.sub(r"[^\d+]", "", phone)

    # Ensure it starts with +
    if not sanitized.startswith("+"):
        sanitized = "+" + sanitized

    return sanitized


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix

    Args:
        text: Input string
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
