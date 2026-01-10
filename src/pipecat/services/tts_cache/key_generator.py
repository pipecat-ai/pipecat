#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cache key generation for TTS caching."""

import hashlib
import json
from typing import Any, Dict, FrozenSet, Optional

# Settings to exclude from cache key (sensitive or non-deterministic)
EXCLUDED_SETTINGS: FrozenSet[str] = frozenset(
    {
        "api_key",
        "api_secret",
        "auth_token",
        "authorization",
        "credentials",
        "timeout",
        "connect_timeout",
        "read_timeout",
        "retry_count",
        "max_retries",
        "log_level",
        "debug",
        "verbose",
        "random_seed",
        "seed",
        "session_id",
        "request_id",
    }
)


def normalize_text(text: str) -> str:
    """Normalize text for consistent cache key generation."""
    return " ".join(text.strip().split())


def normalize_value(value: Any) -> Any:
    """Normalize a setting value for consistent hashing."""
    if value is None:
        return None
    elif isinstance(value, float):
        return round(value, 6)
    elif isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    else:
        return value


def filter_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out sensitive/non-deterministic settings."""
    return {
        k: normalize_value(v)
        for k, v in settings.items()
        if k not in EXCLUDED_SETTINGS and v is not None
    }


def generate_cache_key(
    text: str,
    voice_id: str,
    model: str,
    sample_rate: int,
    settings: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
) -> str:
    """Generate a deterministic SHA-256 cache key from TTS parameters."""
    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Cannot generate cache key for empty text")

    filtered_settings = filter_settings(settings) if settings else {}

    key_data = {
        "text": normalized_text,
        "voice_id": voice_id,
        "model": model,
        "sample_rate": sample_rate,
    }

    if filtered_settings:
        key_data["settings"] = filtered_settings

    if namespace:
        key_data["namespace"] = namespace

    key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()
