#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Compatibility helpers for the httpx and httpx2 HTTP client families.

Service SDKs are migrating from httpx to httpx2, and each installs only the
family it uses: ``openai < 3`` brings httpx, ``openai >= 3`` brings httpx2.
Pipecat supports both, so the objects it hands to an SDK client — and the
exceptions it catches from one — can't name a single family at import time.
"""

import importlib
from typing import Any, TypeAlias

__all__ = ["AsyncHTTPClient", "TIMEOUT_EXCEPTIONS", "connection_limits"]


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_httpx = _optional_module("httpx")
_httpx2 = _optional_module("httpx2")

#: Type of a caller-supplied async HTTP client. The class belongs to whichever
#: family the caller's SDK uses, so there is no single type to name here.
AsyncHTTPClient: TypeAlias = Any

#: Timeout exception types to catch around SDK calls. SDKs wrap request-level
#: timeouts in their own error type, but a timeout while iterating a response
#: stream surfaces as the underlying transport exception, which belongs to
#: whichever family the SDK is built on.
TIMEOUT_EXCEPTIONS: tuple[type[Exception], ...] = tuple(
    module.TimeoutException for module in (_httpx, _httpx2) if module is not None
)


def connection_limits(**kwargs: Any) -> Any:
    """Build a connection-limits object accepted by either client family.

    Args:
        **kwargs: Limits fields (``max_connections``,
            ``max_keepalive_connections``, ``keepalive_expiry``).

    Returns:
        A ``Limits`` instance from the installed HTTP client family.
    """
    # Both families read a limits object field by field, so either one's Limits
    # works with either client; take whichever is installed.
    module = _httpx or _httpx2
    if module is None:
        raise RuntimeError("Neither httpx nor httpx2 is installed")
    return module.Limits(**kwargs)
