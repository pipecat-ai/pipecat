#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Error classification shared by services.

Provides a provider-agnostic vocabulary for *why* an operation failed, so that
callers can tell a transient failure (worth retrying) from a configuration
failure (retrying will never help).
"""

import socket
from enum import Enum

__all__ = [
    "ErrorCategory",
    "classify_exception",
    "classify_status_code",
    "extract_status_code",
]


class ErrorCategory(Enum):
    """Why an operation failed, independent of the provider that failed it.

    Parameters:
        UNKNOWN: The cause could not be determined.
        AUTHENTICATION: Credentials are missing or invalid.
        AUTHORIZATION: Credentials are valid but lack access to the resource.
        INVALID_REQUEST: The request itself is malformed or names something that
            doesn't exist, such as an unknown model or voice.
        RATE_LIMIT: Too many requests were sent in too short a window.
        QUOTA: The account's credit or usage allowance is exhausted.
        CONNECTIVITY: The service could not be reached.
        SERVER: The provider reported an internal failure.
    """

    UNKNOWN = "unknown"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    CONNECTIVITY = "connectivity"
    SERVER = "server"

    @property
    def is_configuration_error(self) -> bool:
        """Whether the failure stems from how the service was configured.

        Configuration errors are terminal: the same request will keep failing
        until credentials or settings change, so retrying is pointless.
        """
        return self in _CONFIGURATION_CATEGORIES


_CONFIGURATION_CATEGORIES = frozenset(
    {
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.INVALID_REQUEST,
    }
)

_STATUS_CODE_CATEGORIES = {
    400: ErrorCategory.INVALID_REQUEST,
    401: ErrorCategory.AUTHENTICATION,
    402: ErrorCategory.QUOTA,
    403: ErrorCategory.AUTHORIZATION,
    404: ErrorCategory.INVALID_REQUEST,
    422: ErrorCategory.INVALID_REQUEST,
    429: ErrorCategory.RATE_LIMIT,
}

_CONNECTIVITY_EXCEPTIONS = (ConnectionError, TimeoutError, socket.gaierror)


def classify_status_code(status_code: int) -> ErrorCategory:
    """Classify an HTTP status code.

    Args:
        status_code: The HTTP status code reported by the provider.

    Returns:
        The matching category, or `ErrorCategory.UNKNOWN` if the code carries
        no provider-independent meaning.
    """
    category = _STATUS_CODE_CATEGORIES.get(status_code)
    if category:
        return category
    if 500 <= status_code < 600:
        return ErrorCategory.SERVER
    return ErrorCategory.UNKNOWN


def extract_status_code(exception: BaseException) -> int | None:
    """Extract an HTTP status code from an exception, if it carries one.

    Recognizes the attribute shapes used by the HTTP and websocket libraries
    services build on: a ``response`` holding ``status_code`` (httpx,
    ``websockets.exceptions.InvalidStatus``) or ``status`` (aiohttp), and those
    same attributes directly on the exception (provider SDKs).

    Args:
        exception: The exception to inspect.

    Returns:
        The status code, or None if the exception doesn't carry one.
    """
    response = getattr(exception, "response", None)
    candidates = (
        getattr(response, "status_code", None),
        getattr(response, "status", None),
        getattr(exception, "status_code", None),
        getattr(exception, "status", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def classify_exception(exception: BaseException) -> ErrorCategory:
    """Classify an exception raised while talking to a provider.

    Args:
        exception: The exception to classify.

    Returns:
        The matching category, or `ErrorCategory.UNKNOWN` when the exception is
        not recognized. Providers that raise SDK-specific exceptions carrying no
        status code need their own classification.
    """
    status_code = extract_status_code(exception)
    if status_code is not None:
        return classify_status_code(status_code)
    if isinstance(exception, _CONNECTIVITY_EXCEPTIONS):
        return ErrorCategory.CONNECTIVITY
    return ErrorCategory.UNKNOWN
