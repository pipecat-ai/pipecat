#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers shared by the Zoom AI services."""

import uuid
from urllib.parse import urlparse

from loguru import logger

from pipecat import version as pipecat_version

_PLAINTEXT_SCHEMES = ("http", "ws")
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def new_tracking_id() -> str:
    """Return a new ``x-zm-trackingid``, which Zoom support uses to find a request or session."""
    return f"zoom_ai_community_pipecat_{uuid.uuid4()}"


def request_headers(api_key: str, tracking_id: str) -> dict[str, str]:
    """Return the headers every Zoom AI Services request carries.

    Args:
        api_key: Zoom AI Services API key or JWT.
        tracking_id: The request's ``x-zm-trackingid``.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"pipecat/{pipecat_version()}",
        "x-zm-trackingid": tracking_id,
    }


def check_url(url: str, *, service: str, data: str) -> str:
    """Return ``url``, warning if it sends the API key unencrypted to a remote host.

    Args:
        url: The endpoint URL to check.
        service: The service's name, for the warning.
        data: What the service sends besides the API key, for the warning.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in _PLAINTEXT_SCHEMES and parsed.hostname not in _LOOPBACK_HOSTS:
        logger.warning(
            f"{service} url is not encrypted ({scheme}://); "
            f"the API key and {data} are sent in plaintext"
        )
    return url
