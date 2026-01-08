#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from __future__ import annotations

import platform
from importlib.metadata import version as lib_version


def sdk_headers() -> dict[str, str]:
    """SDK identification headers for upstream providers."""
    try:
        pipecat_version = lib_version("pipecat-ai")
    except Exception:
        pipecat_version = "unknown"

    return {
        "X-SDK-Source": "Pipecat",
        "X-SDK-Version": pipecat_version,
        "SDK-Language": "Python",
        "sdk-language-version": platform.python_version(),
    }
