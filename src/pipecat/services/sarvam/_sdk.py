#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import platform
from typing import Dict

from pipecat import version as pipecat_version


def sdk_headers() -> Dict[str, str]:
    """SDK identification headers for upstream providers."""
    return {
        "User-Agent": f"Pipecat/{pipecat_version()} Python/{platform.python_version()}",
    }
