#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys
from importlib.metadata import version

from loguru import logger

__version__ = version("pipecat-ai")

event_loop = "asyncio"

if sys.platform in ("linux", "darwin"):
    try:
        import asyncio

        import uvloop

        event_loop = f"uvloop {uvloop.__version__}"
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        logger.debug(f"Couldn't find `uvloop`")
        pass

logger.info(f"ᓚᘏᗢ Pipecat {__version__} ({event_loop}; Python {sys.version}) ᓚᘏᗢ")
