#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from importlib.metadata import version

from loguru import logger

__version__ = version("pipecat-ai")

logger.info(f"ᓚᘏᗢ Pipecat {__version__} (Python {sys.version}) ᓚᘏᗢ")

# We replace `asyncio.wait_for()` for `wait_for2.wait_for()` for Python < 3.12.
#
# In Python 3.12, `asyncio.wait_for()` is implemented in terms of
# `asyncio.timeout()` which fixed a bunch of issues. However, this was never
# backported (because of the lack of `async.timeout()`) and there are still many
# remainig issues, specially in Python 3.10, in `async.wait_for()`.
#
# See https://github.com/python/cpython/pull/98518

import asyncio

if sys.version_info < (3, 12):
    import wait_for2

    # Replace asyncio.wait_for.
    asyncio.wait_for = wait_for2.wait_for
