#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys
from importlib.metadata import version as lib_version

from loguru import logger

__version__ = lib_version("pipecat-ai")


def _should_log_version_banner() -> bool:
    """Whether to emit the import-time version banner.

    The banner is a handy startup marker for bots and interactive use, but it's
    just noise for the ``pipecat`` / ``pc`` CLI — it even corrupts ``--help``
    output — so suppress it there.
    """
    program = os.path.splitext(os.path.basename(sys.argv[0] or ""))[0]
    return program not in ("pipecat", "pc")


if _should_log_version_banner():
    logger.info(f"ᓚᘏᗢ Pipecat {__version__} (Python {sys.version}) ᓚᘏᗢ")


def version() -> str:
    """Returns the Pipecat version."""
    return __version__


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
