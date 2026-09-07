#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import importlib.util
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


def _warn_if_standalone_flows_installed() -> None:
    """Flag the deprecated standalone ``pipecat-ai-flows`` package if it is also installed.

    Pipecat Flows now ships inside ``pipecat-ai`` as ``pipecat.flows``. Older
    ``pipecat-ai-flows`` releases allow ``pipecat-ai<2``, so they can end up
    installed next to a Pipecat that already includes Flows — a redundant, easily
    confused setup. The check lives here, in the top-level package init that runs
    on any ``pipecat`` import, rather than in ``pipecat.flows`` — so it also fires
    for apps still importing the standalone ``pipecat_flows`` (which pulls in core
    Pipecat but never ``pipecat.flows``). Detection uses ``find_spec`` to avoid
    importing the standalone package.
    """
    try:
        installed = importlib.util.find_spec("pipecat_flows") is not None
    except (ImportError, ValueError):
        installed = False
    if installed:
        logger.error(
            "The separate `pipecat-ai-flows` package is installed alongside a version "
            "of Pipecat that already includes Pipecat Flows as `pipecat.flows`. You do "
            "not need both — uninstall `pipecat-ai-flows` and import Flows from "
            "`pipecat.flows`."
        )


if _should_log_version_banner():
    logger.info(f"ᓚᘏᗢ Pipecat {__version__} (Python {sys.version}) ᓚᘏᗢ")
    # Gated like the banner: skip the redundant-package check for the pipecat/pc CLI.
    _warn_if_standalone_flows_installed()


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
