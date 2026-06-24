#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Pipecat Flows - Structured conversation framework for Pipecat.

This package provides a framework for building structured conversations in Pipecat.
The FlowManager handles conversation flows with support for state management,
function calling, and cross-provider compatibility.

Pipecat Flows determines conversation structure at runtime, supporting function
calling, action execution, and seamless transitions between conversation states.
"""

import importlib.util

from loguru import logger

from .exceptions import (
    ActionError,
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from .manager import FlowManager
from .types import (
    ActionConfig,
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowFunctionHandler,
    FlowResult,
    FlowsDirectFunction,
    FlowsFunctionSchema,
    LegacyFunctionHandler,
    NodeConfig,
    ZeroArgFunctionHandler,
    flows_direct_function,
    flows_tool_options,
)


def _warn_if_standalone_flows_installed() -> None:
    """Flag the deprecated standalone ``pipecat-ai-flows`` package if it is also installed.

    Pipecat Flows now ships inside ``pipecat-ai`` as ``pipecat.flows``. Older
    ``pipecat-ai-flows`` releases allow ``pipecat-ai<2``, so they can end up
    installed next to a Pipecat that already includes Flows — a redundant, easily
    confused setup. Surface it so the user removes the separate package. Detection
    uses ``find_spec`` rather than importing it, to avoid running the standalone
    package (and its own deprecation warning).
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


_warn_if_standalone_flows_installed()


__all__ = [
    # Flow Manager
    "FlowManager",
    # Types
    "ActionConfig",
    "ContextStrategy",
    "ContextStrategyConfig",
    "FlowArgs",
    "FlowFunctionHandler",
    "FlowResult",
    "ConsolidatedFunctionResult",
    "FlowsFunctionSchema",
    "LegacyFunctionHandler",
    "FlowsDirectFunction",
    "NodeConfig",
    "ZeroArgFunctionHandler",
    "flows_tool_options",
    "flows_direct_function",
    # Exceptions
    "FlowError",
    "FlowInitializationError",
    "FlowTransitionError",
    "InvalidFunctionError",
    "ActionError",
]
