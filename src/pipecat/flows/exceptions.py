#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom exceptions for the conversation flow system.

This module defines the exception hierarchy used throughout the flow system
for better error handling and debugging. All exceptions inherit from FlowError
to provide a common base for flow-related errors.
"""

from dataclasses import dataclass


class FlowError(Exception):
    """Base exception for all flow-related errors.

    This is the parent class for all flow system exceptions. Use this
    for generic flow errors or when a more specific exception doesn't apply.
    """

    pass


class FlowInitializationError(FlowError):
    """Raised when flow initialization fails.

    This exception occurs during flow manager setup, typically due to
    invalid configuration, missing dependencies, or initialization errors.
    """

    pass


class FlowTransitionError(FlowError):
    """Raised when a state transition fails.

    This exception occurs when transitioning between nodes fails due to
    invalid node configurations, missing target nodes, or transition errors.
    """

    pass


class InvalidFunctionError(FlowError):
    """Raised when an invalid or unavailable function is called.

    This exception occurs when attempting to call functions that are not
    properly registered, have invalid signatures, or cannot be found.
    """

    pass


class ActionError(FlowError):
    """Raised when an action execution fails.

    This exception occurs during action execution, including built-in actions
    like TTS or custom actions, due to invalid configuration or execution errors.
    """

    pass


@dataclass
class FlowProblem:
    """One reference a flow config makes that its tools or variables do not satisfy.

    Parameters:
        code: Stable identifier for the kind of problem: ``missing_tool``,
            ``invalid_tool``, ``missing_handler``, or ``missing_variable``.
        message: Human-readable description naming the node, function, or
            variable involved.
        node: The node the problem is about, when there is one.
        function: The function entry the problem is about, when there is one.
    """

    code: str
    message: str
    node: str | None = None
    function: str | None = None


class FlowReferenceError(FlowError):
    """Raised when a flow config's references cannot all be resolved.

    Constructing a :class:`~pipecat.flows.Flow` checks every tool, action
    handler, and template variable the config names and raises this once,
    with every unresolved reference, rather than stopping at the first.
    """

    def __init__(self, problems: list[FlowProblem]):
        """Initialize with the unresolved references.

        Args:
            problems: Every unresolved reference, in the order found.
        """
        self.problems = problems
        lines = "\n".join(f"- {p.message}" for p in problems)
        count = f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"
        super().__init__(f"flow config has {count}:\n{lines}")
