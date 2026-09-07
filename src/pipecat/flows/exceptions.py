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
