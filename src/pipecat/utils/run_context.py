#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline execution context variables.

This module provides context variables that are used to track the current
execution context across the pipeline, including workflow run IDs and
turn numbers. These context variables are automatically propagated through
async contexts.
"""

import contextvars

# Context variable for tracking the current workflow run ID across the pipeline
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

# Context variable for tracking the current organization ID across the pipeline
org_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("org_id", default=None)

# Context variable for tracking the current turn number in a conversation
turn_var: contextvars.ContextVar[int] = contextvars.ContextVar("turn", default=0)


def get_current_run_id() -> str | None:
    """Get the current workflow run ID from the context.

    Returns:
        The current workflow run ID or None if not set.
    """
    return run_id_var.get()


def set_current_run_id(run_id: str | int | None) -> None:
    """Set the current workflow run ID in the context.

    Args:
        run_id: The workflow run ID to set (string or int), or None to clear it.
            If an int is provided, it will be converted to string.
    """
    if run_id is not None:
        run_id = str(run_id)
    run_id_var.set(run_id)


def get_current_org_id() -> str | None:
    """Get the current organization ID from the context.

    Returns:
        The current organization ID or None if not set.
    """
    return org_id_var.get()


def set_current_org_id(org_id: str | int | None) -> None:
    """Set the current organization ID in the context.

    Args:
        org_id: The organization ID to set (string or int), or None to clear it.
            If an int is provided, it will be converted to string.
    """
    if org_id is not None:
        org_id = str(org_id)
    org_id_var.set(org_id)


def get_current_turn() -> int:
    """Get the current turn number from the context.

    Returns:
        The current turn number, defaulting to 0 if not set.
    """
    return turn_var.get()


def set_current_turn(turn: int) -> None:
    """Set the current turn number in the context.

    Args:
        turn: The turn number to set.
    """
    turn_var.set(turn)
