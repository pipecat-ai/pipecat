#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for the task registry."""

from dataclasses import dataclass


@dataclass
class TaskRegistryEntry:
    """Information about a task in a registry snapshot.

    Parameters:
        name: The task's name.
        parent: Name of the parent task, or None for root tasks.
        active: Whether the task is currently active.
        bridged: Whether the task is bridged.
        started_at: Unix timestamp when the task became ready.
    """

    name: str
    parent: str | None = None
    active: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class TaskReadyData:
    """Information about a registered task.

    Parameters:
        task_name: The name of the task.
        runner: The name of the runner managing this task.
    """

    task_name: str
    runner: str


@dataclass
class TaskErrorData:
    """Information about a task error.

    Parameters:
        task_name: The name of the task that errored.
        error: Description of the error.
    """

    task_name: str
    error: str
