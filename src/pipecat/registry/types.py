#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared types for the worker registry."""

from dataclasses import dataclass


@dataclass
class WorkerRegistryEntry:
    """Information about a worker in a registry snapshot.

    Parameters:
        name: The worker's name.
        parent: Name of the parent worker, or None for root tasks.
        active: Whether the worker is currently active.
        bridged: Whether the worker is bridged.
        started_at: Unix timestamp when the worker became ready.
    """

    name: str
    parent: str | None = None
    active: bool = False
    bridged: bool = False
    started_at: float | None = None


@dataclass
class WorkerReadyData:
    """Information about a registered worker.

    Parameters:
        worker_name: The name of the worker.
        runner: The name of the runner managing this worker.
    """

    worker_name: str
    runner: str


@dataclass
class WorkerErrorData:
    """Information about a worker error.

    Parameters:
        worker_name: The name of the worker that errored.
        error: Description of the error.
    """

    worker_name: str
    error: str
