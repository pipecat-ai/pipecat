#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecated module.

The runner now lives in :mod:`pipecat.workers.runner` as
:class:`~pipecat.workers.runner.WorkerRunner`. This module re-exports it
so existing ``from pipecat.pipeline.runner import WorkerRunner`` imports
keep working, and hosts the deprecated :class:`PipelineRunner` alias.
New code should import :class:`~pipecat.workers.runner.WorkerRunner`
from :mod:`pipecat.workers.runner`.
"""

import warnings

from pipecat.workers.runner import WorkerRunner

__all__ = ["PipelineRunner", "WorkerRunner"]


class PipelineRunner(WorkerRunner):
    """Deprecated alias for :class:`~pipecat.workers.runner.WorkerRunner`.

    .. deprecated:: 1.3.0
        Use :class:`~pipecat.workers.runner.WorkerRunner` instead. The
        runner now runs workers (of which
        :class:`~pipecat.pipeline.worker.PipelineWorker` is one kind), not
        just pipelines. ``PipelineRunner`` will be removed in a future
        release.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the worker runner (deprecated)."""
        warnings.warn(
            "PipelineRunner is deprecated, use WorkerRunner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
