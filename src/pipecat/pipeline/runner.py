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

from pipecat.utils.deprecation import deprecated
from pipecat.workers.runner import WorkerRunner

__all__ = ["PipelineRunner", "WorkerRunner"]


@deprecated(
    "`PipelineRunner` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `WorkerRunner` instead."
)
class PipelineRunner(WorkerRunner):
    """Deprecated alias for :class:`~pipecat.workers.runner.WorkerRunner`.

    .. deprecated:: 1.3.0
        Use :class:`~pipecat.workers.runner.WorkerRunner` instead.
        Will be removed in 2.0.0. The :class:`PipelineRunner` now runs workers
        (of which :class:`~pipecat.pipeline.worker.PipelineWorker` is one kind),
        not just pipelines.
    """

    pass
