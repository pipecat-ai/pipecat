#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecated module.

.. deprecated:: 1.3.0
    Import from :mod:`pipecat.pipeline.worker` instead. Constructing
    :class:`PipelineTask` directly is also deprecated; use
    :class:`PipelineWorker`.
"""

from pipecat.pipeline.worker import (
    IdleFrameObserver,
    PipelineParams,
    PipelineTask,
    PipelineTaskParams,
    PipelineWorker,
    PipelineWorkerParams,
)

__all__ = [
    "IdleFrameObserver",
    "PipelineParams",
    "PipelineTask",
    "PipelineTaskParams",
    "PipelineWorker",
    "PipelineWorkerParams",
]
