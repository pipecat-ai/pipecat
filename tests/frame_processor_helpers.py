#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helpers for setting up frame processors outside a pipeline."""

from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import BaseTaskManager


def frame_processor_setup(
    task_manager: BaseTaskManager | None = None, **kwargs
) -> FrameProcessorSetup:
    """Build a setup configuration for a processor used outside a pipeline.

    Tests that drive a single processor, controller or turn strategy need a
    :class:`FrameProcessorSetup` without a surrounding :class:`PipelineWorker`.
    Fields a test doesn't exercise can be left unset.

    Args:
        task_manager: The task manager the processor should run its tasks on.
            Omit it when the processor under test creates no tasks.
        **kwargs: Any :class:`FrameProcessorSetup` field to override.

    Returns:
        A setup carrying a system clock and no pipeline worker.
    """
    kwargs.setdefault("clock", SystemClock())
    kwargs.setdefault("pipeline_worker", None)
    return FrameProcessorSetup(task_manager=task_manager, **kwargs)  # pyright: ignore[reportArgumentType]
