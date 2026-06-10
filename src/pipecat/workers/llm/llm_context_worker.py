#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM worker with a built-in `LLMContext` and aggregator pair.

Provides the `LLMContextWorker` class that extends `LLMWorker` with a
self-contained conversation context, removing the need for subclasses
to manually wire `LLMContextAggregatorPair`.
"""

from typing import Any

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.workers.llm.llm_worker import LLMWorker


class LLMContextWorker(LLMWorker):
    """LLM worker that owns an `LLMContext` and a context aggregator pair.

    Useful for workers that need to track their own conversation history,
    typically workers that run their own LLM pipeline outside of a shared
    transport pipeline. Subclasses do not need to instantiate the context
    or aggregators themselves; the pipeline is built as
    ``[user_aggregator, llm, assistant_aggregator]`` automatically.

    Example::

        worker = LLMContextWorker(
            "worker",
            llm=OpenAILLMService(...),
        )

        @worker.assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def _on_stopped(aggregator, message):
            ...
    """

    def __init__(
        self,
        name: str,
        *,
        llm: LLMService[Any],
        active: bool = False,
        bridged: tuple[str, ...] | None = None,
        defer_tool_frames: bool = True,
        context: LLMContext | None = None,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
    ):
        """Initialize the LLMContextWorker.

        Args:
            name: Unique name for this worker.
            llm: The LLM service.
            active: Whether the worker starts active. Defaults to False.
            bridged: Bridge configuration forwarded to ``PipelineWorker``.
                Pass ``()`` to wrap the pipeline with bus edges so it
                can exchange frames with another bridged worker.
            defer_tool_frames: Whether to defer frames queued during
                tool execution until all tools complete. Defaults to True.
            context: Optional pre-built `LLMContext`. When omitted, a
                fresh empty context is created.
            user_params: Optional parameters for the user aggregator.
            assistant_params: Optional parameters for the assistant
                aggregator.
        """
        self._context = context or LLMContext()
        self._aggregators = LLMContextAggregatorPair(
            self._context,
            user_params=user_params,
            assistant_params=assistant_params,
        )

        pipeline = Pipeline(
            [
                self._aggregators.user(),
                llm,
                self._aggregators.assistant(),
            ]
        )

        super().__init__(
            name,
            llm=llm,
            pipeline=pipeline,
            active=active,
            bridged=bridged,
            defer_tool_frames=defer_tool_frames,
        )

    @property
    def context(self) -> LLMContext:
        """The `LLMContext` owned by this worker."""
        return self._context

    @property
    def user_aggregator(self) -> LLMUserAggregator:
        """The user-side context aggregator."""
        return self._aggregators.user()

    @property
    def assistant_aggregator(self) -> LLMAssistantAggregator:
        """The assistant-side context aggregator."""
        return self._aggregators.assistant()
