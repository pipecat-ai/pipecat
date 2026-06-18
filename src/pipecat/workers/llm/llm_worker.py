#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM worker with tool registration.

Provides the `LLMWorker` class that extends `PipelineWorker` with an LLM
pipeline and automatic tool registration.
"""

import functools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    PipelineFlushFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.workers.base_worker import WorkerActivationArgs
from pipecat.workers.llm.tool_decorator import _collect_tools

FunctionCallResultCallback = Callable[..., Any]


@dataclass
class LLMWorkerActivationArgs(WorkerActivationArgs):
    """Activation arguments for LLM workers.

    Attributes:
        messages: LLM context messages to inject on activation.
        run_llm: Whether to run the LLM after appending messages.
            Defaults to True when ``messages`` is set.
    """

    messages: list | None = None
    run_llm: bool | None = None


class LLMWorker(PipelineWorker):
    """Worker with an LLM pipeline and automatic tool registration.

    Methods decorated with ``@tool`` are registered as direct functions
    on the LLM and tracked so that frames queued during tool execution
    can be deferred until all tools complete.

    Example::

        class MyTask(LLMWorker):
            @tool
            async def my_function(self, params, arg: str):
                ...

        worker = MyTask("worker", bus=bus, llm=OpenAILLMService(api_key="..."))
    """

    def __init__(
        self,
        name: str,
        *,
        llm: LLMService[Any],
        pipeline: Pipeline | None = None,
        active: bool = False,
        bridged: tuple[str, ...] | None = None,
        defer_tool_frames: bool = True,
    ):
        """Initialize the LLMWorker.

        Args:
            name: Unique name for this worker.
            llm: The LLM service. ``@tool`` decorated methods are
                automatically registered on it.
            pipeline: Optional pipeline override. When ``None``,
                defaults to ``Pipeline([llm])``. Subclasses can pass a
                custom pipeline that wraps the LLM with additional
                processors.
            active: Whether the worker starts active. Defaults to False.
            bridged: Bridge configuration forwarded to ``PipelineWorker``.
                Pass ``()`` to wrap the LLM pipeline with bus edge
                processors so it can exchange frames with another
                bridged worker.
            defer_tool_frames: Whether to defer frames queued during
                tool execution until all tools complete. Defaults to True.
        """
        # State referenced by tool wrapper closures; must be set before
        # _register_tools wraps any handlers.
        self._defer_tool_frames = defer_tool_frames
        self._tool_call_inflight: int = 0
        self._deferred_frames: deque[tuple[Frame, FrameDirection]] = deque()
        self._closing: bool = False

        self._llm = llm
        self._register_tools(llm)

        pipeline = pipeline if pipeline is not None else Pipeline([self._llm])

        super().__init__(
            pipeline,
            name=name,
            bridged=bridged,
            exclude_frames=(PipelineFlushFrame,),
            enable_rtvi=bridged is None,
            idle_timeout_secs=None,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        # PipelineWorker's __init__ doesn't accept active; configure after.
        self._active = active
        self._pending_activation = active

    @property
    def llm(self) -> LLMService:
        """The LLM service this worker wraps."""
        return self._llm

    @property
    def tool_call_active(self) -> bool:
        """True when one or more ``@tool`` methods are executing."""
        return self._tool_call_inflight > 0

    async def on_activated(self, args: dict | None) -> None:
        """Configure the LLM with tools and activation messages.

        Args:
            args: Optional activation arguments with messages to append.
        """
        await super().on_activated(args)

        activation = LLMWorkerActivationArgs.from_dict(args) if args else LLMWorkerActivationArgs()

        tools = self.build_tools()
        if tools:
            await self.queue_frame(LLMSetToolsFrame(tools=ToolsSchema(standard_tools=tools)))

        if activation.messages:
            run_llm = activation.run_llm if activation.run_llm is not None else True
            await self.queue_frame(
                LLMMessagesAppendFrame(messages=activation.messages, run_llm=run_llm)
            )

    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Queue a frame, deferring delivery until all tools complete (if any).

        When tool calls are in progress, the frame is held in an internal
        queue and delivered automatically once the last tool finishes.
        When no tools are active, the frame is queued immediately.

        Args:
            frame: Any ``Frame`` to deliver.
            direction: Direction the frame should travel. Defaults to
                ``FrameDirection.DOWNSTREAM``.
        """
        if self._defer_tool_frames and self._tool_call_inflight > 0 and not self._closing:
            self._deferred_frames.append((frame, direction))
        else:
            await super().queue_frame(frame, direction)

    def build_tools(self) -> list:
        """Return the tools for this worker's LLM.

        By default, returns all methods decorated with ``@tool``.
        Override to provide additional or different tools.

        Returns:
            List of tool functions.
        """
        return _collect_tools(self)

    async def end(
        self,
        *,
        reason: str | None = None,
        messages: list | None = None,
        result_callback: FunctionCallResultCallback | None = None,
    ) -> None:
        """Request a graceful end of the session.

        When called from a ``@tool`` handler, pass ``params.result_callback`` to
        ensure any pending LLM output is fully delivered before ending.

        Args:
            reason: Optional human-readable reason for ending.
            messages: Optional LLM messages to inject and speak before
                ending. The LLM runs immediately so the output is
                delivered before the session terminates.
            result_callback: The ``result_callback`` from
                `FunctionCallParams`.
        """
        self._closing = True
        await self._finish_function_call(result_callback, messages=messages)
        await super().end(reason=reason)

    async def activate_worker(
        self,
        worker_name: str,
        *,
        args: WorkerActivationArgs | None = None,
        deactivate_self: bool = False,
        messages: list | None = None,
        result_callback: FunctionCallResultCallback | None = None,
    ) -> None:
        """Activate another worker, optionally finishing an in-progress tool call.

        When called from a ``@tool`` handler, pass ``params.result_callback`` to
        ensure any pending LLM output is fully delivered before the target is
        activated.

        Args:
            worker_name: The name of the worker to activate.
            args: Optional ``WorkerActivationArgs`` forwarded to the target
                worker's ``on_activated`` handler.
            deactivate_self: Whether to deactivate this worker before activating
                the target.
            messages: Optional LLM messages to inject and speak before
                activating the target. The LLM runs immediately so the output
                is delivered before the transfer completes.
            result_callback: The ``result_callback`` from `FunctionCallParams`.
        """
        await self._finish_function_call(result_callback, messages=messages)
        await super().activate_worker(worker_name, args=args, deactivate_self=deactivate_self)

    async def process_deferred_tool_frames(
        self, frames: list[tuple[Frame, FrameDirection]]
    ) -> list[tuple[Frame, FrameDirection]]:
        """Process deferred frames before they are flushed.

        Called after all in-flight tools complete, before the deferred
        frames are queued into the pipeline. Override to inspect, modify,
        reorder, or filter the frames.

        Args:
            frames: The deferred frames collected during tool execution.

        Returns:
            The frames to queue. Return the list as-is for default behavior.
        """
        return frames

    def _register_tools(self, llm: LLMService) -> None:
        """Register ``@tool`` methods on the LLM in place."""
        for method in _collect_tools(self):
            tracked = self._track_tool_call(method)
            llm._register_direct_function(
                tracked,
                cancel_on_interruption=method._pipecat_cancel_on_interruption,
                timeout_secs=method._pipecat_timeout_secs,
            )

    def _track_tool_call(self, method: Callable) -> Callable:
        @functools.wraps(method)
        async def wrapper(params, *args, **kwargs):
            self._tool_call_inflight += 1
            try:
                return await method(params, *args, **kwargs)
            finally:
                self._tool_call_inflight = max(0, self._tool_call_inflight - 1)
                if not self._closing and self._tool_call_inflight == 0:
                    await self._flush_deferred_frames()

        return wrapper

    async def _flush_deferred_frames(self) -> None:
        # Wait until the function result frame is really processed. flush_pipeline
        # injects the probe straight into the pipeline, bypassing our queue_frame
        # deferral override.
        await self.flush_pipeline()

        frames = list(self._deferred_frames)
        self._deferred_frames.clear()
        for frame, direction in await self.process_deferred_tool_frames(frames):
            await self.queue_frame(frame, direction)

    async def _finish_function_call(
        self,
        result_callback: FunctionCallResultCallback | None,
        *,
        messages: list | None = None,
    ) -> None:
        """Finish an in-progress function call before taking action.

        Optionally injects LLM messages and flushes the pipeline so the
        output is fully delivered before handing off or ending.

        Args:
            result_callback: The callback from `FunctionCallParams`, or None.
            messages: Optional LLM messages to inject before completing.
        """
        if messages:
            # Bypass our deferral override: this runs inside a tool call, so
            # self.queue_frame would defer the frame and the flush below would
            # return before the LLM output is delivered.
            await super().queue_frame(LLMMessagesAppendFrame(messages=messages, run_llm=True))
            await self.flush_pipeline()

        if not result_callback:
            return

        await result_callback(None, properties=FunctionCallResultProperties(run_llm=False))

        # Wait until the function result frame is really processed.
        await self.flush_pipeline()
