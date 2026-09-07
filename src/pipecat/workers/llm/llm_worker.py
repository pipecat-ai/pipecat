#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM worker with tool registration.

Provides the `LLMWorker` class that extends `PipelineWorker` with an LLM
pipeline and automatic tool registration.
"""

import contextvars
import functools
import warnings
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
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

    Parameters:
        messages: LLM context messages to inject on activation.
        run_llm: Whether to run the LLM after appending messages.
            Defaults to True when ``messages`` is set.
    """

    messages: list | None = None
    run_llm: bool | None = None


#: The worker whose ``@tool`` handler is running, for the duration of that
#: handler. A context variable rather than a counter so it answers "did this
#: come from a tool", not "is a tool running somewhere": the bus dispatch tasks
#: were created before any tool and carry a context where this is unset, so
#: their frames are never mistaken for a handler's. It holds the worker rather
#: than a flag because a handler can reach another worker, and only the worker
#: running the tool will release what it holds.
_IN_TOOL_CALL: contextvars.ContextVar["LLMWorker | None"] = contextvars.ContextVar(
    "pipecat_in_tool_call", default=None
)


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
        # A handover or ending a tool asked for, held until its call is done.
        self._pending_handover: Callable[[], Awaitable[None]] | None = None
        self._closing: bool = False

        self._llm = llm
        self._register_tools(llm)

        pipeline = pipeline if pipeline is not None else Pipeline([self._llm])

        super().__init__(
            pipeline,
            name=name,
            bridged=bridged,
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
        """Queue a frame, holding it if a tool handler queued it.

        A frame queued from inside one of this worker's ``@tool`` handlers,
        or from anything that handler awaits, is held and delivered once the
        last tool finishes. Frames from anywhere else are queued
        immediately: the worker's own traffic and frames arriving over the
        bus, which run outside any handler's context, and frames a handler
        on a different worker queues here, which this worker would never
        release.

        Args:
            frame: Any ``Frame`` to deliver.
            direction: Direction the frame should travel. Defaults to
                ``FrameDirection.DOWNSTREAM``.
        """
        if self._defer_tool_frames and _IN_TOOL_CALL.get() is self and not self._closing:
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

        When called from a ``@tool`` handler, deliver the function call result
        first with ``await params.result_callback(result)``: the LLM output it
        triggers is delivered before the session ends.

        Args:
            reason: Optional human-readable reason for ending.
            messages: Optional LLM messages to inject and speak before
                ending. The LLM runs immediately so the output is
                delivered before the session terminates.

                .. deprecated:: 1.8.0
                    Call ``params.result_callback(result)`` before :meth:`end`
                    instead. Will be removed in 2.0.0.
            result_callback: The ``result_callback`` from
                `FunctionCallParams`.

                .. deprecated:: 1.8.0
                    Call ``params.result_callback(result)`` before :meth:`end`
                    instead. Will be removed in 2.0.0.
        """
        self._closing = True
        if messages is not None or result_callback is not None:
            warnings.warn(
                "Passing messages or result_callback to LLMWorker.end() is deprecated, "
                "call params.result_callback(result) before end() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            await self._finish_function_call(result_callback, messages=messages)
        await self._after_tool_calls(lambda: super(LLMWorker, self).end(reason=reason))

    async def activate_worker(
        self,
        worker_name: str,
        *,
        args: WorkerActivationArgs | None = None,
        deactivate_self: bool = False,
        messages: list | None = None,
        result_callback: FunctionCallResultCallback | None = None,
    ) -> None:
        """Activate another worker, draining this worker's pipeline to hand over.

        When called from a ``@tool`` handler, deliver the function call result
        first with ``await params.result_callback(result)``: the output it
        triggers is delivered before the target is activated. The handover
        itself waits until the tool call asking for it has finished.

        Args:
            worker_name: The name of the worker to activate.
            args: Optional ``WorkerActivationArgs`` forwarded to the target
                worker's ``on_activated`` handler.
            deactivate_self: Whether to deactivate this worker before activating
                the target. Deactivating this worker drains its pipeline
                first; staying active does not. A worker that stays active and
                wants to drain anyway can call :meth:`flush_pipeline` before
                this.
            messages: Optional LLM messages to inject and deliver before
                activating the target. The LLM runs immediately so the output
                is delivered before the transfer completes.

                .. deprecated:: 1.8.0
                    Call ``params.result_callback(result)`` before
                    :meth:`activate_worker` instead. Will be removed in 2.0.0.
            result_callback: The ``result_callback`` from `FunctionCallParams`.

                .. deprecated:: 1.8.0
                    Call ``params.result_callback(result)`` before
                    :meth:`activate_worker` instead. Will be removed in 2.0.0.
        """
        if messages is not None or result_callback is not None:
            warnings.warn(
                "Passing messages or result_callback to LLMWorker.activate_worker() is "
                "deprecated, call params.result_callback(result) before activate_worker() "
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            await self._finish_function_call(result_callback, messages=messages)
        await self._after_tool_calls(
            lambda: super(LLMWorker, self).activate_worker(
                worker_name, args=args, deactivate_self=deactivate_self
            )
        )

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

    async def _after_tool_calls(self, handover: Callable[[], Awaitable[None]]) -> None:
        """Run a handover once no tool call is in flight.

        A tool asking to end or hand over is still running when it asks, and
        deactivating a worker mid-call leaves the rest of that call to a worker
        nobody is listening to. Held here and run once the last call finishes;
        outside a call there is nothing to wait for, so it runs now.

        Args:
            handover: What to do once the calls are done.
        """
        if self._tool_call_inflight:
            self._pending_handover = handover
        else:
            await handover()

    def _track_tool_call(self, method: Callable) -> Callable:
        @functools.wraps(method)
        async def wrapper(params, *args, **kwargs):
            self._tool_call_inflight += 1
            token = _IN_TOOL_CALL.set(self)
            try:
                return await method(params, *args, **kwargs)
            finally:
                _IN_TOOL_CALL.reset(token)
                self._tool_call_inflight = max(0, self._tool_call_inflight - 1)
                if self._tool_call_inflight == 0:
                    if not self._closing:
                        await self._flush_deferred_frames()
                    handover, self._pending_handover = self._pending_handover, None
                    if handover:
                        await handover()

        return wrapper

    async def _flush_deferred_frames(self) -> None:
        frames = list(self._deferred_frames)
        self._deferred_frames.clear()

        # Held frames go back in behind the function call result, so wait for
        # that to be processed first. With nothing held there is nothing to
        # order, and the round-trip would buy nothing.
        if frames:
            await self.flush_pipeline()

        for frame, direction in await self.process_deferred_tool_frames(frames):
            await self.queue_frame(frame, direction)

    async def _finish_function_call(
        self,
        result_callback: FunctionCallResultCallback | None,
        *,
        messages: list | None = None,
    ) -> None:
        """Finish an in-progress function call before taking action.

        Optionally injects LLM messages and waits for the output they
        produce, so the call is not settled in the middle of it. The
        caller is ``end`` or ``activate_worker``, both of which drain the
        pipeline afterwards, so the settled call needs no further wait
        here.

        Args:
            result_callback: The callback from `FunctionCallParams`, or None.
            messages: Optional LLM messages to inject before completing.
        """
        if messages:
            # Bypass our deferral override: this runs inside a tool call, so
            # self.queue_frame would park the frame instead of queueing it,
            # and the flush below would return before the output is delivered.
            await super().queue_frame(LLMMessagesAppendFrame(messages=messages, run_llm=True))
            if not await self.flush_pipeline():
                logger.warning(
                    f"{self}: settling the function call before its output was delivered"
                )

        if not result_callback:
            return

        await result_callback(None, properties=FunctionCallResultProperties(run_llm=False))
