#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for LLM extensions: pluggable features or behaviors of an LLM service."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import Frame, FunctionCallFromLLM, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallRegistryItem, LLMService


@dataclass
class LLMExtensionTool:
    """A tool an extension adds to the service's built-in tools.

    Built-in tools go to the model on every inference, beside the tools the
    context advertises. Once a tool is added its name is reserved: an
    application cannot register a tool of its own under it.

    Parameters:
        schema: The tool, carrying its handler.
        cancel_on_interruption: Whether a call of the tool is cancelled when
            the user interrupts; ``False`` makes it an asynchronous tool.
        announce_call_start: Whether the service announces a call of the tool
            when it starts, with its ``on_function_calls_started`` event and
            a ``FunctionCallsStartedFrame``, as it does for an application's
            own tools. ``False``, the default, suits a tool that is the
            extension's own mechanism, which the application and the rest of
            the pipeline need not hear about. Either way, the call's
            ``FunctionCallInProgressFrame`` and ``FunctionCallResultFrame``
            are pushed, and the call and its result go into the context.
    """

    schema: FunctionSchema
    cancel_on_interruption: bool = True
    announce_call_start: bool = False


class BaseLLMExtension(BaseObject):
    """Base class for LLM extensions.

    An LLM extension is a pluggable feature or behavior of an
    :class:`~pipecat.services.llm_service.LLMService`. It can add to the
    service's system instruction, watch the frames the service processes,
    filter what the service pushes, and push frames of its own. It can also
    have the service act: run function calls, process a frame, or cancel its
    inference in progress.

    Extensions wrap the service the way middleware wraps a handler: as the
    layers of an onion, with the service at the center. :attr:`order` places
    an extension among the layers: one with a lower order sits "closer" to
    the service, one with a higher order "farther" from it. An incoming
    frame, one the service processes, goes through the layers from the
    outside in. An outgoing frame, one pushed from the service, goes through
    them from the inside out. With extensions A, B and C, A "closest" to the
    service::

        incoming:  C.process_frame, B.process_frame, A.process_frame, the service
        outgoing:  the service, A.filter_push_frame, B.filter_push_frame, C.filter_push_frame

    To an extension, everything inside its layer is "the service": the
    service itself and the extensions "closer" to it. So an extension's
    author needs to know nothing about what else is attached, and has two
    guarantees:

    - :meth:`filter_push_frame` sees every frame pushed from inside the
      layer, whether the service pushed it or a "closer" extension did.
    - :meth:`process_frame` sees an incoming frame before
      :meth:`filter_push_frame` sees anything produced in reaction to it.

    The second guarantee has one limit, which comes from
    :class:`~pipecat.services.ai_service.AIService`. Frames pushed from its
    lifecycle methods, ``start()``, ``stop()`` and ``cancel()``, reach the
    extensions' :meth:`filter_push_frame` before the triggering frames,
    ``StartFrame``, ``EndFrame`` and ``CancelFrame``, reach their
    :meth:`process_frame`. The service itself sees the same order: a frame
    its ``start()`` pushes goes through its ``push_frame`` before its own
    ``process_frame`` logic handles the ``StartFrame``.

    An extension is attached to a service in its settings
    (``LLMService(settings=...Settings(extensions=[...]))``), later by a
    component that has the service in hand
    (:meth:`~pipecat.services.llm_service.LLMService.add_extension`), or by
    an :class:`~pipecat.frames.frames.LLMUpdateSettingsFrame` whose delta
    lists it in ``extensions``. An extension belongs to one service. From
    then on the service:

    - composes :attr:`system_instruction` onto its system instruction, after
      the application's prompt;
    - adds the tools :meth:`builtin_tools` returns to its built-in tools,
      synced with the registered tools before each inference;
    - hands it every incoming frame, in :meth:`process_frame`, before the
      extensions "closer" to the service, and the service, handle the frame;
    - passes every frame pushed from inside its layer through
      :meth:`filter_push_frame`, which can hold the frame back, drop it, or
      add frames;
    - passes the function calls it is about to run through
      :meth:`filter_function_calls`, which can drop some;
    - sends the frames it pushes with :meth:`push_frame` and
      :meth:`broadcast_frame` through the extensions "farther" from the
      service than it, then out of the service;
    - acts when it asks: runs function calls (:meth:`run_function_calls`),
      processes a frame (:meth:`queue_frame`), or cancels its inference in
      progress (:meth:`cancel_inference_in_progress`).

    A service holds one extension of each class, which
    :meth:`~pipecat.services.llm_service.LLMService.get_extension` finds. If a
    developer needs a parameterized extension twice, they should make a
    subclass for it.

    Events triggered by LLM extensions:

      - `on_push_frame`: Indicates the extension wants to push a frame.
      - `on_broadcast_frame`: Indicates the extension wants to broadcast a frame.
      - `on_run_function_calls`: Indicates the extension wants function calls run.
      - `on_queue_frame`: Indicates the extension wants the service to process a frame.
      - `on_cancel_inference_in_progress`: Indicates the extension wants the
        service's inference for a context frame cancelled.
    """

    order: int = 0
    """Where the extension sits among the layers around a service: an
    extension with a lower order is "closer" to the service. The "closest"
    one filters an outgoing frame first and sees an incoming frame last.
    Instructions compose lowest first. Among extensions of equal order, one
    attached earlier sits "closer"."""

    ORDER_INNERMOST: ClassVar[int] = -sys.maxsize
    """The order of the innermost layer: an extension with it sits "closest"
    to the service, and filters what the service pushes before any other
    extension does. If several extensions have it, the one attached first
    sits "closest", as among any extensions of equal order."""

    def __init__(self, **kwargs):
        """Initialize the extension.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_run_function_calls", sync=True)
        self._register_event_handler("on_queue_frame", sync=True)
        self._register_event_handler("on_cancel_inference_in_progress", sync=True)

    @property
    def attached_service(self) -> LLMService | None:
        """The service the extension is attached to, known once that service is set up; None before."""
        handlers = self._event_handlers["on_push_frame"].handlers
        return getattr(handlers[0], "__self__", None) if handlers else None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the extension, as the service it extends is set up.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)

    @property
    def system_instruction(self) -> str | None:
        """The standing instructions the extension contributes, or ``None``.

        Composed onto the service's system instruction after the
        application's prompt, in extension order, each time the service
        composes it.
        """
        return None

    def builtin_tools(
        self, registered: Mapping[str | None, FunctionCallRegistryItem]
    ) -> list[LLMExtensionTool]:
        """The tools the extension adds, given the tools registered with the service.

        Called whenever the service syncs its registered tools, before an
        inference; the service advertises what is returned and withdraws what
        no longer is. :attr:`system_instruction` is read after it, so an
        instruction can describe the tools.

        Args:
            registered: The service's registered tools by name, the
                extensions' own included; ``None`` keys a catch-all handler.

        Returns:
            The tools to add; none by default.
        """
        return []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Watch an incoming frame, before the service handles it.

        The extensions "farther" from the service than this one have seen the
        frame. The ones "closer" to it handle the frame next, then the service
        does. Anything they produce in reaction reaches
        :meth:`filter_push_frame` after this returns.

        Args:
            frame: The frame.
            direction: The direction of frame processing.
        """
        pass

    async def filter_push_frame(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        """Filter a frame pushed from inside this extension's layer.

        The service pushed the frame, or an extension "closer" to it than this
        one did. Returns the frames to push in its place, in the same
        direction: ``[frame]`` to pass it on, an empty list to hold it back or
        drop it, or more frames to add some. The frames returned go on through
        the extensions "farther" from the service than this one. A frame for
        the other direction is pushed with :meth:`push_frame` or
        :meth:`broadcast_frame`.

        Args:
            frame: The frame being pushed.
            direction: The direction of frame pushing.

        Returns:
            The frames to push on.
        """
        return [frame]

    async def filter_function_calls(
        self, function_calls: Sequence[FunctionCallFromLLM]
    ) -> Sequence[FunctionCallFromLLM]:
        """Filter the function calls the service is about to run.

        Returns the calls to run: all of them to let them through, fewer or
        none to drop some. A dropped call is not run, and nothing of it
        reaches the context or the pipeline. The calls returned go on through
        the extensions "farther" from the service than this one.

        Args:
            function_calls: The function calls about to run.

        Returns:
            The function calls to run.
        """
        return function_calls

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame of the extension's own.

        The frame goes through the :meth:`filter_push_frame` of each extension
        "farther" from the service than this one, then leaves the service.

        Args:
            frame: The frame to push.
            direction: The direction of frame pushing.
        """
        await self._call_event_handler("on_push_frame", frame, direction)

    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcast a frame of the extension's own, upstream and downstream.

        Each of the two frames goes through the :meth:`filter_push_frame` of
        each extension "farther" from the service than this one, then leaves
        the service.

        Args:
            frame_cls: The class of the frame to broadcast.
            **kwargs: Keyword arguments passed to the frame's constructor.
        """
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)

    async def run_function_calls(self, function_calls: Sequence[FunctionCallFromLLM]):
        """Run function calls of the extension's own, as the service runs its LLM's.

        The calls go through the :meth:`filter_function_calls` of each
        extension "farther" from the service than this one, then run.

        Args:
            function_calls: The function calls to run.
        """
        await self._call_event_handler("on_run_function_calls", function_calls)

    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Queue a frame for the service to process, as if it came from the pipeline.

        The frame goes through every extension's :meth:`process_frame`, this
        one's included, then the service handles it.

        Args:
            frame: The frame to queue.
            direction: The direction of frame processing.
        """
        await self._call_event_handler("on_queue_frame", frame, direction)

    async def cancel_inference_in_progress(self, context_frame: LLMContextFrame):
        """Cancel the service's inference for a context frame, if it is the one in progress.

        The response ends there: what the inference already pushed stays, and
        the frames queued behind the context frame are processed next. If the
        service is processing any other frame, nothing happens.

        Call it while processing a system frame, which the service handles
        apart from its inference. A call from the inference itself (from
        :meth:`filter_push_frame` or :meth:`filter_function_calls` on what the
        inference outputs) is refused.

        Args:
            context_frame: The context frame whose inference to cancel.
        """
        await self._call_event_handler("on_cancel_inference_in_progress", context_frame)
