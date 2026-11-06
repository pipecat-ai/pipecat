#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for LLM extensions: pluggable features or behaviors of an LLM service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallRegistryItem


@dataclass
class LLMExtensionTool:
    """A tool an extension adds to the service's built-in tools.

    Built-in tools go to the model on every inference, beside the tools the
    context advertises, and their calls stay inside the service: they are
    left out of ``on_function_calls_started`` and the frames that report
    calls to the pipeline.

    Parameters:
        schema: The tool, carrying its handler.
        cancel_on_interruption: Whether a call of the tool is cancelled when
            the user interrupts; ``False`` makes it an asynchronous tool.
    """

    schema: FunctionSchema
    cancel_on_interruption: bool = True


class BaseLLMExtension(BaseObject):
    """Base class for LLM extensions.

    An LLM extension is a pluggable feature or behavior of an
    :class:`~pipecat.services.llm_service.LLMService`. It can add to the
    service's system instruction, watch the frames the service processes,
    filter what the service pushes, and push frames of its own. The service
    runs its extensions in :attr:`order`.

    An extension is attached to a service at construction
    (``LLMService(extensions=[...])``), later by a component that has the
    service in hand
    (:meth:`~pipecat.services.llm_service.LLMService.add_extension`), or by
    an :class:`~pipecat.frames.frames.LLMUpdateSettingsFrame` whose delta
    lists it in ``extensions``. From then on the service:

    - composes :attr:`system_instruction` onto its system instruction, after
      the application's prompt;
    - adds the tools :meth:`builtin_tools` returns to its built-in tools,
      synced with the registered tools before each inference;
    - hands it every frame it processes, in :meth:`process_frame`, before its
      own handling of the frame;
    - passes every frame it pushes through :meth:`filter_push`, which can
      hold the frame back, drop it, or add frames;
    - pushes the frames the extension pushes with :meth:`push_frame` and
      :meth:`broadcast_frame` as they are: its extensions don't filter them.

    Events triggered by LLM extensions:

      - `on_push_frame`: Indicates the extension wants to push a frame.
      - `on_broadcast_frame`: Indicates the extension wants to broadcast a frame.
    """

    installed_by: str | None = None
    """The name of the one component allowed to install the extension, when
    the extension is half of a feature that component owns, such as a user
    turn strategy. ``None`` for an extension an application attaches itself.
    ``LLMService(extensions=[...])`` refuses an extension that names an
    installer, since on its own it would be half the feature."""

    order: int = 0
    """Where the extension runs among a service's extensions, lower first,
    both in filtering what the service pushes and in composing instructions.
    Extensions of equal order run in the order they were attached."""

    def __init__(self, **kwargs):
        """Initialize the extension.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)

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
        """Watch a frame the service processes, before the service handles it.

        Args:
            frame: The frame.
            direction: The direction of frame processing.
        """
        pass

    async def filter_push(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        """Filter a frame the service pushes.

        Returns the frames to push in its place, in the same direction:
        ``[frame]`` to pass it on, an empty list to hold it back or drop it, or
        more frames to add some. The frames returned continue through the
        extensions after this one. A frame for the other direction is pushed
        with :meth:`push_frame` or :meth:`broadcast_frame`.

        Args:
            frame: The frame the service is pushing.
            direction: The direction of frame pushing.

        Returns:
            The frames to push on.
        """
        return [frame]

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame from the service, as it is.

        Args:
            frame: The frame to push.
            direction: The direction of frame pushing.
        """
        await self._call_event_handler("on_push_frame", frame, direction)

    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcast a frame from the service, upstream and downstream.

        Args:
            frame_cls: The class of the frame to broadcast.
            **kwargs: Keyword arguments passed to the frame's constructor.
        """
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)
