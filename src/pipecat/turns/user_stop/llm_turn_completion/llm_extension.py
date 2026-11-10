#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extension that reads the turn-completion marker opening each response."""

from loguru import logger

from pipecat.expressive.instructions import perception_turn_completion_instructions
from pipecat.expressive.perception import PerceptionCapabilities
from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMMarkerResponseFrame,
    LLMTextFrame,
    PerceptionMetadataFrame,
    UserTurnCompletionFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.base_llm_extension import BaseLLMExtension
from pipecat.turns.user_turn_completion import (
    IncompleteType,
    TurnMarker,
    UserTurnCompletionConfig,
)


class LLMTurnCompletionLLMExtension(BaseLLMExtension):
    """Reads the turn-completion marker that opens each of the model's responses.

    Each response starts with a marker: ● (the turn is complete), ◐ (the user
    will continue shortly) or ○ (the user needs more time). The extension
    strips the marker from the spoken text and reports it:

    - upstream, once per inference, as a ``UserTurnCompletionFrame`` for the
      ``LLMTurnCompletionUserTurnStopStrategy`` to decide on. A tool call
      before any marker reports a complete turn, and a response without a
      marker reports no kind (``None``);
    - in both directions as a ``UserTurnInferenceCompletedFrame`` when the
      turn is complete;
    - downstream as an ``LLMMarkerFrame``, which the assistant aggregator
      writes to the context.

    Text after ◐ or ○ is not spoken: the marker is the whole response. When
    the response ends, its raw text and marker are reported downstream as an
    ``LLMMarkerResponseFrame``.

    The extension contributes the turn-completion instructions to the system
    instruction, with a note that a reading of how the user spoke doesn't
    decide the turn when the LLM is taught one (expressive conversation).

    Installed by the strategy, which acts on its reports and sends the
    re-prompts: ``user_turn_strategies=FilterIncompleteUserTurnStrategies()``
    on the user aggregator is how an application turns turn completion on.
    """

    installed_by = "LLMTurnCompletionUserTurnStopStrategy"
    # Before any other reader of the model's output: the marker opens the
    # response, and what follows it reads differently once it is stripped.
    order = -1

    def __init__(self, config: UserTurnCompletionConfig | None = None, **kwargs):
        """Initialize the extension.

        Args:
            config: The turn completion configuration: the markers, the
                instructions, the re-prompts and their timeouts. Defaults when
                omitted.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._config = config or UserTurnCompletionConfig()
        self._expressive = False
        # With expressive conversation on, the signals perceived of how the
        # user spoke, from the latest PerceptionMetadataFrame.
        self._perception: PerceptionCapabilities | None = None
        # The LLMContextFrame that started the current inference, and whether
        # its turn-completion judgment has been reported upstream yet.
        self._context_frame_id: int | None = None
        self._turn_completion_reported = False
        self._turn_text_buffer = ""
        # The current response as the LLM produced it, and the marker read
        # from it, reported together when the response ends.
        self._response_raw = ""
        self._response_marker: tuple[str, str] | None = None
        # Completion verdict for the current LLM response, set when a marker is
        # detected in the text stream. ``None`` means no marker yet, so keep
        # buffering text until one appears. ``INCOMPLETE`` also doubles as a
        # safety latch: the prompt tells the LLM to emit only the marker, but if
        # it disobeys and streams more text we keep suppressing it.
        self._turn_marker: TurnMarker | None = None

    @property
    def config(self) -> UserTurnCompletionConfig:
        """The turn completion configuration."""
        return self._config

    @config.setter
    def config(self, config: UserTurnCompletionConfig):
        self._config = config

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the extension.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._expressive = setup.enable_expressive_conversation

    @property
    def system_instruction(self) -> str:
        """The turn-completion instructions, with the perception note when the LLM is taught a reading."""
        parts = [self._config.completion_instructions]
        if self._perception and self._perception.signals:
            parts.append(perception_turn_completion_instructions(*self._config.markers))
        return "\n\n".join(parts)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Follow the inference the service runs, and reset on an interruption.

        Args:
            frame: The frame the service processes.
            direction: The direction of frame processing.
        """
        if isinstance(frame, InterruptionFrame):
            for text in self._reset_turn():
                await self.push_frame(text)
        elif isinstance(frame, LLMContextFrame):
            self._context_frame_id = frame.id
            self._turn_completion_reported = False
        elif isinstance(frame, PerceptionMetadataFrame) and self._expressive:
            self._perception = frame.perception

    async def filter_push(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        """Read the marker from the response's text, and report it.

        Args:
            frame: The frame the service is pushing.
            direction: The direction of frame pushing.

        Returns:
            The frames to push on: the text held back until the marker is
            read, the marker frame, and the response's report at its end.
        """
        if isinstance(frame, FunctionCallsStartedFrame):
            # A tool call is the LLM judging the turn complete. Reported before
            # the function dispatches, so the turn can end before its result
            # reaches the assistant aggregator.
            await self._report_turn_completion(TurnMarker.COMPLETE.value, None)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._response_raw = ""
            self._response_marker = None
        elif isinstance(frame, LLMFullResponseEndFrame):
            # A response that ends without a marker or a tool call carries no judgment.
            await self._report_turn_completion(None, None)
            frames: list[Frame] = []
            if report := self._response_report():
                frames.append(report)
            frames.extend(self._reset_turn())
            frames.append(frame)
            return frames
        elif isinstance(frame, LLMTextFrame):
            return await self._filter_text(frame)
        return [frame]

    async def _filter_text(self, frame: LLMTextFrame) -> list[Frame]:
        """Hold the response's text until its marker is read, then let it through.

        - Until a marker appears, the text is buffered.
        - ◐ or ○ is the whole response: the marker is reported and pushed, and
          the rest of the response is dropped.
        - After ●, the text after it is pushed, and all later text flows
          through as it comes.
        """
        self._response_raw += frame.text

        # The prompt tells the LLM to emit only the marker; if it disobeys and
        # streams more text (e.g. "◐ Please continue..."), keep suppressing it.
        if self._turn_marker == TurnMarker.INCOMPLETE:
            return []

        if self._turn_marker == TurnMarker.COMPLETE:
            return [frame]

        config = self._config
        self._turn_text_buffer += frame.text

        # ◐ (short) or ○ (long): the user was cut off or needs time, and the
        # marker is the whole response.
        incomplete_type: IncompleteType | None = None
        if config.incomplete_short_marker in self._turn_text_buffer:
            incomplete_type = IncompleteType.SHORT
        elif config.incomplete_long_marker in self._turn_text_buffer:
            incomplete_type = IncompleteType.LONG

        if incomplete_type:
            marker = (
                config.incomplete_short_marker
                if incomplete_type == IncompleteType.SHORT
                else config.incomplete_long_marker
            )
            logger.debug(
                f"INCOMPLETE {incomplete_type.value.upper()} ({marker}) detected, suppressing text"
            )
            self._turn_marker = TurnMarker.INCOMPLETE
            self._turn_text_buffer = ""
            # The bot produces no spoken output for an incomplete turn, so the
            # marker is the whole context entry: a stand-alone assistant message.
            return [await self._mark(marker, incomplete_type.value)]

        if config.complete_marker in self._turn_text_buffer:
            logger.debug(f"COMPLETE ({config.complete_marker}) detected, pushing buffered text")
            # The assistant aggregator prepends the marker to the aggregated
            # text that follows, so the context message is "● <response>".
            frames: list[Frame] = [
                await self._mark(config.complete_marker, TurnMarker.COMPLETE.value)
            ]

            # The marker and text can arrive in one chunk ("● Hello!"), so
            # split the buffer at the marker.
            marker_end = self._turn_text_buffer.index(config.complete_marker) + len(
                config.complete_marker
            )
            remaining_text = self._turn_text_buffer[marker_end:]
            if remaining_text.startswith(" "):
                remaining_text = remaining_text[1:]
            if remaining_text:
                frames.append(LLMTextFrame(remaining_text))

            self._turn_text_buffer = ""
            self._turn_marker = TurnMarker.COMPLETE
            return frames

        return []

    async def _mark(self, marker: str, kind: str) -> LLMMarkerFrame:
        """Report the marker read from the response, and return its frame for the context."""
        self._response_marker = (marker, kind)
        await self._report_turn_completion(kind, marker)
        # Only a complete turn's marker belongs to the response that follows.
        return LLMMarkerFrame(marker, append_to_context_immediately=kind != "complete")

    async def _report_turn_completion(self, kind: str | None, marker: str | None):
        """Push the turn-completion judgment upstream, once per inference.

        A complete turn is also broadcast as a ``UserTurnInferenceCompletedFrame``.
        """
        if self._turn_completion_reported:
            return
        self._turn_completion_reported = True
        await self.push_frame(
            UserTurnCompletionFrame(
                kind=kind, marker=marker, context_frame_id=self._context_frame_id
            ),
            FrameDirection.UPSTREAM,
        )
        if kind == TurnMarker.COMPLETE.value:
            await self.broadcast_frame(UserTurnInferenceCompletedFrame)

    def _response_report(self) -> LLMMarkerResponseFrame | None:
        """The response's raw text and marker, once there was a response to read."""
        if not self._response_raw and self._response_marker is None:
            return None
        marker, kind = self._response_marker or (None, None)
        report = LLMMarkerResponseFrame(
            raw=self._response_raw,
            marker=marker,
            kind=kind,
            markers=list(self._config.markers),
        )
        self._response_raw = ""
        self._response_marker = None
        return report

    def _reset_turn(self) -> list[Frame]:
        """Reset the turn's state between responses.

        Returns:
            The text buffered without a marker ever appearing, as a frame to
            push so it is not lost; empty otherwise.
        """
        frames: list[Frame] = []
        if self._turn_marker is None and self._turn_text_buffer:
            logger.warning(
                f"{self}: the LLM response did not contain a turn completion marker "
                f"({'/'.join(self._config.markers)}). Pushing its text anyway. The system "
                "prompt may be missing the turn completion instructions."
            )
            frames.append(LLMTextFrame(self._turn_text_buffer))
        self._turn_text_buffer = ""
        self._turn_marker = None
        return frames
