#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI observer for converting pipeline frames to outgoing RTVI messages."""

import inspect
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Optional,
)

from loguru import logger
from pydantic import BaseModel

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.utils import calculate_audio_volume
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import (
    RTVIConfigureObserverFrame,
    RTVIServerMessageFrame,
    RTVIServerResponseFrame,
    RTVIUICommandFrame,
    RTVIUIJobGroupFrame,
)
from pipecat.processors.frameworks.rtvi.models import BotOutputTransformResult
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.utils.string import match_endofsentence

if TYPE_CHECKING:
    from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class RTVIFunctionCallReportLevel(StrEnum):
    """Level of detail to include in function call RTVI events.

    Controls what information is exposed in function call events for security.

    Values:
        DISABLED: No events emitted for this function call.
        NONE: Events only with tool_call_id, no function name or metadata (most secure).
        NAME: Events with function name, no arguments or results.
        FULL: Events with function name, arguments, and results.
    """

    DISABLED = "disabled"
    NONE = "none"
    NAME = "name"
    FULL = "full"


@dataclass
class RTVIObserverParams:
    """Parameters for configuring RTVI Observer behavior.

    Parameters:
        bot_output_enabled: Indicates if bot output messages should be sent.
        bot_llm_enabled: Indicates if the bot's LLM messages should be sent.
        bot_tts_enabled: Indicates if the bot's TTS messages should be sent.
        bot_speaking_enabled: Indicates if the bot's started/stopped speaking messages should be sent.
        bot_audio_level_enabled: Indicates if bot's audio level messages should be sent.
        user_llm_enabled: Indicates if the user's LLM input messages should be sent.
        user_speaking_enabled: Indicates if the user's started/stopped speaking messages should be sent.
        vad_user_speaking_enabled: Indicates if raw VAD user started/stopped speaking messages
            should be sent. These reflect the VAD signal directly, independent of turn
            finalization (unlike ``user_speaking_enabled``, which a turn strategy may gate or
            defer). Off by default. Defaults to False.
        user_transcription_enabled: Indicates if user's transcription messages should be sent.
        user_audio_level_enabled: Indicates if user's audio level messages should be sent.
        metrics_enabled: Indicates if metrics messages should be sent.
        system_logs_enabled: Indicates if system logs should be sent.
        ignored_sources: List of frame processors whose frames should be silently ignored
            by this observer. Useful for suppressing RTVI messages from secondary pipeline
            branches (e.g. a silent evaluation LLM) that should not be visible to clients.
            Sources can also be added and removed dynamically via ``add_ignored_source()``
            and ``remove_ignored_source()``.
        skip_aggregator_types: List of aggregation types to skip sending as tts/output messages.
            Note: if using this to avoid sending secure information, be sure to also disable
            bot_llm_enabled to avoid leaking through LLM messages.
        bot_output_transforms: A list of callables to transform text before sending it to the
            client. Each callable should use the 4-parameter signature::

                async def my_transform(
                    text: str,
                    agg_type: AggregationType | str,
                    accumulated_text: str | None = None,
                    remaining_text: str | None = None,
                ) -> BotOutputTransformResult: ...

            When ``accumulated_text`` and ``remaining_text`` are ``None`` the transform is being
            called for the full segment text (from ``bot-output``). When they are provided the
            transform is being called for a progress event and must return a
            ``BotOutputTransformResult`` with ``accumulated_text`` and ``remaining_text`` set.

            The 2-parameter signature ``(text, agg_type) -> str`` is deprecated. Transforms using
            it will still work but will emit a ``DeprecationWarning`` at registration time.

            To register, provide a list of tuples of (aggregation_type | '*', transform_function).
        audio_level_period_secs: How often audio levels should be sent if enabled.
        function_call_report_level: Controls what information is exposed in function call
            events for security. A dict mapping function names to levels, where ``"*"``
            sets the default level for unlisted functions::

                function_call_report_level={
                    "*": RTVIFunctionCallReportLevel.NONE,  # Default: events with no metadata
                    "get_weather": RTVIFunctionCallReportLevel.FULL,  # Expose everything
                }

            Levels:
                - DISABLED: No events emitted for this function.
                - NONE: Events with tool_call_id only (most secure when events needed).
                - NAME: Adds function name to events.
                - FULL: Adds function name, arguments, and results.

            Defaults to ``{"*": RTVIFunctionCallReportLevel.NONE}``.
    """

    bot_output_enabled: bool = True
    bot_llm_enabled: bool = True
    bot_tts_enabled: bool = True
    bot_speaking_enabled: bool = True
    bot_audio_level_enabled: bool = False
    user_llm_enabled: bool = True
    user_speaking_enabled: bool = True
    vad_user_speaking_enabled: bool = False
    user_mute_enabled: bool = True
    user_transcription_enabled: bool = True
    user_audio_level_enabled: bool = False
    metrics_enabled: bool = True
    system_logs_enabled: bool = False
    ignored_sources: list[FrameProcessor] = field(default_factory=list)
    skip_aggregator_types: list[AggregationType | str] | None = None
    bot_output_transforms: (
        list[
            tuple[
                AggregationType | str,
                Callable[..., Awaitable[BotOutputTransformResult | str]],
            ]
        ]
        | None
    ) = None
    audio_level_period_secs: float = 0.15
    function_call_report_level: dict[str, RTVIFunctionCallReportLevel] = field(
        default_factory=lambda: {"*": RTVIFunctionCallReportLevel.NONE}
    )


class RTVIObserver(BaseObserver):
    """Pipeline frame observer for RTVI server message handling.

    This observer monitors pipeline frames and converts them into appropriate RTVI messages
    for client communication. It handles various frame types including speech events,
    transcriptions, LLM responses, and TTS events.

    Note:
        This observer only handles outgoing messages. Incoming RTVI client messages
        are handled by the RTVIProcessor.
    """

    def __init__(
        self,
        rtvi: Optional["RTVIProcessor"] = None,
        *,
        params: RTVIObserverParams | None = None,
        **kwargs,
    ):
        """Initialize the RTVI observer.

        Args:
            rtvi: The RTVI processor to push frames to.
            params: Settings to enable/disable specific messages.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._rtvi = rtvi
        self._params = params or RTVIObserverParams()

        self._ignored_sources: set[FrameProcessor] = set(self._params.ignored_sources)
        self._frames_seen = set()

        self._bot_transcription = ""
        self._last_user_audio_level = 0
        self._last_bot_audio_level = 0

        # Track bot speaking state for queuing aggregated text frames
        self._bot_is_speaking = False
        self._queued_aggregated_text_frames: list[AggregatedTextFrame] = []

        if self._params.system_logs_enabled:
            self._system_logger_id = logger.add(self._logger_sink)

        self._aggregation_transforms: list[
            tuple[
                AggregationType | str,
                Callable[..., Awaitable[BotOutputTransformResult | str]],
                bool,
            ]
        ] = []
        for agg_type, fn in self._params.bot_output_transforms or []:
            self.add_bot_output_transformer(fn, agg_type)

    @staticmethod
    def _check_progress_aware(fn: Callable) -> bool:
        """Return True if ``fn`` declares 4 parameters (the progress-aware signature)."""
        try:
            return len(inspect.signature(fn).parameters) >= 4
        except (ValueError, TypeError):
            return False

    @property
    def _is_legacy_client(self) -> bool:
        """Return True when the connected client uses a deprecated 1.x protocol."""
        if not self._rtvi:
            return False
        return self._rtvi.client_version[0] == RTVI.LEGACY_SUPPORTED_MAJOR

    def add_bot_output_transformer(
        self,
        transform_function: Callable[..., Awaitable[BotOutputTransformResult | str]],
        aggregation_type: AggregationType | str = "*",
    ):
        """Register a text transformer for a specific aggregation type.

        The preferred transform signature is::

            async def my_transform(
                text: str,
                agg_type: AggregationType | str,
                accumulated_text: str | None = None,
                remaining_text: str | None = None,
            ) -> BotOutputTransformResult: ...

        When ``accumulated_text`` and ``remaining_text`` are ``None`` the transform is being
        called for the full segment text (``bot-output`` message). When they are provided it is
        being called for a progress event; return a ``BotOutputTransformResult`` with
        ``accumulated_text`` and ``remaining_text`` populated so the client can display
        word-level progress on the transformed text.

        The legacy 2-parameter signature ``(text, agg_type) -> str`` is still accepted but
        deprecated. Progress events will fall back to applying the transform independently to
        each partial string, which may produce inconsistent results for context-dependent
        transforms.

        Args:
            transform_function: The transform callable.
            aggregation_type: Aggregation type to match, or ``"*"`` for all types.
        """
        is_progress_aware = self._check_progress_aware(transform_function)
        if not is_progress_aware:
            warnings.warn(
                f"Bot output transform '{transform_function.__name__}' uses the deprecated "
                "2-parameter signature '(text, agg_type) -> str'. Update to the 4-parameter "
                "signature '(text, agg_type, accumulated_text, remaining_text) -> "
                "BotOutputTransformResult' to support word-level progress transforms.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._aggregation_transforms.append(
            (aggregation_type, transform_function, is_progress_aware)
        )

    def remove_bot_output_transformer(
        self,
        transform_function: Callable[..., Awaitable[BotOutputTransformResult | str]],
        aggregation_type: AggregationType | str = "*",
    ):
        """Remove a text transformer for a specific aggregation type.

        Args:
            transform_function: The function to remove.
            aggregation_type: The type of aggregation to remove the transformer for.
        """
        self._aggregation_transforms = [
            (agg_type, func, aware)
            for agg_type, func, aware in self._aggregation_transforms
            if not (agg_type == aggregation_type and func == transform_function)
        ]

    def add_ignored_source(self, source: FrameProcessor):
        """Ignore all frames pushed by the given processor.

        Any frame whose source matches ``source`` will be silently skipped,
        preventing RTVI messages from being emitted for activity in that
        processor. Useful for suppressing events from secondary pipeline
        branches (e.g. a silent evaluation LLM) that should not be visible
        to clients.

        Args:
            source: The frame processor to ignore.
        """
        self._ignored_sources.add(source)

    def remove_ignored_source(self, source: FrameProcessor):
        """Stop ignoring frames pushed by the given processor.

        Reverses a previous call to ``add_ignored_source()``. If ``source``
        was not previously ignored this is a no-op.

        Args:
            source: The frame processor to stop ignoring.
        """
        self._ignored_sources.discard(source)

    def _get_function_call_report_level(self, function_name: str) -> RTVIFunctionCallReportLevel:
        """Get the report level for a specific function call.

        Args:
            function_name: The name of the function to get the report level for.

        Returns:
            The report level for the function. Looks up the function name first,
            then falls back to "*" key, then NONE.
        """
        levels = self._params.function_call_report_level
        if function_name in levels:
            return levels[function_name]
        return levels.get("*", RTVIFunctionCallReportLevel.NONE)

    def _apply_config(self, frame: RTVIConfigureObserverFrame):
        """Apply a dynamic observer-config update (only the fields that are set).

        Args:
            frame: The config frame; ``None`` fields leave the current value
                unchanged.
        """
        if frame.function_call_report_level is not None:
            self._params.function_call_report_level = frame.function_call_report_level
            logger.debug(
                f"{self}: function_call_report_level set to {frame.function_call_report_level}"
            )
        if frame.vad_user_speaking_enabled is not None:
            self._params.vad_user_speaking_enabled = frame.vad_user_speaking_enabled
            logger.debug(
                f"{self}: vad_user_speaking_enabled set to {frame.vad_user_speaking_enabled}"
            )

    async def _logger_sink(self, message):
        """Logger sink so we can send system logs to RTVI clients."""
        message = RTVI.SystemLogMessage(data=RTVI.TextMessageData(text=message))
        await self.send_rtvi_message(message)

    async def cleanup(self):
        """Cleanup RTVI observer resources."""
        await super().cleanup()
        if self._params.system_logs_enabled:
            logger.remove(self._system_logger_id)

    async def send_rtvi_message(self, model: BaseModel, exclude_none: bool = True):
        """Send an RTVI message.

        By default, we push a transport frame. But this function can be
        overridden by subclass to send RTVI messages in different ways.

        Args:
            model: The message to send.
            exclude_none: Whether to exclude None values from the model dump.

        """
        if self._rtvi:
            await self._rtvi.push_transport_message(model, exclude_none)

    async def on_push_frame(self, data: FramePushed):
        """Process a frame being pushed through the pipeline.

        Args:
            data: Frame push event data containing source, frame, direction, and timestamp.
        """
        src = data.source
        frame = data.frame
        direction = data.direction

        # Frames from explicitly ignored sources are always skipped.
        if self._ignored_sources and src in self._ignored_sources:
            return

        # For broadcast frames (pushed in both directions), only process
        # the downstream copy to avoid sending duplicate RTVI messages.
        if frame.broadcast_sibling_id is not None and direction != FrameDirection.DOWNSTREAM:
            return

        # If we have already seen this frame, let's skip it.
        if frame.id in self._frames_seen:
            return

        # This tells whether the frame is already processed. If false, we will try
        # again the next time we see the frame.
        mark_as_seen = True

        if (
            isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame))
            and self._params.user_speaking_enabled
        ):
            await self._handle_interruptions(frame)
        elif (
            isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame))
            and self._params.vad_user_speaking_enabled
        ):
            await self._handle_vad_speaking(frame)
        elif (
            isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame))
            and self._params.user_mute_enabled
        ):
            await self._handle_user_mute(frame)
        elif (
            isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame))
            and self._params.bot_speaking_enabled
        ):
            await self._handle_bot_speaking(frame)
        elif isinstance(frame, InterruptionFrame) and self._params.bot_speaking_enabled:
            # The bot's in-flight output was cut off (VAD barge-in or a programmatic
            # run_immediately interrupt). Let clients drop what it was mid-saying.
            await self.send_rtvi_message(RTVI.BotInterruptedMessage())
        elif (
            isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and self._params.user_transcription_enabled
        ):
            await self._handle_user_transcriptions(frame)
        elif isinstance(frame, LLMContextFrame) and self._params.user_llm_enabled:
            await self._handle_context(frame)
        elif isinstance(frame, LLMFullResponseStartFrame) and self._params.bot_llm_enabled:
            await self.send_rtvi_message(RTVI.BotLLMStartedMessage())
        elif isinstance(frame, LLMFullResponseEndFrame) and self._params.bot_llm_enabled:
            await self.send_rtvi_message(RTVI.BotLLMStoppedMessage())
        elif isinstance(frame, LLMTextFrame) and self._params.bot_llm_enabled:
            await self._handle_llm_text_frame(frame)
        elif isinstance(frame, TTSStartedFrame) and self._params.bot_tts_enabled:
            await self.send_rtvi_message(RTVI.BotTTSStartedMessage())
        elif isinstance(frame, TTSStoppedFrame) and self._params.bot_tts_enabled:
            await self.send_rtvi_message(RTVI.BotTTSStoppedMessage())
        elif isinstance(frame, AggregatedTextProgressFrame):
            if not isinstance(src, BaseOutputTransport):
                # This check is to make sure we handle the frame when it has gone
                # through the transport and has correct timing.
                mark_as_seen = False
            else:
                await self._handle_aggregated_progress(frame)
        elif isinstance(frame, AggregatedTextFrame) and (
            self._params.bot_output_enabled or self._params.bot_tts_enabled
        ):
            if not isinstance(src, BaseOutputTransport):
                # This check is to make sure we handle the frame when it has gone
                # through the transport and has correct timing.
                mark_as_seen = False
            else:
                await self._handle_aggregated_llm_text(frame)
        elif isinstance(frame, MetricsFrame) and self._params.metrics_enabled:
            await self._handle_metrics(frame)
        elif isinstance(frame, FunctionCallsStartedFrame):
            for function_call in frame.function_calls:
                report_level = self._get_function_call_report_level(function_call.function_name)
                if report_level == RTVIFunctionCallReportLevel.DISABLED:
                    continue
                data = RTVI.LLMFunctionCallStartMessageData()
                if report_level in (
                    RTVIFunctionCallReportLevel.NAME,
                    RTVIFunctionCallReportLevel.FULL,
                ):
                    data.function_name = function_call.function_name
                message = RTVI.LLMFunctionCallStartMessage(data=data)
                await self.send_rtvi_message(message)
        elif isinstance(frame, FunctionCallInProgressFrame):
            report_level = self._get_function_call_report_level(frame.function_name)
            if report_level != RTVIFunctionCallReportLevel.DISABLED:
                data = RTVI.LLMFunctionCallInProgressMessageData(tool_call_id=frame.tool_call_id)
                if report_level in (
                    RTVIFunctionCallReportLevel.NAME,
                    RTVIFunctionCallReportLevel.FULL,
                ):
                    data.function_name = frame.function_name
                if report_level == RTVIFunctionCallReportLevel.FULL:
                    data.arguments = frame.arguments
                message = RTVI.LLMFunctionCallInProgressMessage(data=data)
                await self.send_rtvi_message(message)
        elif isinstance(frame, FunctionCallCancelFrame):
            report_level = self._get_function_call_report_level(frame.function_name)
            if report_level != RTVIFunctionCallReportLevel.DISABLED:
                data = RTVI.LLMFunctionCallStoppedMessageData(
                    tool_call_id=frame.tool_call_id,
                    cancelled=True,
                )
                if report_level in (
                    RTVIFunctionCallReportLevel.NAME,
                    RTVIFunctionCallReportLevel.FULL,
                ):
                    data.function_name = frame.function_name
                message = RTVI.LLMFunctionCallStoppedMessage(data=data)
                await self.send_rtvi_message(message)
        elif isinstance(frame, FunctionCallResultFrame):
            report_level = self._get_function_call_report_level(frame.function_name)
            if report_level != RTVIFunctionCallReportLevel.DISABLED:
                data = RTVI.LLMFunctionCallStoppedMessageData(
                    tool_call_id=frame.tool_call_id,
                    cancelled=False,
                )
                if report_level in (
                    RTVIFunctionCallReportLevel.NAME,
                    RTVIFunctionCallReportLevel.FULL,
                ):
                    data.function_name = frame.function_name
                if report_level == RTVIFunctionCallReportLevel.FULL:
                    data.result = frame.result if frame.result else None
                message = RTVI.LLMFunctionCallStoppedMessage(data=data)
                await self.send_rtvi_message(message)
        elif isinstance(frame, RTVIServerMessageFrame):
            message = RTVI.ServerMessage(data=frame.data)
            await self.send_rtvi_message(message)
        elif isinstance(frame, RTVIUICommandFrame):
            message = RTVI.UICommandMessage(
                data=RTVI.UICommandData(command=frame.command, payload=frame.payload)
            )
            await self.send_rtvi_message(message)
        elif isinstance(frame, RTVIUIJobGroupFrame):
            if frame.data is not None:
                message = RTVI.UIJobGroupMessage(data=frame.data)
                await self.send_rtvi_message(message)
        elif isinstance(frame, RTVIConfigureObserverFrame):
            self._apply_config(frame)
        elif isinstance(frame, RTVIServerResponseFrame):
            if frame.error is not None:
                await self._send_error_response(frame)
            else:
                await self._send_server_response(frame)
        elif isinstance(frame, InputAudioRawFrame) and self._params.user_audio_level_enabled:
            curr_time = time.time()
            diff_time = curr_time - self._last_user_audio_level
            if diff_time > self._params.audio_level_period_secs:
                level = calculate_audio_volume(frame.audio, frame.sample_rate)
                message = RTVI.UserAudioLevelMessage(data=RTVI.AudioLevelMessageData(value=level))
                await self.send_rtvi_message(message)
                self._last_user_audio_level = curr_time
        elif isinstance(frame, TTSAudioRawFrame) and self._params.bot_audio_level_enabled:
            curr_time = time.time()
            diff_time = curr_time - self._last_bot_audio_level
            if diff_time > self._params.audio_level_period_secs:
                level = calculate_audio_volume(frame.audio, frame.sample_rate)
                message = RTVI.BotAudioLevelMessage(data=RTVI.AudioLevelMessageData(value=level))
                await self.send_rtvi_message(message)
                self._last_bot_audio_level = curr_time

        if mark_as_seen:
            self._frames_seen.add(frame.id)

    async def _handle_interruptions(self, frame: Frame):
        """Handle user speaking interruption frames."""
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = RTVI.UserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = RTVI.UserStoppedSpeakingMessage()

        if message:
            await self.send_rtvi_message(message)

    async def _handle_vad_speaking(self, frame: Frame):
        """Emit raw VAD user started/stopped speaking messages."""
        message = None
        if isinstance(frame, VADUserStartedSpeakingFrame):
            message = RTVI.VADUserStartedSpeakingMessage()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            message = RTVI.VADUserStoppedSpeakingMessage()

        if message:
            await self.send_rtvi_message(message)

    async def _handle_user_mute(self, frame: Frame):
        """Handle user mute/unmute frames."""
        message = None
        if isinstance(frame, UserMuteStartedFrame):
            message = RTVI.UserMuteStartedMessage()
        elif isinstance(frame, UserMuteStoppedFrame):
            message = RTVI.UserMuteStoppedMessage()

        if message:
            await self.send_rtvi_message(message)

    async def _handle_bot_speaking(self, frame: Frame):
        """Handle bot speaking event frames."""
        if isinstance(frame, BotStartedSpeakingFrame):
            message = RTVI.BotStartedSpeakingMessage()
            await self.send_rtvi_message(message)
            # Flush any queued aggregated text frames
            for queued_frame in self._queued_aggregated_text_frames:
                await self._send_aggregated_llm_text(queued_frame)
            self._queued_aggregated_text_frames.clear()
            self._bot_is_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            message = RTVI.BotStoppedSpeakingMessage()
            await self.send_rtvi_message(message)
            self._bot_is_speaking = False

    async def _handle_aggregated_llm_text(self, frame: AggregatedTextFrame):
        """Handle aggregated LLM text output frames."""
        if self._bot_is_speaking:
            # Bot has already started speaking, send directly
            await self._send_aggregated_llm_text(frame)
        else:
            # Bot hasn't started speaking yet, queue the frame
            self._queued_aggregated_text_frames.append(frame)

    async def _handle_aggregated_progress(self, frame: AggregatedTextProgressFrame):
        """Handle progress frames."""
        # 1.4.x clients use the old separate bot-output-progress event which was never
        # released, so progress events are simply not sent to legacy clients.
        if self._is_legacy_client:
            return

        logger.trace(
            f"{self} TTS progress: context_id={frame.context_id} "
            f"source_segment_id={frame.segment_id} "
            f"accumulated={repr(frame.accumulated_text)} "
            f"remaining={repr(frame.remaining_text)}"
        )

        accumulated = frame.accumulated_text
        remaining = frame.remaining_text
        text = frame.text
        agg_type = frame.aggregated_by

        for aggregation_type, transform, is_progress_aware in self._aggregation_transforms:
            if aggregation_type == agg_type or aggregation_type == "*":
                if is_progress_aware:
                    result = await transform(text, agg_type, accumulated, remaining)
                    if isinstance(result, BotOutputTransformResult):
                        accumulated = result.accumulated_text or accumulated
                        remaining = result.remaining_text or remaining
                        text = result.text or text
                else:
                    accumulated = await transform(accumulated, agg_type)
                    remaining = await transform(remaining, agg_type)
                    text = await transform(text, agg_type)

        if self._params.bot_output_enabled:
            spoken_status: RTVI.SpokenStatus = "completed" if remaining == "" else "in-progress"
            message = RTVI.BotOutputMessage(
                data=RTVI.BotOutputMessageData(
                    text=text,
                    will_be_spoken=True,
                    aggregated_by=agg_type,
                    segment_id=frame.segment_id,
                    spoken_status=spoken_status,
                    spoken_progress=RTVI.SpokenProgressData(
                        accumulated_text=accumulated,
                        remaining_text=remaining,
                    ),
                )
            )
            await self.send_rtvi_message(message)

    async def _send_aggregated_llm_text(self, frame: AggregatedTextFrame):
        """Send aggregated LLM text messages."""
        # Skip certain aggregator types if configured to do so.
        if (
            self._params.skip_aggregator_types
            and frame.aggregated_by in self._params.skip_aggregator_types
        ):
            return

        agg_type = frame.aggregated_by

        # For 2.0.0+ clients, word and token types are not emitted as bot-output events;
        # word-level progress is covered by the spoken_status/spoken_progress fields.
        # bot-tts-text is a separate channel and is NOT suppressed here.
        suppress_bot_output = not self._is_legacy_client and agg_type in (
            AggregationType.WORD,
            AggregationType.TOKEN,
        )

        text = frame.text
        for aggregation_type, transform, is_progress_aware in self._aggregation_transforms:
            if aggregation_type == agg_type or aggregation_type == "*":
                if is_progress_aware:
                    result = await transform(text, agg_type, None, None)
                else:
                    result = await transform(text, agg_type)
                text = result.text if isinstance(result, BotOutputTransformResult) else result

        isTTS = isinstance(frame, TTSTextFrame)
        will_be_spoken = frame.will_be_spoken
        if self._params.bot_output_enabled and not suppress_bot_output:
            if will_be_spoken:
                if isTTS:
                    # push_text_frames path: TTSTextFrame arrives after synthesis completes
                    spoken_status: RTVI.SpokenStatus = "completed"
                    progress: RTVI.SpokenProgressData | None = RTVI.SpokenProgressData(
                        accumulated_text=text, remaining_text=""
                    )
                else:
                    # word-timestamp path: AggregatedTextFrame arrives before synthesis starts
                    spoken_status = "new"
                    progress = RTVI.SpokenProgressData(accumulated_text="", remaining_text=text)
            else:
                spoken_status = None
                progress = None

            data = RTVI.BotOutputMessageData(
                text=text,
                spoken=isTTS,
                will_be_spoken=will_be_spoken,
                aggregated_by=agg_type,
                segment_id=frame.id,
                spoken_status=spoken_status,
                spoken_progress=progress,
            )
            message = RTVI.BotOutputMessage(data=data)
            await self.send_rtvi_message(message)

        if isTTS and self._params.bot_tts_enabled:
            tts_message = RTVI.BotTTSTextMessage(data=RTVI.TextMessageData(text=text))
            await self.send_rtvi_message(tts_message)

    async def _handle_llm_text_frame(self, frame: LLMTextFrame):
        """Handle LLM text output frames."""
        message = RTVI.BotLLMTextMessage(data=RTVI.TextMessageData(text=frame.text))
        await self.send_rtvi_message(message)

        # TODO (mrkb): Remove all this logic when we fully deprecate bot-transcription messages.
        self._bot_transcription += frame.text

        if match_endofsentence(self._bot_transcription) and len(self._bot_transcription) > 0:
            await self.send_rtvi_message(
                RTVI.BotTranscriptionMessage(
                    data=RTVI.TextMessageData(text=self._bot_transcription)
                )
            )
            self._bot_transcription = ""

    async def _handle_user_transcriptions(self, frame: Frame):
        """Handle user transcription frames."""
        message = None
        if isinstance(frame, TranscriptionFrame):
            message = RTVI.UserTranscriptionMessage(
                data=RTVI.UserTranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=True
                )
            )
        elif isinstance(frame, InterimTranscriptionFrame):
            message = RTVI.UserTranscriptionMessage(
                data=RTVI.UserTranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp, final=False
                )
            )

        if message:
            await self.send_rtvi_message(message)

    async def _handle_context(self, frame: LLMContextFrame):
        """Process LLM context frames to extract user messages for the RTVI client."""
        try:
            messages = frame.context.get_messages()
            if not messages:
                return

            message = messages[-1]

            # Handle Google LLM format (protobuf objects with attributes)
            # Note: not possible if frame is a universal LLMContextFrame
            if hasattr(message, "role") and message.role == "user" and hasattr(message, "parts"):
                text = "".join(part.text for part in message.parts if hasattr(part, "text"))
                if text:
                    rtvi_message = RTVI.UserLLMTextMessage(data=RTVI.TextMessageData(text=text))
                    await self.send_rtvi_message(rtvi_message)

            # Handle OpenAI format (original implementation)
            elif isinstance(message, dict):
                if message.get("role") == "user":
                    content = message["content"]
                    if isinstance(content, list):
                        text = " ".join(item["text"] for item in content if "text" in item)
                    else:
                        text = content
                    rtvi_message = RTVI.UserLLMTextMessage(data=RTVI.TextMessageData(text=text))
                    await self.send_rtvi_message(rtvi_message)

        except Exception as e:
            logger.warning(f"Caught an error while trying to handle context: {e}")

    async def _handle_metrics(self, frame: MetricsFrame):
        """Handle metrics frames and convert to RTVI metrics messages."""
        metrics = {}
        for d in frame.data:
            if isinstance(d, TTFBMetricsData):
                if "ttfb" not in metrics:
                    metrics["ttfb"] = []
                metrics["ttfb"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, TTFAMetricsData):
                if "ttfa" not in metrics:
                    metrics["ttfa"] = []
                metrics["ttfa"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, ProcessingMetricsData):
                if "processing" not in metrics:
                    metrics["processing"] = []
                metrics["processing"].append(d.model_dump(exclude_none=True))
            elif isinstance(d, LLMUsageMetricsData):
                if "tokens" not in metrics:
                    metrics["tokens"] = []
                metrics["tokens"].append(d.value.model_dump(exclude_none=True))
            elif isinstance(d, TTSUsageMetricsData):
                if "characters" not in metrics:
                    metrics["characters"] = []
                metrics["characters"].append(d.model_dump(exclude_none=True))

        message = RTVI.MetricsMessage(data=metrics)
        await self.send_rtvi_message(message)

    async def _send_server_response(self, frame: RTVIServerResponseFrame):
        """Send a response to the client for a specific request."""
        message = RTVI.ServerResponse(
            id=str(frame.client_msg.msg_id),
            data=RTVI.RawServerResponseData(t=frame.client_msg.type, d=frame.data),
        )
        await self.send_rtvi_message(message)

    async def _send_error_response(self, frame: RTVIServerResponseFrame):
        """Send a response to the client for a specific request."""
        message = RTVI.ErrorResponse(
            id=str(frame.client_msg.msg_id), data=RTVI.ErrorResponseData(error=frame.error)
        )
        await self.send_rtvi_message(message)
