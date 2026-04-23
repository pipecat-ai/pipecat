#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld AI Text-to-Speech Service Implementation.

Contains two TTS services:
- InworldTTSService: WebSocket-based TTS service.
- InworldHttpTTSService: HTTP-based TTS service.

Inworld’s text-to-speech (TTS) models offer ultra-realistic, context-aware speech synthesis and precise voice cloning capabilities, enabling developers to build natural and engaging experiences with human-like speech quality at an accessible price point.
"""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
)

import aiohttp
import websockets
from loguru import logger

from pipecat import version as pipecat_version

USER_AGENT = f"pipecat/{pipecat_version()}"
from pydantic import BaseModel

from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Inworld WebSocket TTS, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TextAggregationMode, TTSService, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class InworldTTSSettings(TTSSettings):
    """Settings for InworldTTSService and InworldHttpTTSService.

    Parameters:
        speaking_rate: Speaking rate for speech synthesis.
        temperature: Temperature for speech synthesis.
    """

    speaking_rate: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {
        "voiceId": "voice",
        "modelId": "model",
    }

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        """Construct settings from a plain dict, destructuring legacy nested ``audioConfig``."""
        flat = dict(settings)
        nested = flat.pop("audioConfig", None)
        if isinstance(nested, dict):
            flat.setdefault("speaking_rate", nested.get("speakingRate"))
        return super().from_mapping(flat)


class InworldHttpTTSService(TTSService):
    """Inworld AI HTTP-based TTS service.

    Supports both streaming and non-streaming modes via the `streaming` parameter.
    Outputs LINEAR16 audio at configurable sample rates with word-level timestamps.
    """

    Settings = InworldTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Inworld TTS configuration.

        .. deprecated:: 0.0.105
            Use ``InworldHttpTTSService.Settings`` directly via the ``settings`` parameter instead.

        Parameters:
            temperature: Temperature for speech synthesis.
            speaking_rate: Speaking rate for speech synthesis.
            timestamp_transport_strategy: The strategy to use for timestamp transport.
        """

        temperature: float | None = None
        speaking_rate: float | None = None
        timestamp_transport_strategy: Literal["ASYNC", "SYNC"] | None = "ASYNC"

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str | None = None,
        model: str | None = None,
        streaming: bool = True,
        sample_rate: int | None = None,
        encoding: str = "LINEAR16",
        timestamp_transport_strategy: Literal["ASYNC", "SYNC"] | None = "ASYNC",
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Inworld TTS service.

        Args:
            api_key: Inworld API key.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            voice_id: ID of the voice to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=InworldHttpTTSService.Settings(voice=...)`` instead.

            model: ID of the model to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=InworldHttpTTSService.Settings(model=...)`` instead.

            streaming: Whether to use streaming mode.
            sample_rate: Audio sample rate in Hz.
            encoding: Audio encoding format.
            timestamp_transport_strategy: Strategy for timestamp transport
                ("ASYNC" or "SYNC"). Defaults to "ASYNC".
            params: Input parameters for Inworld TTS configuration.

                .. deprecated:: 0.0.105
                    Use ``settings=InworldHttpTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent class.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="inworld-tts-1.5-max",
            voice="Ashley",
            language=None,
            speaking_rate=None,
            temperature=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.speaking_rate is not None:
                    default_settings.speaking_rate = params.speaking_rate
                if params.temperature is not None:
                    default_settings.temperature = params.temperature
                if params.timestamp_transport_strategy is not None:
                    timestamp_transport_strategy = params.timestamp_transport_strategy

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._streaming = streaming
        self._timestamp_type = "WORD"

        if streaming:
            self._base_url = "https://api.inworld.ai/tts/v1/voice:stream"
        else:
            self._base_url = "https://api.inworld.ai/tts/v1/voice"

        self._cumulative_time = 0.0
        self._current_run_had_timestamps = False

        # Init-only config (not runtime-updatable).
        self._audio_encoding = encoding
        self._audio_sample_rate = 0  # Set in start()
        self._timestamp_transport_strategy = timestamp_transport_strategy

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld TTS service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._audio_sample_rate = self.sample_rate

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            self._cumulative_time = 0.0

    def _calculate_word_times(
        self,
        timestamp_info: dict[str, Any],
    ) -> tuple[list[tuple[str, float]], float]:
        """Calculate word timestamps from Inworld HTTP API word-level response.

        Note: Inworld HTTP provides timestamps that reset for each request.
        We track cumulative time across requests to maintain continuity.

        Args:
            timestamp_info: The timestamp information from Inworld API.

        Returns:
            Tuple of (word_times, chunk_end_time) where chunk_end_time is the
            end time of the last word in this chunk (not cumulative).
        """
        word_times: list[tuple[str, float]] = []
        chunk_end_time = 0.0

        alignment = timestamp_info.get("wordAlignment", {})
        words = alignment.get("words", [])
        start_times = alignment.get("wordStartTimeSeconds", [])
        end_times = alignment.get("wordEndTimeSeconds", [])

        if words and start_times and len(words) == len(start_times):
            for i, word in enumerate(words):
                word_start = self._cumulative_time + start_times[i]
                word_times.append((word, word_start))

            # Track the end time of the last word in this chunk
            if end_times and len(end_times) > 0:
                chunk_end_time = end_times[-1]

        return (word_times, chunk_end_time)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate TTS audio for the given text.

        Args:
            text: The text to generate TTS audio for.
            context_id: Unique identifier for this TTS context.

        Returns:
            An asynchronous generator of frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}] (streaming={self._streaming})")

        self._current_run_had_timestamps = False

        audio_config = {
            "audioEncoding": self._audio_encoding,
            "sampleRateHertz": self._audio_sample_rate,
        }
        if self._settings.speaking_rate is not None:
            audio_config["speakingRate"] = self._settings.speaking_rate

        payload = {
            "text": text,
            "voiceId": self._settings.voice,
            "modelId": self._settings.model,
            "audioConfig": audio_config,
        }

        if self._settings.temperature is not None:
            payload["temperature"] = self._settings.temperature

        # Use WORD timestamps for simplicity and correct spacing/capitalization
        payload["timestampType"] = self._timestamp_type
        if self._timestamp_transport_strategy is not None:
            payload["timestampTransportStrategy"] = self._timestamp_transport_strategy

        request_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
            "X-User-Agent": USER_AGENT,
            "X-Request-Id": request_id,
        }

        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Inworld API error (request_id={request_id}): {error_text}")
                    yield ErrorFrame(error=f"Inworld API error: {error_text}")
                    return

                if self._streaming:
                    async for frame in self._process_streaming_response(response, context_id):
                        yield frame
                else:
                    async for frame in self._process_non_streaming_response(response, context_id):
                        yield frame

            await self.start_tts_usage_metrics(text)

            # If no timestamps were received, push the full text so the LLM
            # conversation context still reflects what the agent spoke. On
            # interruption this means the full text is committed rather than
            # only the portion that was spoken.
            if not self._current_run_had_timestamps:
                text_clean = text.rstrip()
                if text_clean:
                    logger.debug(
                        f"{self}: No timestamps received, pushing fallback text: [{text_clean}]"
                    )
                    fallback = TTSTextFrame(
                        text_clean, aggregated_by=AggregationType.SENTENCE, context_id=context_id
                    )
                    ctx = self._tts_contexts.get(context_id)
                    fallback.append_to_context = ctx.append_to_context if ctx else True
                    await self.push_frame(fallback)

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        finally:
            await self.stop_all_metrics()

    async def _process_streaming_response(
        self, response: aiohttp.ClientResponse, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Process a streaming response from the Inworld API.

        Args:
            response: The response from the Inworld API.
            context_id: Unique identifier for this TTS context.

        Yields:
            An asynchronous generator of frames.
        """
        buffer = b""
        # Track the duration of this utterance based on the last word's end time
        utterance_duration = 0.0

        async for chunk in response.content.iter_any():
            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_str = line.decode("utf-8").strip()

                if not line_str:
                    continue

                try:
                    chunk_data = json.loads(line_str)

                    if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                        await self.stop_ttfb_metrics()
                        async for frame in self._process_audio_chunk(
                            base64.b64decode(chunk_data["result"]["audioContent"]), context_id
                        ):
                            yield frame

                    if "result" in chunk_data and "timestampInfo" in chunk_data["result"]:
                        timestamp_info = chunk_data["result"]["timestampInfo"]
                        word_times, chunk_end_time = self._calculate_word_times(timestamp_info)
                        if word_times:
                            self._current_run_had_timestamps = True
                            await self.add_word_timestamps(
                                word_times, context_id, includes_inter_frame_spaces=True
                            )
                        # Track the maximum end time across all chunks
                        utterance_duration = max(utterance_duration, chunk_end_time)

                except json.JSONDecodeError:
                    continue

        # After processing all chunks, add the total utterance duration
        # to the cumulative time to ensure next utterance starts after this one
        if utterance_duration > 0:
            self._cumulative_time += utterance_duration

    async def _process_non_streaming_response(
        self, response: aiohttp.ClientResponse, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Process a non-streaming response from the Inworld API.

        Args:
            response: The response from the Inworld API.
            context_id: Unique identifier for this TTS context.

        Returns:
            An asynchronous generator of frames.
        """
        response_data = await response.json()

        if "audioContent" not in response_data:
            logger.error("No audioContent in Inworld API response")
            yield ErrorFrame(error="No audioContent in response")
            return

        utterance_duration = 0.0
        if "timestampInfo" in response_data:
            timestamp_info = response_data["timestampInfo"]
            word_times, chunk_end_time = self._calculate_word_times(timestamp_info)
            if word_times:
                self._current_run_had_timestamps = True
                await self.add_word_timestamps(
                    word_times, context_id, includes_inter_frame_spaces=True
                )
            utterance_duration = chunk_end_time

        audio_data = base64.b64decode(response_data["audioContent"])

        if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
            audio_data = audio_data[44:]

        chunk_size = self.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            if chunk:
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=chunk, sample_rate=self.sample_rate, num_channels=1, context_id=context_id
                )

        # After processing all audio, add the utterance duration to cumulative time
        if utterance_duration > 0:
            self._cumulative_time += utterance_duration

    async def _process_audio_chunk(
        self, audio_chunk: bytes, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk from the Inworld API.

        Args:
            audio_chunk: The audio chunk to process.
            context_id: Unique identifier for this TTS context.

        Returns:
            An asynchronous generator of frames.
        """
        if not audio_chunk:
            return

        audio_data = audio_chunk

        if len(audio_chunk) > 44 and audio_chunk.startswith(b"RIFF"):
            audio_data = audio_chunk[44:]

        if audio_data:
            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )


class InworldTTSService(WebsocketTTSService):
    """Inworld AI WebSocket-based TTS service.

    Uses bidirectional WebSocket for lower latency streaming. Supports multiple
    independent audio contexts per connection (max 5). Outputs LINEAR16 audio
    with word-level timestamps.
    """

    Settings = InworldTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Inworld WebSocket TTS configuration.

        .. deprecated:: 0.0.105
            Use ``InworldTTSService.Settings`` directly via the ``settings`` parameter instead.

        Parameters:
            temperature: Temperature for speech synthesis.
            speaking_rate: Speaking rate for speech synthesis.
            apply_text_normalization: Whether to apply text normalization.
            max_buffer_delay_ms: Maximum buffer delay in milliseconds.
            buffer_char_threshold: Buffer character threshold.
            auto_mode: Whether to use auto mode. Recommended when texts are sent
                in full sentences/phrases. When enabled, the server controls
                flushing of buffered text to achieve minimal latency while
                maintaining high quality audio output. If None (default),
                automatically set based on aggregate_sentences.
            timestamp_transport_strategy: The strategy to use for timestamp transport.
        """

        temperature: float | None = None
        speaking_rate: float | None = None
        apply_text_normalization: str | None = None
        max_buffer_delay_ms: int | None = None
        buffer_char_threshold: int | None = None
        auto_mode: bool | None = True
        timestamp_transport_strategy: Literal["ASYNC", "SYNC"] | None = "ASYNC"

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        model: str | None = None,
        url: str = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional",
        sample_rate: int | None = None,
        encoding: str = "LINEAR16",
        auto_mode: bool | None = None,
        apply_text_normalization: str | None = None,
        timestamp_transport_strategy: Literal["ASYNC", "SYNC"] | None = "ASYNC",
        params: InputParams | None = None,
        settings: Settings | None = None,
        aggregate_sentences: bool | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        append_trailing_space: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Inworld WebSocket TTS service.

        Args:
            api_key: Inworld API key.
            voice_id: ID of the voice to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=InworldTTSService.Settings(voice=...)`` instead.

            model: ID of the model to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=InworldTTSService.Settings(model=...)`` instead.

            url: URL of the Inworld WebSocket API.
            sample_rate: Audio sample rate in Hz.
            encoding: Audio encoding format.
            auto_mode: Whether to use auto mode. When enabled, the server
                controls flushing of buffered text. If None (default),
                automatically set based on ``aggregate_sentences``.
            apply_text_normalization: Whether to apply text normalization.
            timestamp_transport_strategy: Strategy for timestamp transport
                ("ASYNC" or "SYNC"). Defaults to "ASYNC".
            params: Input parameters for Inworld WebSocket TTS configuration.

                .. deprecated:: 0.0.105
                    Use ``settings=InworldTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.

            text_aggregation_mode: How to aggregate text before synthesis.
            append_trailing_space: Whether to append a trailing space to text before sending to TTS.
            **kwargs: Additional arguments passed to the parent class.
        """
        # Derive auto_mode from aggregate_sentences if not explicitly set
        if auto_mode is None:
            auto_mode = True if aggregate_sentences is None else aggregate_sentences

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="inworld-tts-1.5-max",
            voice="Ashley",
            language=None,
            speaking_rate=None,
            temperature=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        _buffer_max_delay_ms = None
        _buffer_char_threshold = None
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.speaking_rate is not None:
                    default_settings.speaking_rate = params.speaking_rate
                if params.temperature is not None:
                    default_settings.temperature = params.temperature
                if params.apply_text_normalization is not None:
                    apply_text_normalization = params.apply_text_normalization
                if params.timestamp_transport_strategy is not None:
                    timestamp_transport_strategy = params.timestamp_transport_strategy
                if params.auto_mode is not None:
                    auto_mode = params.auto_mode
            _buffer_max_delay_ms = params.max_buffer_delay_ms
            _buffer_char_threshold = params.buffer_char_threshold

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_text_frames=False,
            push_stop_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            aggregate_sentences=aggregate_sentences,
            text_aggregation_mode=text_aggregation_mode,
            append_trailing_space=append_trailing_space,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._timestamp_type = "WORD"

        self._buffer_settings = {
            "maxBufferDelayMs": _buffer_max_delay_ms,
            "bufferCharThreshold": _buffer_char_threshold,
        }

        self._receive_task = None
        self._keepalive_task = None

        # Track cumulative time across generations for monotonic timestamps within a turn.
        # When auto_mode is enabled, the server controls generations and timestamps reset
        # to 0 after each generation, as indicated by a "flushCompleted" message. We
        # add _cumulative_time to maintain monotonically increasing timestamps.
        self._cumulative_time = 0.0
        # Track the end time of the last word in the current generation
        self._generation_end_time = 0.0

        # Context IDs already sent to the server via _send_context, used to
        # make _send_context idempotent so on_turn_context_created can eagerly
        # open contexts without causing duplicate creates in run_tts.
        self._sent_context_ids: set[str] = set()

        # Fallback tracking for when timestamps are not received. Without
        # timestamps, interruptions commit the full text rather than only the
        # portion that was spoken.
        self._context_texts: dict[str, str] = {}
        self._contexts_with_timestamps: set[str] = set()

        # Init-only config (not runtime-updatable).
        self._audio_encoding = encoding
        self._audio_sample_rate = 0  # Set in start()
        self._auto_mode = auto_mode
        self._apply_text_normalization = apply_text_normalization
        self._timestamp_transport_strategy = timestamp_transport_strategy

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld WebSocket TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld WebSocket TTS service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._audio_sample_rate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Inworld WebSocket TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld WebSocket TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio without closing the context.

        This triggers synthesis of all accumulated text in the buffer while
        keeping the context open for subsequent text. The context is only
        closed on interruption, disconnect, or end of session.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if flush_id and self._websocket:
            logger.trace(f"Flushing audio for context {flush_id}")
            await self._send_flush(flush_id)

    def _reset_generation_timing(self):
        """Reset the cumulative time and generation end time for a new generation."""
        self._cumulative_time = 0.0
        self._generation_end_time = 0.0

    async def on_turn_context_created(self, context_id: str):
        """Eagerly open the context on the server when a new turn starts.

        This overlaps server-side context creation with sentence aggregation
        time, so the context is ready by the time text arrives in run_tts.
        """
        try:
            await self._send_context(context_id)
        except Exception as e:
            logger.warning(f"{self}: Failed to pre-open context: {e}")

    def _calculate_word_times(self, timestamp_info: dict[str, Any]) -> list[tuple[str, float]]:
        """Calculate word timestamps from Inworld WebSocket API response.

        Adds cumulative time offset to maintain monotonically increasing timestamps
        across multiple generations within an agent turn. Also tracks the generation
        end time for updating cumulative time on flush.

        Args:
            timestamp_info: The timestamp information from Inworld API.

        Returns:
            List of (word, timestamp) tuples with cumulative offset applied.
        """
        word_times: list[tuple[str, float]] = []

        alignment = timestamp_info.get("wordAlignment", {})
        words = alignment.get("words", [])
        start_times = alignment.get("wordStartTimeSeconds", [])
        end_times = alignment.get("wordEndTimeSeconds", [])

        if words and start_times and len(words) == len(start_times):
            for i, word in enumerate(words):
                word_start = self._cumulative_time + start_times[i]
                word_times.append((word, word_start))

            # Track cumulative end time for this generation
            if end_times and len(end_times) > 0:
                self._generation_end_time = self._cumulative_time + end_times[-1]

            logger.trace(
                f"{self}: Word timestamps - raw_start_times={start_times}, "
                f"cumulative_offset={self._cumulative_time}, "
                f"adjusted_times={[t for _, t in word_times]}, "
                f"generation_end_time={self._generation_end_time}"
            )

        return word_times

    async def _close_context(self, context_id: str):
        if context_id and self._websocket:
            logger.info(f"{self}: Closing context {context_id} due to interruption or completion")
            try:
                await self._send_close_context(context_id)
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        self._sent_context_ids.discard(context_id)

    async def on_turn_context_completed(self):
        """Close the server-side context at end of turn.

        Sends close_context so contextClosed arrives immediately after the
        last audio byte.
        """
        ctx_id = self._turn_context_id
        await super().on_turn_context_completed()
        await self._close_context(ctx_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Callback invoked when an audio context has been interrupted."""
        await self._maybe_push_fallback_text(context_id)
        await self._close_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def _maybe_push_fallback_text(self, context_id: str):
        """Push the full text as fallback when no timestamps were received.

        so that the LLM conversation context still reflects what the agent spoke.
        Without timestamps, the full text is always committed — even on
        interruption — since there is no timing information to determine which
        portion was actually spoken.
        """
        if not context_id:
            return
        had_timestamps = context_id in self._contexts_with_timestamps
        text = self._context_texts.pop(context_id, "").strip()
        self._contexts_with_timestamps.discard(context_id)
        if had_timestamps or not text:
            return
        logger.debug(f"{self}: No timestamps for context {context_id}, pushing fallback: [{text}]")
        fallback = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        fallback.context_id = context_id
        ctx = self._tts_contexts.get(context_id)
        fallback.append_to_context = ctx.append_to_context if ctx else True
        await self.push_frame(fallback)

    def _get_websocket(self):
        """Get the websocket for the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _connect(self):
        """Connect to the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored but not applied to the active connection.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        await self._disconnect()
        await self._connect()

        return changed

    async def _connect_websocket(self):
        """Connect to the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            request_id = str(uuid.uuid4())
            logger.debug(f"Connecting to Inworld WebSocket TTS (request_id={request_id})")
            headers = [
                ("Authorization", f"Basic {self._api_key}"),
                ("X-User-Agent", USER_AGENT),
                ("X-Request-Id", request_id),
            ]
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Disconnect from the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Inworld WebSocket TTS")
                audio_contexts = self.get_audio_contexts()
                if audio_contexts:
                    for ctx_id in audio_contexts:
                        await self._send_close_context(ctx_id)
                await self._websocket.close()
                logger.debug("Disconnected from Inworld WebSocket TTS")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._sent_context_ids.clear()
            self._reset_generation_timing()
            self._context_texts.clear()
            self._contexts_with_timestamps.clear()
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Inworld."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message")
                continue

            result = msg.get("result", {})
            ctx_id = result.get("contextId") or result.get("context_id")

            # Log all incoming messages for debugging
            msg_types = [
                k
                for k in ["contextCreated", "audioChunk", "flushCompleted", "contextClosed"]
                if k in result
            ]
            logger.trace(f"{self}: Received message types={msg_types}, ctx_id={ctx_id}")

            # Check for errors
            status = result.get("status", {})
            if status.get("code", 0) != 0:
                error_msg = status.get("message", "Unknown error")
                error_code = status.get("code")

                # Handle "Context not found" error (code 5)
                # This can happen when a keepalive message is sent but no context is available.
                if error_code == 5 and "not found" in error_msg.lower():
                    logger.debug(f"{self}: Context {ctx_id} not found.")
                    continue

                # For other errors, push error frame
                await self.push_error(error_msg=f"Inworld API error: {error_msg}")
                continue

            if "error" in msg:
                await self.push_error(error_msg=str(msg["error"]))
                continue

            # Handle context created confirmation
            if "contextCreated" in result:
                logger.trace(f"{self}: Context created on server: {ctx_id}")

            # Process audio chunk
            audio_chunk = result.get("audioChunk", {})
            audio_b64 = audio_chunk.get("audioContent")

            if audio_b64:
                logger.trace(f"{self}: Processing audio chunk for context {ctx_id}")
                audio = base64.b64decode(audio_b64)
                if len(audio) > 44 and audio.startswith(b"RIFF"):
                    audio = audio[44:]
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=ctx_id)

                if ctx_id:
                    await self.append_to_audio_context(ctx_id, frame)

            # timestampInfo is inside audioChunk
            timestamp_info = audio_chunk.get("timestampInfo")
            if timestamp_info:
                word_times = self._calculate_word_times(timestamp_info)
                if word_times:
                    if ctx_id:
                        self._contexts_with_timestamps.add(ctx_id)
                    await self.add_word_timestamps(
                        word_times, ctx_id, includes_inter_frame_spaces=True
                    )

            # Handle flush completion, which indicates the end of a generation
            if "flushCompleted" in result:
                logger.trace(
                    f"{self}: Generation completed - updating cumulative_time: "
                    f"{self._cumulative_time} -> {self._generation_end_time}"
                )
                self._cumulative_time = self._generation_end_time

            # Handle context closed - context no longer exists on server
            if "contextClosed" in result:
                logger.debug(f"{self}: Context closed on server: {ctx_id}")
                await self._maybe_push_fallback_text(ctx_id)
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                await self.remove_audio_context(ctx_id)

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 60
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    context_id = self.get_active_audio_context_id()
                    if context_id:
                        keepalive_message = {
                            "send_text": {"text": ""},
                            "contextId": context_id,
                        }
                        logger.trace(f"Sending keepalive for context {context_id}")
                    else:
                        keepalive_message = {"send_text": {"text": ""}}
                        logger.trace("Sending keepalive without context")
                    await self._websocket.send(json.dumps(keepalive_message))
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    async def _send_context(self, context_id: str):
        """Send a context to the Inworld WebSocket TTS service.

        Idempotent: skips the send if this context was already opened on the
        server (e.g., eagerly via on_turn_context_created).

        Args:
            context_id: The context ID.
        """
        if context_id in self._sent_context_ids:
            return
        self._sent_context_ids.add(context_id)

        audio_config = {
            "audioEncoding": self._audio_encoding,
            "sampleRateHertz": self._audio_sample_rate,
        }
        if self._settings.speaking_rate is not None:
            audio_config["speakingRate"] = self._settings.speaking_rate

        create_config: dict[str, Any] = {
            "voiceId": self._settings.voice,
            "modelId": self._settings.model,
            "audioConfig": audio_config,
        }

        if self._settings.temperature is not None:
            create_config["temperature"] = self._settings.temperature
        if self._apply_text_normalization is not None:
            create_config["applyTextNormalization"] = self._apply_text_normalization
        if self._auto_mode is not None:
            create_config["autoMode"] = self._auto_mode
        if self._timestamp_transport_strategy is not None:
            create_config["timestampTransportStrategy"] = self._timestamp_transport_strategy

        # Set buffer settings for timely audio generation.
        # Use provided values or defaults that work well for streaming LLM output.
        create_config["maxBufferDelayMs"] = self._buffer_settings["maxBufferDelayMs"] or 3000
        create_config["bufferCharThreshold"] = self._buffer_settings["bufferCharThreshold"] or 250

        create_config["timestampType"] = self._timestamp_type

        msg = {"create": create_config, "contextId": context_id}
        logger.trace(f"{self}: Sending context create: {create_config}")
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_text(self, context_id: str, text: str):
        """Send text to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
            text: The text to send.
        """
        msg = {"send_text": {"text": text}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_flush(self, context_id: str):
        """Send a flush to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
        """
        msg = {"flush_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_close_context(self, context_id: str):
        """Send a close context to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
        """
        msg = {"close_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate TTS audio for the given text using the Inworld WebSocket TTS service.

        Args:
            text: The text to generate TTS audio for.
            context_id: Unique identifier for this TTS context.

        Returns:
            An asynchronous generator of frames.
        """
        logger.debug(f"{self}: Generating WebSocket TTS [{text}, for context: {context_id}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.audio_context_available(context_id):
                    self._reset_generation_timing()
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    await self._send_context(context_id)

                self._context_texts[context_id] = self._context_texts.get(context_id, "") + text

                await self._send_text(context_id, text)
                await self.start_tts_usage_metrics(text)

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
