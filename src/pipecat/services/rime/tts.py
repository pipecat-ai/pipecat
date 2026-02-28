#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime text-to-speech service implementations.

This module provides both WebSocket and HTTP-based text-to-speech services
using Rime's API for streaming and batch audio synthesis.
"""

import base64
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, ClassVar, Dict, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import (
    AudioContextTTSService,
    InterruptibleTTSService,
    TextAggregationMode,
    TTSService,
)
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Rime, you need to `pip install pipecat-ai[rime]`.")
    raise Exception(f"Missing module: {e}")


def language_to_rime_language(language: Language) -> str:
    """Convert pipecat Language to Rime language code.

    Args:
        language: The pipecat Language enum value.

    Returns:
        Three-letter language code used by Rime (e.g., 'eng' for English).
    """
    LANGUAGE_MAP = {
        Language.DE: "ger",
        Language.FR: "fra",
        Language.EN: "eng",
        Language.ES: "spa",
        Language.HI: "hin",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class RimeTTSSettings(TTSSettings):
    """Settings for Rime WS JSON and HTTP TTS services.

    Parameters:
        audioFormat: Audio output format.
        samplingRate: Audio sample rate.
        segment: Text segmentation mode ("immediate", "bySentence", "never").
        speedAlpha: Speech speed multiplier (mistv2 only).
        reduceLatency: Whether to reduce latency at potential quality cost (mistv2 only).
        pauseBetweenBrackets: Whether to add pauses between bracketed content (mistv2 only).
        phonemizeBetweenBrackets: Whether to phonemize bracketed content (mistv2 only).
        noTextNormalization: Whether to disable text normalization (mistv2 only).
        saveOovs: Whether to save out-of-vocabulary words (mistv2 only).
        inlineSpeedAlpha: Inline speed control markup.
        repetition_penalty: Token repetition penalty (arcana only, 1.0-2.0).
        temperature: Sampling temperature (arcana only, 0.0-1.0).
        top_p: Cumulative probability threshold (arcana only, 0.0-1.0).
    """

    audioFormat: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    samplingRate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    segment: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speedAlpha: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    reduceLatency: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pauseBetweenBrackets: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    phonemizeBetweenBrackets: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    noTextNormalization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    saveOovs: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    inlineSpeedAlpha: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    repetition_penalty: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"speaker": "voice"}


@dataclass
class RimeNonJsonTTSSettings(TTSSettings):
    """Settings for Rime non-JSON WS TTS service.

    Parameters:
        audioFormat: Audio output format.
        samplingRate: Audio sample rate.
        segment: Text segmentation mode ("immediate", "bySentence", "never").
        repetition_penalty: Token repetition penalty (1.0-2.0).
        temperature: Sampling temperature (0.0-1.0).
        top_p: Cumulative probability threshold (0.0-1.0).
    """

    audioFormat: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    samplingRate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    segment: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    repetition_penalty: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"speaker": "voice"}


class RimeTTSService(AudioContextTTSService):
    """Text-to-Speech service using Rime's websocket API.

    Uses Rime's websocket JSON API to convert text to speech with word-level timing
    information. Supports interruptions and maintains context across multiple messages
    within a turn.
    """

    _settings: RimeTTSSettings

    class InputParams(BaseModel):
        """Configuration parameters for Rime TTS service.

        Parameters:
            language: Language for synthesis. Defaults to English.
            segment: Text segmentation mode ("immediate", "bySentence", "never").
            repetition_penalty: Token repetition penalty (arcana only).
            temperature: Sampling temperature (arcana only).
            top_p: Cumulative probability threshold (arcana only).
            speed_alpha: Speech speed multiplier (mistv2 only).
            reduce_latency: Whether to reduce latency at potential quality cost (mistv2 only).
            pause_between_brackets: Whether to add pauses between bracketed content (mistv2 only).
            phonemize_between_brackets: Whether to phonemize bracketed content (mistv2 only).
            no_text_normalization: Whether to disable text normalization (mistv2 only).
            save_oovs: Whether to save out-of-vocabulary words (mistv2 only).
        """

        language: Optional[Language] = Language.EN
        segment: Optional[str] = None
        # Arcana params
        repetition_penalty: Optional[float] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        # Mistv2 params
        speed_alpha: Optional[float] = None
        reduce_latency: Optional[bool] = None
        pause_between_brackets: Optional[bool] = None
        phonemize_between_brackets: Optional[bool] = None
        no_text_normalization: Optional[bool] = None
        save_oovs: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "wss://users-ws.rime.ai/ws3",
        model: str = "arcana",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        text_aggregator: Optional[BaseTextAggregator] = None,
        text_aggregation_mode: Optional[TextAggregationMode] = None,
        aggregate_sentences: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize Rime TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.
            url: Rime websocket API endpoint.
            model: Model ID to use for synthesis.
            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.
            text_aggregator: Custom text aggregator for processing input text.

                .. deprecated:: 0.0.95
                    Use an LLMTextProcessor before the TTSService for custom text aggregation.

            text_aggregation_mode: How to aggregate incoming text before synthesis.
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.

            **kwargs: Additional arguments passed to parent class.
        """
        # Initialize with parent class settings for proper frame handling
        params = params or RimeTTSService.InputParams()

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            supports_word_timestamps=True,
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=RimeTTSSettings(
                model=model,
                voice=voice_id,
                audioFormat="pcm",
                samplingRate=0,  # updated in start()
                language=self.language_to_service_language(params.language)
                if params.language
                else None,
                segment=params.segment,
                inlineSpeedAlpha=None,  # Not applicable here
                # Arcana params
                repetition_penalty=params.repetition_penalty,
                temperature=params.temperature,
                top_p=params.top_p,
                # Mistv2 params
                speedAlpha=params.speed_alpha,
                reduceLatency=params.reduce_latency,
                pauseBetweenBrackets=params.pause_between_brackets,
                phonemizeBetweenBrackets=params.phonemize_between_brackets,
                noTextNormalization=params.no_text_normalization,
                saveOovs=params.save_oovs,
            ),
            **kwargs,
        )

        if not text_aggregator:
            # Always skip tags added for spelled-out text
            # Note: This is primarily to support backwards compatibility.
            #    The preferred way of taking advantage of Rime spelling is
            #    to use an LLMTextProcessor and/or a text_transformer to identify
            #    and insert these tags for the purpose of the TTS service alone.
            self._text_aggregator = SkipTagsAggregator(
                [("spell(", ")")], aggregation_type=self._text_aggregation_mode
            )

        # Store service configuration
        self._api_key = api_key
        self._url = url

        # State tracking
        self._receive_task = None
        self._cumulative_time = 0  # Accumulates time across messages
        self._extra_msg_fields = {}  # Extra fields for next message

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    def _build_ws_params(self) -> dict[str, Any]:
        """Build query params for the WebSocket URL from current settings.

        Returns:
            Dictionary of query parameters for the WebSocket URL.
            Only explicitly-set values are included. Boolean mistv2 params
            are serialized with ``json.dumps()`` for the wire format.
        """
        params: dict[str, Any] = {
            "speaker": self._settings.voice,
            "modelId": self._settings.model,
            "audioFormat": self._settings.audioFormat,
            "samplingRate": self._settings.samplingRate,
        }
        if self._settings.language is not None:
            params["lang"] = self._settings.language
        if self._settings.segment is not None:
            params["segment"] = self._settings.segment
        if self._settings.speedAlpha is not None:
            params["speedAlpha"] = self._settings.speedAlpha

        if self._settings.model == "arcana":
            if self._settings.repetition_penalty is not None:
                params["repetition_penalty"] = self._settings.repetition_penalty
            if self._settings.temperature is not None:
                params["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                params["top_p"] = self._settings.top_p
        else:  # mistv2/mist
            if self._settings.reduceLatency is not None:
                params["reduceLatency"] = self._settings.reduceLatency
            if self._settings.pauseBetweenBrackets is not None:
                params["pauseBetweenBrackets"] = json.dumps(self._settings.pauseBetweenBrackets)
            if self._settings.phonemizeBetweenBrackets is not None:
                params["phonemizeBetweenBrackets"] = json.dumps(
                    self._settings.phonemizeBetweenBrackets
                )
            if self._settings.noTextNormalization is not None:
                params["noTextNormalization"] = json.dumps(self._settings.noTextNormalization)
            if self._settings.saveOovs is not None:
                params["saveOovs"] = json.dumps(self._settings.saveOovs)

        return params

    # A set of Rime-specific helpers for text transformations
    def SPELL(text: str) -> str:
        """Wrap text in Rime spell function."""
        return f"spell({text})"

    def PAUSE_TAG(seconds: float) -> str:
        """Convenience method to create a pause tag."""
        return f"<{seconds * 1000}>"

    def PRONOUNCE(self, text: str, word: str, phoneme: str) -> str:
        """Convenience method to support Rime's custom pronunciations feature.

        https://docs.rime.ai/api-reference/custom-pronunciation
        """
        self._extra_msg_fields["phonemizeBetweenBrackets"] = True
        return text.replace(word, f"{phoneme}")

    def INLINE_SPEED(self, text: str, speed: float) -> str:
        """Convenience method to support inline speeds."""
        if not self._extra_msg_fields:
            self._extra_msg_fields = {}
        speed_vals = self._extra_msg_fields.get("inlineSpeedAlpha", "").split(",")
        self._extra_msg_fields["inlineSpeedAlpha"] = ",".join(speed_vals + [str(speed)])
        return f"[{text}]"

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if necessary.

        Since all settings are WebSocket URL query parameters,
        any setting change requires reconnecting to apply the new values.
        """
        changed = await super()._update_settings(delta)

        if changed and self._websocket:
            await self._disconnect()
            await self._connect()

        return changed

    def _build_msg(self, text: str = "") -> dict:
        """Build JSON message for Rime API."""
        msg = {"text": text, "contextId": self.get_active_audio_context_id()}
        if self._extra_msg_fields:
            msg |= self._extra_msg_fields
            self._extra_msg_fields = {}
        return msg

    def _build_clear_msg(self) -> dict:
        """Build clear operation message."""
        return {"operation": "clear"}

    def _build_eos_msg(self) -> dict:
        """Build end-of-stream operation message."""
        return {"operation": "eos"}

    async def start(self, frame: StartFrame):
        """Start the service and establish websocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings.samplingRate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel current operation and clean up.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish websocket connection and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close websocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Rime websocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_params = self._build_ws_params()
            params = "&".join(f"{k}={v}" for k, v in ws_params.items() if v is not None)
            url = f"{self._url}?{params}"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(url, additional_headers=headers)

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close websocket connection and reset state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.send(json.dumps(self._build_eos_msg()))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _close_context(self, context_id: str):
        """Clear the Rime speech queue and stop metrics."""
        await self.stop_all_metrics()
        if context_id:
            await self._get_websocket().send(json.dumps(self._build_clear_msg()))

    async def on_audio_context_interrupted(self, context_id: str):
        """Clear the Rime speech queue and stop metrics when the bot is interrupted."""
        await self._close_context(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Clear server-side state and stop metrics after the Rime context finishes playing.

        Rime does not send a server-side completion signal (e.g. ``done`` / ``end_of_stream`` /
        ``audio_end``), so we explicitly send a ``clear`` message to clean up
        any residual server-side state once all audio has been delivered.
        """
        await self._close_context(context_id)

    def _calculate_word_times(self, words: list, starts: list, ends: list) -> list:
        """Calculate word timing pairs with proper spacing and punctuation.

        Args:
            words: List of words from Rime.
            starts: List of start times for each word.
            ends: List of end times for each word.

        Returns:
            List of (word, timestamp) pairs with proper timing.
        """
        word_pairs = []
        for i, (word, start_time, _) in enumerate(zip(words, starts, ends)):
            if not word.strip():
                continue

            # Adjust timing by adding cumulative time
            adjusted_start = start_time + self._cumulative_time

            # Handle punctuation by appending to previous word
            is_punctuation = bool(word.strip(",.!?") == "")
            if is_punctuation and word_pairs:
                prev_word, prev_time = word_pairs[-1]
                word_pairs[-1] = (prev_word + word, prev_time)
            else:
                word_pairs.append((word, adjusted_start))

        return word_pairs

    async def flush_audio(self):
        """Flush any pending audio synthesis."""
        context_id = self.get_active_audio_context_id()
        if not context_id or not self._websocket:
            return

        logger.trace(f"{self}: flushing audio")
        await self._get_websocket().send(json.dumps({"operation": "flush"}))
        self.reset_active_audio_context()

    async def _receive_messages(self):
        """Process incoming websocket messages."""
        async for message in self._get_websocket():
            msg = json.loads(message)

            if not msg or not self.audio_context_available(msg.get("contextId")):
                continue

            context_id = msg["contextId"]
            if msg["type"] == "chunk":
                # Process audio chunk
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                await self.append_to_audio_context(context_id, frame)

            elif msg["type"] == "timestamps":
                # Process word timing information
                timestamps = msg.get("word_timestamps", {})
                words = timestamps.get("words", [])
                starts = timestamps.get("start", [])
                ends = timestamps.get("end", [])

                if words and starts:
                    # Calculate word timing pairs
                    word_pairs = self._calculate_word_times(words, starts, ends)
                    if word_pairs:
                        await self.add_word_timestamps(word_pairs, context_id=context_id)
                        self._cumulative_time = ends[-1] + self._cumulative_time
                        logger.debug(f"Updated cumulative time to: {self._cumulative_time}")

            elif msg["type"] == "error":
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Error: {msg['message']}")
                self.reset_active_audio_context()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push frame and handle end-of-turn conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Rime's streaming API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.has_active_audio_context():
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    self._cumulative_time = 0
                    await self.create_audio_context(context_id)

                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class RimeHttpTTSService(TTSService):
    """Rime HTTP-based text-to-speech service.

    Provides text-to-speech synthesis using Rime's HTTP API for batch processing.
    Suitable for use cases where streaming is not required.
    """

    _settings: RimeTTSSettings

    class InputParams(BaseModel):
        """Configuration parameters for Rime HTTP TTS service.

        Parameters:
            language: Language for synthesis. Defaults to English.
            pause_between_brackets: Whether to add pauses between bracketed content.
            phonemize_between_brackets: Whether to phonemize bracketed content.
            inline_speed_alpha: Inline speed control markup.
            speed_alpha: Speech speed multiplier. Defaults to 1.0.
            reduce_latency: Whether to reduce latency at potential quality cost.
        """

        language: Optional[Language] = Language.EN
        pause_between_brackets: Optional[bool] = False
        phonemize_between_brackets: Optional[bool] = False
        inline_speed_alpha: Optional[str] = None
        speed_alpha: Optional[float] = 1.0
        reduce_latency: Optional[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "mistv2",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Rime HTTP TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            model: Model ID to use for synthesis.
            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        params = params or RimeHttpTTSService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            settings=RimeTTSSettings(
                model=model,
                language=self.language_to_service_language(params.language)
                if params.language
                else "eng",
                audioFormat="pcm",
                samplingRate=0,
                segment=None,
                speedAlpha=params.speed_alpha,
                reduceLatency=params.reduce_latency,
                pauseBetweenBrackets=params.pause_between_brackets,
                phonemizeBetweenBrackets=params.phonemize_between_brackets,
                noTextNormalization=None,
                saveOovs=None,
                inlineSpeedAlpha=params.inline_speed_alpha if params.inline_speed_alpha else None,
                repetition_penalty=None,
                temperature=None,
                top_p=None,
                voice=voice_id,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = "https://users.rime.ai/v1/rime-tts"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Rime's HTTP API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "lang": self._settings.language,
            "speedAlpha": self._settings.speedAlpha,
            "reduceLatency": self._settings.reduceLatency,
            "pauseBetweenBrackets": self._settings.pauseBetweenBrackets,
            "phonemizeBetweenBrackets": self._settings.phonemizeBetweenBrackets,
        }
        if self._settings.inlineSpeedAlpha is not None:
            payload["inlineSpeedAlpha"] = self._settings.inlineSpeedAlpha
        payload["text"] = text
        payload["speaker"] = self._settings.voice
        payload["modelId"] = self._settings.model
        payload["samplingRate"] = self.sample_rate

        # Arcana does not support PCM audio
        if payload["modelId"] == "arcana":
            headers["Accept"] = "audio/wav"
            need_to_strip_wav_header = True
        else:
            need_to_strip_wav_header = False

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_message = f"Rime TTS error: HTTP {response.status}"
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame(context_id=context_id)

                CHUNK_SIZE = self.chunk_size

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(CHUNK_SIZE),
                    strip_wav_header=need_to_strip_wav_header,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)


class RimeNonJsonTTSService(InterruptibleTTSService):
    """Pipecat TTS service for Rime's non-JSON WebSocket API.

    .. deprecated:: 0.0.102
        Arcana now supports JSON WebSocket with word-level timestamps via the
        ``wss://users-ws.rime.ai/ws3`` endpoint. Use :class:`RimeTTSService`
        with ``model="arcana"`` instead.

    This service enables Text-to-Speech synthesis over WebSocket endpoints
    that require plain text (not JSON) messages and return raw audio bytes.

    Limitations:
        - Does not support word-level timestamps or context IDs.
        - Intended specifically for integrations where the TTS provider only
          accepts and returns non-JSON messages.
    """

    _settings: RimeNonJsonTTSSettings

    class InputParams(BaseModel):
        """Configuration parameters for Rime Non-JSON WebSocket TTS service.

        Args:
            language: Language for synthesis. Defaults to English.
            segment: Text segmentation mode ("immediate", "bySentence", "never").
            repetition_penalty: Token repetition penalty (1.0-2.0).
            temperature: Sampling temperature (0.0-1.0).
            top_p: Cumulative probability threshold (0.0-1.0).
            extra: Additional parameters to pass to the API (for future compatibility).
        """

        language: Optional[Language] = None
        segment: Optional[str] = None
        repetition_penalty: Optional[float] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        extra: Optional[dict[str, Any]] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "wss://users.rime.ai/ws",
        model: str = "arcana",
        audio_format: str = "pcm",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        aggregate_sentences: Optional[bool] = None,
        text_aggregation_mode: Optional[TextAggregationMode] = None,
        **kwargs,
    ):
        """Initialize Rime Non-JSON WebSocket TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.
            url: Rime websocket API endpoint.
            model: Model ID to use for synthesis.
            audio_format: Audio format to use.
            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead. Set to ``TextAggregationMode.SENTENCE``
                    to aggregate text into sentences before synthesis, or
                    ``TextAggregationMode.TOKEN`` to stream tokens directly for lower latency.

            text_aggregation_mode: How to aggregate text before synthesis.
            **kwargs: Additional arguments passed to parent class.
        """
        params = params or RimeNonJsonTTSService.InputParams()
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=aggregate_sentences,
            text_aggregation_mode=text_aggregation_mode,
            push_stop_frames=True,
            pause_frame_processing=True,
            append_trailing_space=True,
            settings=RimeNonJsonTTSSettings(
                voice=voice_id,
                model=model,
                audioFormat=audio_format,
                samplingRate=sample_rate,
                language=self.language_to_service_language(params.language)
                if params.language
                else None,
                segment=params.segment,
                repetition_penalty=params.repetition_penalty,
                temperature=params.temperature,
                top_p=params.top_p,
            ),
            **kwargs,
        )
        self._api_key = api_key
        self._url = url
        # Add any extra parameters for future compatibility
        if params.extra:
            self._settings.extra.update(params.extra)

        self._receive_task = None
        self._context_id: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime Non-JSON WebSocket service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert pipecat Language enum to Rime language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            Three-letter Rime language code (e.g., 'eng' for English).
            Falls back to the language's base code with a warning if not in the verified list.
        """
        return language_to_rime_language(language)

    async def start(self, frame: StartFrame):
        """Start the Rime Non-JSON WebSocket TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings.samplingRate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connection."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel current operation and clean up."""
        await super().cancel(frame)
        await self._disconnect()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)

    async def _connect(self):
        """Establish WebSocket connection and start receive task."""
        await super()._connect()

        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close WebSocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Rime non-JSON websocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            # Build URL with query parameters (only given, non-None values)
            settings_dict = {
                "speaker": self._settings.voice,
                "modelId": self._settings.model,
                "audioFormat": self._settings.audioFormat,
                "samplingRate": self._settings.samplingRate,
            }
            if self._settings.language is not None:
                settings_dict["lang"] = self._settings.language
            if self._settings.segment is not None:
                settings_dict["segment"] = self._settings.segment
            if self._settings.repetition_penalty is not None:
                settings_dict["repetition_penalty"] = self._settings.repetition_penalty
            if self._settings.temperature is not None:
                settings_dict["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                settings_dict["top_p"] = self._settings.top_p
            # Include extras
            settings_dict.update(self._settings.extra)
            params = "&".join(f"{k}={v}" for k, v in settings_dict.items() if v is not None)
            url = f"{self._url}?{params}"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(
                url, additional_headers=headers, max_size=1024 * 1024 * 16
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                # Send EOS command to gracefully close
                await self._websocket.send("<EOS>")
                await self._websocket.close()
                logger.debug("Disconnected from Rime non-JSON websocket")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._context_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active WebSocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self):
        """Flush any pending audio synthesis."""
        if not self._websocket:
            return

        logger.trace(f"{self}: flushing audio")
        await self._websocket.send("<FLUSH>")

    async def _receive_messages(self):
        """Process incoming WebSocket messages (raw audio bytes)."""
        async for message in self._get_websocket():
            try:
                # Rime Arcana sends raw audio bytes directly (not JSON)
                if isinstance(message, bytes):
                    await self.stop_ttfb_metrics()

                    frame = TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=self._context_id,
                    )
                    await self.push_frame(frame)
            except Exception as e:
                await self.push_error(error_msg=f"Error: {e}", exception=e)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Rime's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()
            try:
                await self.start_ttfb_metrics()
                # Store context_id for use in _receive_messages
                self._context_id = context_id
                yield TTSStartedFrame(context_id=context_id)
                # Send bare text (not JSON)
                await self._get_websocket().send(text)
                await self.start_tts_usage_metrics(text)

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if necessary.

        Since all settings are WebSocket URL query parameters,
        any setting change requires reconnecting to apply the new values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            logger.debug("Settings changed, reconnecting WebSocket with new parameters")
            await self._disconnect()
            await self._connect()

        return changed
