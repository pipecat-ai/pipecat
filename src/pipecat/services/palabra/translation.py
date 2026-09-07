#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Palabra AI speech-to-speech translation service implementation.

Streams the user's audio to Palabra's Speech-to-Speech Translation API and
pushes back, over the same WebSocket, the source transcription, the text
translation into each target language, and the synthesized translated speech.
The service sits where an STT service usually goes and needs no LLM or TTS
behind it: a pipeline of ``transport.input() -> service -> transport.output()``
is a complete live interpreter.

Palabra API reference: https://docs.palabra.ai/docs/streaming_api/management
"""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

from loguru import logger
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TranslationFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import PALABRA_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

# ``{random_hash}`` is replaced with a fresh random string per connection; Palabra
# uses it to spread connections across translation servers.
DEFAULT_URL = "wss://streaming.palabra.ai/streaming-api/{random_hash}/v1/speech-to-speech/stream"

# Palabra always synthesizes translated speech at 24 kHz mono.
OUTPUT_SAMPLE_RATE = 24000

# Input sample rates Palabra accepts for raw PCM over WebSocket.
MIN_INPUT_SAMPLE_RATE = 16000
MAX_INPUT_SAMPLE_RATE = 48000

# Palabra recommends sending audio in chunks of about 320 ms.
DEFAULT_AUDIO_CHUNK_MS = 320

# Palabra allows one ``get_task`` every 2 seconds.
GET_TASK_INTERVAL = 2.1

# Seconds to wait for ``end_of_stream`` after ``end_task`` before closing anyway.
END_OF_STREAM_TIMEOUT = 10.0

DEFAULT_MESSAGE_TYPES = [
    "partial_transcription",
    "validated_transcription",
    "translated_transcription",
]


class _Drop(Enum):
    """Sentinel for a target language whose audio is not played back."""

    TOKEN = auto()


def language_to_palabra_language(language: Language) -> str:
    """Convert a Pipecat Language to a Palabra language code.

    For the list of supported languages, see:
    https://docs.palabra.ai/docs/streaming_api/realtime_stt
    """
    return language.value.lower()


def _to_language(code: str | None) -> Language | None:
    """Convert a Palabra language code to a Language enum, or None if unknown."""
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        return None


@dataclass
class PalabraTranslationSettings(STTSettings):
    """Settings for PalabraTranslationService.

    Changing any setting re-sends the translation task to Palabra, which
    applies it to the running pipeline without reconnecting.

    Parameters:
        language: Source language. ``None`` enables automatic language detection
            (experimental on Palabra's side).
        target_languages: Languages to translate into. Each one gets its own
            text translation and, with ``generate_speech`` enabled, its own
            synthesized speech. Required.
        generate_speech: Whether Palabra synthesizes the translations. ``False``
            turns the service into a transcription-plus-translation stream with
            no audio output.
        voice: Voice id used for every target language. ``None`` uses Palabra's
            default voice for each language.
        voice_cloning: Whether Palabra clones the speaker's voice for the
            translated speech. Takes precedence over ``voice``. ``None`` uses the
            Palabra default (disabled).
        translate_partials: Whether Palabra starts translating and speaking a
            segment before it is confirmed, trading accuracy for latency.
            ``None`` uses the Palabra default (disabled).
        silence_threshold: Seconds of silence after which Palabra confirms a
            segment. ``None`` uses the Palabra default.
        transcription_options: Extra fields merged into the task's
            ``pipeline.transcription`` object, for settings this class does not
            expose (glossaries, verification, sentence splitting, ...).
        translation_options: Extra fields merged into every entry of the task's
            ``pipeline.translations`` list.
        pipeline_options: Extra fields merged into the task's ``pipeline``
            object (for example ``translation_queue_configs``).
        audio_destinations: Where the translated speech of each target language
            is played. Maps a target language to a transport destination name
            (registered in the transport's ``audio_out_destinations``, e.g. a
            Daily custom audio track) or to ``None`` for the transport's default
            output. Languages left out of the mapping get no audio, only text.
            ``None`` sends every language to the default output, so several
            targets play over each other; set it whenever there is more than
            one target language.
    """

    target_languages: list[Language] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    generate_speech: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_cloning: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    translate_partials: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    silence_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    transcription_options: dict[str, Any] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    translation_options: dict[str, Any] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pipeline_options: dict[str, Any] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_destinations: dict[Language, str | None] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class PalabraTranslationService(WebsocketSTTService):
    """Speech-to-speech translation service using Palabra's Streaming API.

    The user's audio is batched into ``audio_chunk_ms`` chunks and sent as
    base64 ``input_audio_data`` messages. Palabra answers with:

    - ``partial_transcription`` and ``validated_transcription``, pushed as
      :class:`~pipecat.frames.frames.InterimTranscriptionFrame` and
      :class:`~pipecat.frames.frames.TranscriptionFrame`;
    - ``translated_transcription``, pushed as
      :class:`~pipecat.frames.frames.TranslationFrame` per target language;
    - ``output_audio_data``, pushed as 24 kHz mono
      :class:`~pipecat.frames.frames.TTSAudioRawFrame`, wrapped in
      :class:`~pipecat.frames.frames.TTSStartedFrame` and
      :class:`~pipecat.frames.frames.TTSStoppedFrame` so the output transport
      reports the interpreter as the speaking bot. With several target
      languages, ``audio_destinations`` routes each language's speech to its
      own transport destination; each destination gets its own started and
      stopped frames.

    The speaker is expected to keep talking while the translation plays, so
    the pipeline should not use interruption strategies. Call :meth:`interrupt`
    to drop the phrase currently being spoken and :meth:`speak` to inject text
    into the translation pipeline.

    Event handlers available:

    - on_task_ready: Called once Palabra confirms the translation task is running.

    For complete API documentation, see:
    https://docs.palabra.ai/docs/streaming_api/management
    """

    Settings = PalabraTranslationSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_URL,
        sample_rate: int | None = None,
        audio_chunk_ms: int = DEFAULT_AUDIO_CHUNK_MS,
        task_ready_timeout: float = 30.0,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = PALABRA_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Palabra translation service.

        Args:
            api_key: Palabra API key. Create one at https://platform.palabra.ai/api-keys.
            url: Palabra Speech-to-Speech WebSocket URL. May contain a
                ``{random_hash}`` placeholder, filled per connection.
            sample_rate: Input audio sample rate in Hz. If ``None``, inherits from
                the pipeline. Rates outside 16000-48000 are resampled to 16000
                before sending.
            audio_chunk_ms: Duration, in milliseconds, of the audio chunks sent to
                Palabra. Chunks must decode to at least 768 bytes.
            task_ready_timeout: Seconds to wait for Palabra to confirm the
                translation task before reporting an error.
            settings: Runtime-updatable settings. ``target_languages`` is required.
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds. Override for your deployment.
            **kwargs: Additional arguments passed to :class:`WebsocketSTTService`.
        """
        default_settings = self.Settings(
            model=None,
            language=Language.EN,
            target_languages=None,
            generate_speech=True,
            voice=None,
            voice_cloning=None,
            translate_partials=None,
            silence_threshold=None,
            transcription_options=None,
            translation_options=None,
            pipeline_options=None,
            audio_destinations=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        if not default_settings.target_languages:
            raise ValueError(
                "PalabraTranslationService requires at least one target language: "
                "pass settings=PalabraTranslationService.Settings(target_languages=[...])"
            )

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._audio_chunk_ms = audio_chunk_ms
        self._task_ready_timeout = task_ready_timeout

        self._input_sample_rate = 0
        self._resampler = create_stream_resampler()
        self._audio_buffer = bytearray()

        self._task_ready = asyncio.Event()
        self._end_of_stream = asyncio.Event()
        # Per transport destination: the (transcription_id, language) pairs whose
        # audio is still arriving, and the context id of the open speech stream.
        self._speaking: dict[str | None, set[tuple[str, str]]] = {}
        self._tts_context_ids: dict[str | None, str] = {}

        self._receive_task: asyncio.Task | None = None
        self._ready_task: asyncio.Task | None = None

        self._register_event_handler("on_task_ready")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Palabra translation supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Palabra language code.

        Args:
            language: The language to convert.

        Returns:
            The Palabra-specific language code.
        """
        return language_to_palabra_language(language)

    @property
    def task_ready(self) -> bool:
        """Whether Palabra has confirmed the translation task is running."""
        return self._task_ready.is_set()

    async def interrupt(self, languages: Sequence[Language] | None = None, pause: bool = False):
        """Drop the phrase Palabra is currently speaking.

        Nothing is finalized: the phrase in progress is discarded and the next
        one is processed normally.

        Args:
            languages: Target languages to interrupt. ``None`` interrupts all.
            pause: Whether to pause the task afterwards. Resume it by updating
                the settings, which re-sends the task.
        """
        codes = (
            [language_to_palabra_language(lang) for lang in languages] if languages else ["global"]
        )
        await self._send("interrupt_task", {"languages": codes, "pause_task": pause})

    async def speak(self, text: str, language: Language, translate: bool = False):
        """Inject text into the translation pipeline.

        Args:
            text: Text to speak, up to 2048 characters.
            language: With ``translate=False``, the target language to speak
                the text in; it must be one of ``target_languages``. With
                ``translate=True``, the language the text is written in.
            translate: Whether to translate the text into every target language
                before speaking it.
        """
        await self._send(
            "tts_task",
            {
                "text": text,
                "language": language_to_palabra_language(language),
                "translate_text": translate,
            },
        )

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and re-send the task so Palabra picks it up live.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed and self._websocket and self._websocket.state is State.OPEN:
            await self._send("set_task", self._build_task())
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        if MIN_INPUT_SAMPLE_RATE <= self.sample_rate <= MAX_INPUT_SAMPLE_RATE:
            self._input_sample_rate = self.sample_rate
        else:
            self._input_sample_rate = MIN_INPUT_SAMPLE_RATE
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Finish the task, letting Palabra deliver the translation tail first.

        Args:
            frame: The end frame.
        """
        if self._websocket and self._websocket.state is State.OPEN and self.task_ready:
            await self._flush_audio_buffer()
            self._end_of_stream.clear()
            await self._send("end_task", {})
            try:
                await asyncio.wait_for(self._end_of_stream.wait(), END_OF_STREAM_TIMEOUT)
            except TimeoutError:
                logger.warning(f"{self}: Palabra did not send end_of_stream, closing anyway")
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the service immediately.

        Args:
            frame: The cancel frame.
        """
        self._audio_buffer.clear()
        await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Buffer audio and send it to Palabra in ``audio_chunk_ms`` chunks.

        Audio received before Palabra confirms the task is dropped.

        Args:
            audio: Raw audio bytes to translate.

        Yields:
            None. Results arrive on the receive task.
        """
        if not self.task_ready:
            yield None
            return

        if self._input_sample_rate != self.sample_rate:
            audio = await self._resampler.resample(audio, self.sample_rate, self._input_sample_rate)

        self._audio_buffer.extend(audio)
        chunk_size = int(self._input_sample_rate * 2 * self._audio_chunk_ms / 1000)
        while len(self._audio_buffer) >= chunk_size:
            chunk = bytes(self._audio_buffer[:chunk_size])
            del self._audio_buffer[:chunk_size]
            await self._send_audio(chunk)
        yield None

    async def _send_audio(self, audio: bytes):
        await self._send("input_audio_data", {"data": base64.b64encode(audio).decode()})

    async def _flush_audio_buffer(self):
        if self._audio_buffer:
            chunk = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            await self._send_audio(chunk)

    async def _send(self, message_type: str, data: dict[str, Any]):
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"message_type": message_type, "data": data}))
            except Exception as e:
                logger.warning(f"{self}: send of {message_type} failed: {e}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def _build_task(self) -> dict[str, Any]:
        """Build the ``set_task`` payload from the current settings."""
        s = self._settings

        transcription: dict[str, Any] = {"source_language": s.language or "auto"}
        if s.silence_threshold is not None:
            transcription["segment_confirmation_silence_threshold"] = s.silence_threshold
        if s.transcription_options:
            transcription.update(s.transcription_options)

        speech_generation: dict[str, Any] = {}
        if s.voice_cloning:
            speech_generation["voice_cloning"] = True
        elif s.voice:
            speech_generation["voice_id"] = s.voice

        translations = []
        for target in s.target_languages or []:
            entry: dict[str, Any] = {
                "target_language": language_to_palabra_language(target),
                "speech_generation": dict(speech_generation),
            }
            if s.translate_partials is not None:
                entry["translate_partial_transcriptions"] = s.translate_partials
            if s.translation_options:
                entry.update(s.translation_options)
            translations.append(entry)

        message_types = list(DEFAULT_MESSAGE_TYPES)
        if s.translate_partials:
            message_types.append("partial_translated_transcription")

        pipeline: dict[str, Any] = {
            "transcription": transcription,
            "translations": translations,
            "allowed_message_types": message_types,
        }
        if s.pipeline_options:
            pipeline.update(s.pipeline_options)

        output_stream = None
        if s.generate_speech:
            output_stream = {
                "content_type": "audio",
                "target": {"type": "ws", "format": "pcm_s16le"},
            }

        return {
            "input_stream": {
                "content_type": "audio",
                "source": {
                    "type": "ws",
                    "format": "pcm_s16le",
                    "sample_rate": self._input_sample_rate,
                    "channels": 1,
                },
            },
            "output_stream": output_stream,
            "pipeline": pipeline,
        }

    async def _connect(self):
        """Connect to Palabra, send the task and wait for its confirmation."""
        await self._connect_websocket()

        await super()._connect()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._ready_task:
            self._ready_task = self.create_task(self._wait_task_ready())

    async def _disconnect(self):
        """Stop the tasks and close the connection."""
        await super()._disconnect()

        if self._ready_task:
            await self.cancel_task(self._ready_task)
            self._ready_task = None

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Palabra translation")

            url = self._url.format(random_hash=uuid.uuid4().hex)
            self._websocket = await self._websocket_connect(f"{url}?token={self._api_key}")

            self._task_ready.clear()
            self._end_of_stream.clear()
            await self._get_websocket().send(
                json.dumps({"message_type": "set_task", "data": self._build_task()})
            )

            await self._call_event_handler("on_connected")
            logger.debug("Connected to Palabra translation")
        except Exception as e:
            self._websocket = None
            await self.push_error(
                error_msg=f"Unable to connect to Palabra translation: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Palabra translation")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Palabra websocket: {e}", exception=e)
        finally:
            await self._finish_speaking()
            self._task_ready.clear()
            self._audio_buffer.clear()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect and restart the task-readiness poll."""
        result = await super()._reconnect_websocket(attempt_number)
        if result:
            if self._ready_task:
                await self.cancel_task(self._ready_task)
            self._ready_task = self.create_task(self._wait_task_ready())
        return result

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _wait_task_ready(self):
        """Poll ``get_task`` until Palabra reports the task as running.

        Palabra does not acknowledge ``set_task``; ``get_task`` answers with a
        ``NOT_FOUND`` error until the pipeline is up and with ``current_task``
        afterwards.
        """
        deadline = asyncio.get_running_loop().time() + self._task_ready_timeout
        while not self._task_ready.is_set():
            await self._send("get_task", {})
            try:
                await asyncio.wait_for(self._task_ready.wait(), GET_TASK_INTERVAL)
            except TimeoutError:
                pass
            if not self._task_ready.is_set() and asyncio.get_running_loop().time() > deadline:
                await self.push_error(
                    error_msg=(
                        f"Palabra did not confirm the translation task within "
                        f"{self._task_ready_timeout:.0f}s"
                    )
                )
                return
        self._ready_task = None

    async def _receive_messages(self):
        """Receive Palabra messages and push the resulting frames."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
                # The server may double-encode the payload.
                if isinstance(msg.get("data"), str):
                    msg["data"] = json.loads(msg["data"])
            except (json.JSONDecodeError, AttributeError):
                logger.warning(f"{self}: received malformed Palabra message: {message!r}")
                continue

            message_type = msg.get("message_type")
            data = msg.get("data") or {}

            if message_type == "output_audio_data":
                await self._handle_output_audio(data)
            elif message_type == "partial_transcription":
                await self._handle_transcription_message(data, is_final=False)
            elif message_type == "validated_transcription":
                await self._handle_transcription_message(data, is_final=True)
            elif message_type == "translated_transcription":
                await self._handle_translation_message(data)
            elif message_type == "current_task":
                if not self._task_ready.is_set():
                    self._task_ready.set()
                    logger.debug(f"{self}: Palabra translation task is running")
                    await self._call_event_handler("on_task_ready")
            elif message_type == "end_of_stream":
                await self._finish_speaking()
                self._end_of_stream.set()
            elif message_type == "warning":
                logger.warning(f"{self}: Palabra warning {data.get('code')}: {data.get('message')}")
            elif message_type == "error":
                code = data.get("code")
                if code == "NOT_FOUND" and not self._task_ready.is_set():
                    # Expected while polling get_task before the pipeline is up.
                    continue
                await self.push_error(
                    error_msg=f"Palabra translation error {code}: {data.get('desc')}"
                )
            else:
                logger.trace(f"{self}: ignoring Palabra message {message_type}")

    async def _handle_transcription_message(self, data: dict[str, Any], is_final: bool):
        transcription = data.get("transcription") or {}
        text = transcription.get("text", "")
        if not text:
            return
        language = _to_language(transcription.get("language"))
        if is_final:
            await self.emit_stt_usage_metrics()
            await self.push_frame(
                TranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=language,
                    result=transcription,
                )
            )
            await self._handle_transcription(text, is_final=True, language=language)
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=language,
                    result=transcription,
                )
            )

    async def _handle_translation_message(self, data: dict[str, Any]):
        transcription = data.get("transcription") or {}
        text = transcription.get("text", "")
        if not text:
            return
        await self.push_frame(
            TranslationFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=_to_language(transcription.get("language")),
            )
        )

    _DROP = _Drop.TOKEN

    def _destination_for(self, language: str) -> str | None | Literal[_Drop.TOKEN]:
        """Resolve the transport destination for a target language's audio.

        Returns the destination name, ``None`` for the default output, or
        ``_DROP`` when the language is excluded from playback.
        """
        mapping = self._settings.audio_destinations
        if not mapping:
            return None
        for lang, destination in mapping.items():
            if language_to_palabra_language(lang) == language:
                return destination
        return self._DROP

    async def _handle_output_audio(self, data: dict[str, Any]):
        language = str(data.get("language", ""))
        key = (str(data.get("transcription_id", "")), language)
        audio = base64.b64decode(data.get("data") or "")

        destination = self._destination_for(language)
        if destination is self._DROP:
            return

        speaking = self._speaking.setdefault(destination, set())
        if audio:
            if not speaking:
                context_id = str(uuid.uuid4())
                self._tts_context_ids[destination] = context_id
                started = TTSStartedFrame(context_id=context_id)
                started.transport_destination = destination
                await self.push_frame(started)
            speaking.add(key)
            frame = TTSAudioRawFrame(
                audio=audio,
                sample_rate=OUTPUT_SAMPLE_RATE,
                num_channels=1,
                context_id=self._tts_context_ids.get(destination),
            )
            frame.transport_destination = destination
            await self.push_frame(frame)

        if data.get("last_chunk"):
            speaking.discard(key)
            if not speaking:
                await self._finish_speaking(destination)

    async def _finish_speaking(self, destination: str | None | Literal[_Drop.TOKEN] = _Drop.TOKEN):
        """Close the open speech stream of ``destination``, or of every destination."""
        destinations = list(self._tts_context_ids) if destination is self._DROP else [destination]
        for dest in destinations:
            self._speaking.pop(dest, None)
            context_id = self._tts_context_ids.pop(dest, None)
            if context_id is not None:
                stopped = TTSStoppedFrame(context_id=context_id)
                stopped.transport_destination = dest
                await self.push_frame(stopped)
