#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA text-to-speech service implementation.

This module provides integration with NVIDIA TTS through
gRPC API for high-quality speech synthesis.

Refer to the NVIDIA TTS NIM documentation for usage, customization,
and local deployment steps:
https://docs.nvidia.com/nim/speech/latest/tts/
"""

import asyncio
import os
import queue
import textwrap
import threading
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, is_given
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.deprecation import deprecated

try:
    import grpc
    import riva.client
    import riva.client.proto.riva_tts_pb2 as rtts
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use NVIDIA TTS, you need to `uv add "pipecat-ai[nvidia]"`.')
    raise ImportError(f"Missing module: {e}") from e


class NvidiaTTSSynthesisMode(StrEnum):
    """Controls how text is sent to NVIDIA TTS.

    Parameters:
        PER_SENTENCE: Open a separate ``SynthesizeOnline`` call for each
            ``run_tts`` invocation. This is the default mode and can be used
            with all supported NVIDIA TTS NIMs, including Chatterbox,
            Magpie multilingual, and Magpie zero-shot. Text may still be
            chunked within that call to satisfy model request length limits.
        STITCHED: Reuse one ``SynthesizeOnline`` stream across multiple
            ``run_tts`` calls within the same LLM response. Enable this only
            for models with cross-sentence stitching support, such as Magpie
            multilingual and Magpie zero-shot v1.7.0 or later. Compatible
            models can still use ``per_sentence``, but ``stitched`` can improve
            multi-sentence synthesis quality. Individual ``run_tts`` inputs may
            still be chunked within the shared stream to satisfy model request
            length limits.
    """

    PER_SENTENCE = "per_sentence"
    STITCHED = "stitched"


@dataclass
class NvidiaTTSSettings(TTSSettings):
    """Settings for NvidiaTTSService.

    Parameters:
        quality: Audio quality setting (0-100). For Magpie zero-shot, NVIDIA
            expects values in the range ``1`` to ``40``.
        synthesis_mode: Whether to synthesize one sentence per request or stitch
            multiple sentences across a single streaming request.
    """

    quality: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    synthesis_mode: NvidiaTTSSynthesisMode | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _SynthesisStreamState:
    """Runtime state for one active synthesis stream."""

    context_id: str
    text_queue: queue.Queue
    response_queue: asyncio.Queue
    stop_event: threading.Event
    rpc_call: Any = None
    synth_task: asyncio.Task | None = None
    response_task: asyncio.Task | None = None


class NvidiaTTSService(TTSService):
    """NVIDIA TTS service.

    Provides high-quality text-to-speech synthesis using both locally deployed and cloud-based NVIDIA TTS models. Supports multiple voices, languages, and configurable
    quality settings.
    """

    Settings = NvidiaTTSSettings
    _settings: Settings
    _MAX_CHUNK_LEN = 200

    @deprecated(
        "`NvidiaTTSService.InputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. "
        "Use `NvidiaTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for NVIDIA TTS configuration.

        .. deprecated:: 0.0.105
            Use ``NvidiaTTSService.Settings`` directly via the ``settings`` parameter instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language code for synthesis. Defaults to US English.
            quality: Audio quality setting (0-100). For Magpie zero-shot,
                NVIDIA expects values in the range ``1`` to ``40``. Defaults to 20.
        """

        language: Language | None = Language.EN_US
        quality: int | None = 20

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str | None = None,
        sample_rate: int | None = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
            "model_name": "magpie-tts-multilingual",
        },
        params: InputParams | None = None,
        settings: Settings | None = None,
        use_ssl: bool = True,
        custom_dictionary: dict | None = None,
        zero_shot_audio_prompt_file: str | os.PathLike[str] | None = None,
        audio_prompt_encoding: AudioEncoding | None = AudioEncoding.ENCODING_UNSPECIFIED,
        encoding: AudioEncoding | None = AudioEncoding.LINEAR_PCM,
        **kwargs,
    ):
        """Initialize the NVIDIA TTS service.

        Args:
            api_key: NVIDIA API key for authentication. Required when using the
                cloud endpoint. Not needed for local deployments.
            server: gRPC server endpoint. Defaults to NVIDIA's cloud endpoint.
                For local deployments, pass the local address (e.g. ``localhost:50051``).
            voice_id: Voice model identifier. Defaults to multilingual Aria voice.

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Audio sample rate. If None, uses service default.
            model_function_map: Dictionary containing function_id and model_name for the TTS model.
            params: Additional configuration parameters for TTS synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside
                deprecated parameters, ``settings`` values take precedence.
            use_ssl: Whether to use SSL for the gRPC connection. Defaults to True
                for the NVIDIA cloud endpoint. Set to False for local deployments.
            custom_dictionary: Custom pronunciation dictionary mapping words
                (graphemes) to IPA phonetic representations (phonemes),
                e.g. ``{"NVIDIA": "ɛn.vɪ.diː.ʌ"}``. See
                https://docs.nvidia.com/nim/speech/latest/tts/phoneme-support.html
                for the list of supported IPA phonemes.
            zero_shot_audio_prompt_file: Optional audio prompt file for Magpie
                zero-shot voice cloning. NVIDIA recommends a 16-bit mono WAV
                prompt, sample rate 22.05 kHz or higher, and duration 3 to 10
                seconds. Access to NVIDIA's hosted zero-shot models requires
                approval through:
                https://developer.nvidia.com/riva-tts-zeroshot-models
            audio_prompt_encoding: Audio encoding for ``zero_shot_audio_prompt_file``.
                Use this when the server expects a specific prompt encoding for
                Magpie zero-shot voice cloning.
            encoding: Output audio encoding format. Defaults to ``AudioEncoding.LINEAR_PCM``.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=model_function_map.get("model_name"),
            voice="Magpie-Multilingual.EN-US.Aria",
            language=Language.EN_US,
            quality=20,
            synthesis_mode=NvidiaTTSSynthesisMode.PER_SENTENCE,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.quality is not None:
                    default_settings.quality = params.quality

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=False,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._server = server
        self._api_key = api_key
        self._function_id = model_function_map.get("function_id")
        self._use_ssl = use_ssl

        self._custom_dictionary: str | None = None
        if custom_dictionary:
            entries = [f"{k}  {v}" for k, v in custom_dictionary.items()]
            self._custom_dictionary = ",".join(entries)
        self._zero_shot_audio_prompt_file: Path | None = None
        self._zero_shot_audio_prompt_data: bytes | None = None
        if zero_shot_audio_prompt_file is not None:
            self._zero_shot_audio_prompt_file = Path(zero_shot_audio_prompt_file).expanduser()
        self._audio_prompt_encoding = audio_prompt_encoding
        self._encoding = encoding

        self._service = None
        self._config = None

        # Runtime state for the active streaming turn.
        self._stream_state: _SynthesisStreamState | None = None
        # In-flight gRPC call for per-sentence mode (used for cancellation).
        self._per_sentence_rpc_call: Any = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True as this service supports metric generation.
        """
        return True

    @deprecated(
        "`NvidiaTTSService.set_model` is deprecated since 0.0.104 and will be removed in 2.0.0. "
        "No replacement."
    )
    async def set_model(self, model: str):
        """Set the TTS model.

        .. deprecated:: 0.0.104
            No replacement.
            Will be removed in 2.0.0.

            Example::

                NvidiaTTSService(
                    api_key=...,
                    model_function_map={"function_id": "<UUID>", "model_name": "<model_name>"},
                )

        Args:
            model: The model name to set.
        """
        return None

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored and will take effect on the next synthesis turn.
        Mid-stream changes cannot be applied to the active gRPC connection.
        """
        changed = await super()._update_settings(delta)
        if changed:
            fields = ", ".join(sorted(changed))
            logger.debug(f"{self.name}: settings updated [{fields}], will apply on next turn")
        return changed

    def _initialize_client(self):
        if self._service is not None:
            return

        metadata = []
        if self._function_id:
            metadata.append(["function-id", self._function_id])
        if self._api_key:
            metadata.append(["authorization", f"Bearer {self._api_key}"])
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)

        self._service = riva.client.SpeechSynthesisService(auth)

    def _create_synthesis_config(self) -> rtts.RivaSynthesisConfigResponse:
        """Fetch and validate synthesis configuration from the server."""
        if not self._service:
            raise RuntimeError("TTS service not initialized")

        try:
            config = self._service.stub.GetRivaSynthesisConfig(
                riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
            )
            return config
        except grpc.RpcError as e:
            status = e.code().name if hasattr(e, "code") else "UNKNOWN"
            details = e.details() if hasattr(e, "details") else str(e)
            logger.error(
                f"{self} failed to get synthesis config from server (gRPC {status}): {details}"
            )
            raise RuntimeError(
                f"{self}: startup failed while fetching synthesis config (gRPC {status})"
            ) from e

    def _load_zero_shot_audio_prompt(self):
        """Load and cache zero-shot prompt audio bytes, if configured."""
        if (
            self._zero_shot_audio_prompt_file is None
            or self._zero_shot_audio_prompt_data is not None
        ):
            return

        try:
            with self._zero_shot_audio_prompt_file.open("rb") as prompt_file:
                self._zero_shot_audio_prompt_data = prompt_file.read()
        except OSError as e:
            raise RuntimeError(
                f"{self}: failed to read zero-shot audio prompt file "
                f"{self._zero_shot_audio_prompt_file}"
            ) from e

    async def start(self, frame: StartFrame):
        """Start the NVIDIA TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_synthesis_config()
        self._load_zero_shot_audio_prompt()
        logger.debug(f"Initialized NvidiaTTSService with model: {self._settings.model}")

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        """Release all resources held by the service."""
        await super().cleanup()
        await self._teardown()

    async def _teardown(self):
        """Abort the active synthesis stream and close the gRPC client.

        Idempotent so it can run from ``stop()``, ``cancel()``, and
        ``cleanup()`` without duplicating teardown work.
        """
        await self._abort_synthesis_stream()
        self._close_client()

    def _start_synthesis_stream(self, context_id: str):
        """Start a persistent gRPC synthesis stream for the current turn.

        Creates a queue-backed generator that feeds text to
        ``synthesize_online``. The gRPC stream stays open until a ``None``
        sentinel is pushed into the queue.
        """
        state = _SynthesisStreamState(
            context_id=context_id,
            text_queue=queue.Queue(),
            response_queue=asyncio.Queue(),
            stop_event=threading.Event(),
        )
        self._stream_state = state
        logger.debug(f"{self}: starting synthesis stream")

        state.synth_task = self.create_task(
            self._synth_task_handler(state), name="nvidia-tts-synth"
        )
        state.response_task = self.create_task(
            self._process_responses(state), name="nvidia-tts-response"
        )

    def _build_base_request(self) -> rtts.SynthesizeSpeechRequest:
        """Build a reusable ``SynthesizeSpeechRequest`` with current settings."""
        req = rtts.SynthesizeSpeechRequest(
            text="",
            language_code=str(self._settings.language or "en-US"),
            sample_rate_hz=self.sample_rate,
            encoding=self._encoding,
        )
        voice = self._settings.voice
        if voice:
            req.voice_name = voice
        if self._custom_dictionary:
            req.custom_dictionary = self._custom_dictionary
        if self._zero_shot_audio_prompt_data is not None:
            req.zero_shot_data.audio_prompt = self._zero_shot_audio_prompt_data
            if self._audio_prompt_encoding is not None:
                req.zero_shot_data.encoding = self._audio_prompt_encoding
            zero_shot_quality = self._settings.quality if is_given(self._settings.quality) else 20
            req.zero_shot_data.quality = int(zero_shot_quality)
        return req

    def _synthesis_handler(self, state: _SynthesisStreamState):
        """Run ``SynthesizeOnline`` gRPC stream in a blocking call.

        Uses a queue-backed generator to feed text chunks into a single
        ``SynthesizeOnline`` call, enabling Magpie's cross-sentence stitching.
        Audio responses are forwarded to the async response queue.
        """
        event_loop = self.get_event_loop()
        base_req = self._build_base_request()

        def request_generator():
            while True:
                if state.stop_event.is_set():
                    break
                text = state.text_queue.get()
                if text is None or state.stop_event.is_set():
                    break
                base_req.text = text
                yield base_req

        try:
            call = self._service.stub.SynthesizeOnline(
                request_generator(),
                metadata=self._service.auth.get_auth_metadata(),
            )
            state.rpc_call = call

            for resp in call:
                if state.stop_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(state.response_queue.put(resp), event_loop)
        except Exception as e:
            if not state.stop_event.is_set():
                logger.error(f"{self} gRPC synthesis stream error: {e}")
                asyncio.run_coroutine_threadsafe(state.response_queue.put(e), event_loop)
        finally:
            state.rpc_call = None
            asyncio.run_coroutine_threadsafe(state.response_queue.put(None), event_loop)

    async def _synth_task_handler(self, state: _SynthesisStreamState):
        """Wrap ``_synthesis_handler`` as an asyncio-managed task."""
        await asyncio.to_thread(self._synthesis_handler, state)

    def _cancel_stream_call(self, state: _SynthesisStreamState):
        """Best-effort cancellation of in-flight gRPC call."""
        call = state.rpc_call
        if call is not None and hasattr(call, "cancel"):
            try:
                call.cancel()
            except Exception as e:
                logger.debug(f"{self}: failed to cancel gRPC stream call: {e}")

    def _close_client(self):
        """Close the underlying gRPC channel and release client references."""
        auth = getattr(self._service, "auth", None)
        channel = getattr(auth, "channel", None)
        if channel is not None and hasattr(channel, "close"):
            try:
                channel.close()
            except Exception as e:
                logger.debug(f"{self}: failed to close gRPC channel: {e}")

        self._service = None
        self._config = None

    async def _process_responses(self, state: _SynthesisStreamState):
        """Consume gRPC responses and append audio to the active audio context."""
        while True:
            item = await state.response_queue.get()
            if item is None:
                if self.audio_context_available(state.context_id):
                    await self.remove_audio_context(state.context_id)
                break
            if isinstance(item, Exception):
                # Treat stream exceptions as terminal for this stream. Once
                # SynthesizeOnline raises, no further reliable audio is expected.
                # Ignore stale or interruption-driven exceptions to avoid noisy
                # errors during handoff to a new stream.
                if self._stream_state is state and not state.stop_event.is_set():
                    await self.push_error(f"{self} synthesis error: {item}")
                break

            # Stale stream responses must never leak into a newer stream context.
            if self._stream_state is not state:
                continue

            await self.stop_ttfb_metrics()
            frame = TTSAudioRawFrame(
                audio=item.audio,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=state.context_id,
            )
            await self.append_to_audio_context(state.context_id, frame)

        # Finalize ownership once the stream drains naturally.
        if self._stream_state is state and not state.stop_event.is_set():
            self._stream_state = None

    def _signal_synthesis_close(self, state: _SynthesisStreamState):
        """Signal the active synthesis request generator to close."""
        state.text_queue.put(None)

    async def _abort_synthesis_stream(self):
        """Abort the active gRPC synthesis stream immediately.

        Cancels the response task first to stop delivering audio, then
        drains the text queue and signals the synthesis handler to stop.
        Pending audio is discarded.
        """
        self._cancel_per_sentence_call()

        state = self._stream_state
        if state is None:
            return

        state.stop_event.set()

        if state.response_task is not None:
            await self.cancel_task(state.response_task)
            state.response_task = None

        while not state.text_queue.empty():
            try:
                state.text_queue.get_nowait()
            except queue.Empty:
                break
        self._signal_synthesis_close(state)

        self._cancel_stream_call(state)

        if state.synth_task is not None:
            await self.cancel_task(state.synth_task)
            state.synth_task = None

        if self._stream_state is state:
            self._stream_state = None

    def _cancel_per_sentence_call(self):
        """Best-effort cancellation of an in-flight per-sentence gRPC call."""
        call = self._per_sentence_rpc_call
        if call is not None and hasattr(call, "cancel"):
            try:
                call.cancel()
            except Exception as e:
                logger.debug(f"{self}: failed to cancel per-sentence gRPC call: {e}")
        self._per_sentence_rpc_call = None

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio and finalize the current context.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        state = self._stream_state
        if state is not None:
            self._signal_synthesis_close(state)
        await super().flush_audio(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the active gRPC synthesis stream when the bot is interrupted.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        await self._abort_synthesis_stream()
        await super().on_audio_context_interrupted(context_id)

    @staticmethod
    def _split_text_into_chunks(text: str) -> list[str]:
        """Split text into <=200-character chunks at whitespace boundaries.

        Magpie stitches chunks seamlessly in the gRPC stream, so splitting
        conservatively at 200 chars avoids max char limits without affecting audio
        quality.

        Args:
            text: Input text to split.

        Returns:
            List of text chunks, each at most 200 characters.
        """
        text = text.strip()
        if not text:
            return []
        return textwrap.wrap(
            text,
            width=NvidiaTTSService._MAX_CHUNK_LEN,
            break_long_words=True,
            break_on_hyphens=False,
        )

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using NVIDIA TTS.

        On the first call for a turn, starts a persistent ``synthesize_online``
        gRPC stream. Subsequent calls within the same turn feed text into the
        existing stream, enabling Magpie's cross-sentence stitching.

        Text is split into chunks respecting Magpie's per-request limits. Each chunk becomes
        a separate request in the gRPC stream, stitched seamlessly by Magpie.

        In ``per_sentence`` mode, opens a fresh ``SynthesizeOnline`` call per
        invocation and yields audio directly before returning.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            None on success in ``stitched`` mode. ``TTSAudioRawFrame`` objects
                in ``per_sentence`` mode. ``ErrorFrame`` on failure.
        """
        text = text.strip()
        if not text or not any(c.isalnum() for c in text):
            return

        try:
            assert self._service is not None, "TTS service not initialized"
            if self._settings.synthesis_mode == NvidiaTTSSynthesisMode.PER_SENTENCE:
                async for frame in self._run_tts_per_sentence(text, context_id):
                    yield frame
                return

            # First call for this turn: create audio context and start gRPC stream
            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)
                await self.start_ttfb_metrics()
                yield TTSStartedFrame(context_id=context_id)
                self._start_synthesis_stream(context_id)
                logger.trace(f"{self}: Started synthesis stream for context {context_id}")

            logger.debug(f"{self}: Generating TTS [{text}]")

            state = self._stream_state
            if state is None:
                raise RuntimeError("Synthesis stream not started")

            for chunk in self._split_text_into_chunks(text):
                if any(c.isalnum() for c in chunk):
                    state.text_queue.put(chunk)

            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")

    async def _run_tts_per_sentence(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        """Open one fresh ``SynthesizeOnline`` call per split chunk."""
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

        logger.debug(f"{self}: Generating TTS [{text}]")

        chunks = [
            chunk for chunk in self._split_text_into_chunks(text) if any(c.isalnum() for c in chunk)
        ]
        if not chunks:
            return

        await self.start_tts_usage_metrics(text)

        response_queue: asyncio.Queue = asyncio.Queue()
        event_loop = self.get_event_loop()
        stop_event = threading.Event()

        def run_grpc():
            try:
                for chunk in chunks:
                    if stop_event.is_set():
                        break

                    base_req = self._build_base_request()

                    def request_gen():
                        base_req.text = chunk
                        yield base_req

                    call = self._service.stub.SynthesizeOnline(
                        request_gen(),
                        metadata=self._service.auth.get_auth_metadata(),
                    )
                    self._per_sentence_rpc_call = call
                    try:
                        for resp in call:
                            if stop_event.is_set():
                                break
                            asyncio.run_coroutine_threadsafe(response_queue.put(resp), event_loop)
                    finally:
                        if self._per_sentence_rpc_call is call:
                            self._per_sentence_rpc_call = None
            except Exception as e:
                if not stop_event.is_set():
                    asyncio.run_coroutine_threadsafe(response_queue.put(e), event_loop)
            finally:
                self._per_sentence_rpc_call = None
                asyncio.run_coroutine_threadsafe(response_queue.put(None), event_loop)

        grpc_task = self.create_task(asyncio.to_thread(run_grpc), name="nvidia-tts-per-sentence")
        try:
            while True:
                item = await response_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=item.audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        finally:
            stop_event.set()
            self._cancel_per_sentence_call()
            await self.cancel_task(grpc_task)
