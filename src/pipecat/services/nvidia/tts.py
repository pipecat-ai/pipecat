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
    synthesis_mode: NvidiaTTSSynthesisMode = NvidiaTTSSynthesisMode.PER_SENTENCE


@dataclass
class _SynthesisState:
    """Runtime state for the active synthesis manager."""

    context_id: str
    mode: NvidiaTTSSynthesisMode
    text_queue: queue.Queue
    response_queue: asyncio.Queue
    stop_event: threading.Event
    rpc_call: Any = None
    synth_task: asyncio.Task | None = None
    response_task: asyncio.Task | None = None


_RECONNECT = object()
_CLOSE_STREAM = object()
_REQUEST_DONE = object()


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

        # Runtime state for the active synthesis manager.
        self._synthesis_state: _SynthesisState | None = None

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
        context_ids = self._active_context_ids()
        await self._abort_all_synthesis()
        for context_id in context_ids:
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)
        await super().stop(frame)
        self._close_client()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA TTS service.

        Args:
            frame: The cancel frame.
        """
        context_ids = self._active_context_ids()
        await self._abort_all_synthesis()
        for context_id in context_ids:
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)
        await super().cancel(frame)
        self._close_client()

    def _active_context_ids(self) -> tuple[str, ...]:
        """Return the active audio context IDs owned by in-flight synthesis."""
        if self._synthesis_state is None:
            return ()
        return (self._synthesis_state.context_id,)

    def _start_synthesis_state(self, context_id: str, mode: NvidiaTTSSynthesisMode):
        """Start the active synthesis manager for the requested mode."""
        state = _SynthesisState(
            context_id=context_id,
            mode=mode,
            text_queue=queue.Queue(),
            response_queue=asyncio.Queue(),
            stop_event=threading.Event(),
        )
        self._synthesis_state = state
        logger.debug(f"{self}: starting {mode} synthesis state")

        state.synth_task = self.create_task(
            asyncio.to_thread(self._synthesis_handler, state),
            name="nvidia-tts-synth",
        )
        if mode == NvidiaTTSSynthesisMode.STITCHED:
            state.response_task = self.create_task(
                self._process_stitched_responses(state), name="nvidia-tts-response"
            )

    async def _ensure_synthesis_state(
        self, context_id: str, mode: NvidiaTTSSynthesisMode
    ) -> _SynthesisState:
        """Return an active synthesis manager for the given context and mode."""
        state = self._synthesis_state
        if state is not None and state.context_id == context_id and state.mode == mode:
            return state

        await self._abort_all_synthesis()
        self._start_synthesis_state(context_id, mode)

        assert self._synthesis_state is not None
        return self._synthesis_state

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

    def _run_synthesis_call(
        self,
        requests: Any,
        state: _SynthesisState,
    ):
        """Run one blocking ``SynthesizeOnline`` gRPC call.

        Args:
            requests: Blocking iterator of ``SynthesizeSpeechRequest`` objects.
            state: Runtime state owning the active RPC call and response queue.
        """
        event_loop = self.get_event_loop()
        error_label = (
            "gRPC stitched synthesis stream error"
            if state.mode == NvidiaTTSSynthesisMode.STITCHED
            else "gRPC per-sentence synthesis request error"
        )
        try:
            call = self._service.stub.SynthesizeOnline(
                requests,
                metadata=self._service.auth.get_auth_metadata(),
            )
            state.rpc_call = call

            for resp in call:
                if state.stop_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(state.response_queue.put(resp), event_loop)
        except Exception as e:
            if not state.stop_event.is_set():
                logger.error(f"{self} {error_label}: {e}")
                asyncio.run_coroutine_threadsafe(state.response_queue.put(e), event_loop)
        finally:
            state.rpc_call = None
            asyncio.run_coroutine_threadsafe(state.response_queue.put(_REQUEST_DONE), event_loop)

    def _synthesis_handler(self, state: _SynthesisState):
        """Process queued synthesis requests for stitched or per-sentence mode."""
        try:
            while not state.stop_event.is_set():
                first_item = state.text_queue.get()
                if first_item is _CLOSE_STREAM or state.stop_event.is_set():
                    break
                if first_item is _RECONNECT:
                    continue

                base_req = self._build_base_request()
                close_reason = {"value": _RECONNECT}

                def request_generator():
                    current_item: str | object = first_item
                    while True:
                        if state.stop_event.is_set():
                            close_reason["value"] = _CLOSE_STREAM
                            return
                        if current_item is _RECONNECT:
                            close_reason["value"] = _RECONNECT
                            return
                        if current_item is _CLOSE_STREAM:
                            close_reason["value"] = _CLOSE_STREAM
                            return

                        base_req.text = current_item
                        yield base_req
                        current_item = state.text_queue.get()

                self._run_synthesis_call(request_generator(), state)
                if close_reason["value"] is _CLOSE_STREAM:
                    break
        finally:
            event_loop = self.get_event_loop()
            asyncio.run_coroutine_threadsafe(state.response_queue.put(None), event_loop)

    def _cancel_rpc_call(self, state: _SynthesisState):
        """Best-effort cancellation of an in-flight gRPC call."""
        call = state.rpc_call
        cancel_label = (
            "stitched gRPC call"
            if state.mode == NvidiaTTSSynthesisMode.STITCHED
            else "per-sentence gRPC call"
        )
        if call is not None and hasattr(call, "cancel"):
            try:
                call.cancel()
            except Exception as e:
                logger.debug(f"{self}: failed to cancel {cancel_label}: {e}")

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

    async def _process_stitched_responses(self, state: _SynthesisState):
        """Consume gRPC responses and append audio to the active audio context."""
        while True:
            item = await state.response_queue.get()
            if item is _REQUEST_DONE:
                continue
            if item is None:
                if self.audio_context_available(state.context_id):
                    await self.remove_audio_context(state.context_id)
                break
            if isinstance(item, Exception):
                # Treat stream exceptions as terminal for this stream. Once
                # SynthesizeOnline raises, no further reliable audio is expected.
                # Ignore stale or interruption-driven exceptions to avoid noisy
                # errors during handoff to a new stream.
                if self._synthesis_state is state and not state.stop_event.is_set():
                    await self.push_error(f"{self} synthesis error: {item}")
                    if self.audio_context_available(state.context_id):
                        await self.remove_audio_context(state.context_id)
                    self._synthesis_state = None
                break

            # Stale stream responses must never leak into a newer stream context.
            if self._synthesis_state is not state:
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
        if self._synthesis_state is state and not state.stop_event.is_set():
            self._synthesis_state = None

    async def _abort_all_synthesis(self):
        """Abort any active synthesis work."""
        state = self._synthesis_state
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
        state.text_queue.put(_CLOSE_STREAM)

        self._cancel_rpc_call(state)

        if state.synth_task is not None:
            await self.cancel_task(state.synth_task)
            state.synth_task = None

        if self._synthesis_state is state:
            self._synthesis_state = None

    async def _ensure_audio_context_started(self, context_id: str) -> TTSStartedFrame | None:
        """Create the audio context if needed and start TTFB metrics once."""
        if self.audio_context_available(context_id):
            return None

        await self.create_audio_context(context_id)
        await self.start_ttfb_metrics()
        return TTSStartedFrame(context_id=context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio and finalize the current context.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        state = self._synthesis_state
        if state is not None:
            state.text_queue.put(_CLOSE_STREAM)
        await super().flush_audio(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the active gRPC synthesis stream when the bot is interrupted.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        await self._abort_all_synthesis()
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

        Uses the configured synthesis mode to either yield audio directly
        (``per_sentence``) or queue text for asynchronous delivery
        (``stitched``).

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            ``TTSAudioRawFrame`` objects in ``per_sentence`` mode, ``None`` in
            ``stitched`` mode once text has been queued for asynchronous
            delivery, or ``ErrorFrame`` on failure.
        """
        text = text.strip()
        if not text or not any(c.isalnum() for c in text):
            return

        try:
            assert self._service is not None, "TTS service not initialized"
            if self._settings.synthesis_mode == NvidiaTTSSynthesisMode.STITCHED:
                async for frame in self._run_tts_stitched(text, context_id):
                    yield frame
            else:
                async for frame in self._run_tts_per_sentence(text, context_id):
                    yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")

    async def _run_tts_stitched(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        """Send text through one turn-scoped stitched synthesis stream."""
        if start_frame := await self._ensure_audio_context_started(context_id):
            yield start_frame

        logger.debug(f"{self}: Generating TTS [{text}]")
        state = await self._ensure_synthesis_state(context_id, NvidiaTTSSynthesisMode.STITCHED)

        for chunk in self._split_text_into_chunks(text):
            if any(c.isalnum() for c in chunk):
                state.text_queue.put(chunk)

        await self.start_tts_usage_metrics(text)
        yield None

    async def _run_tts_per_sentence(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        """Open one synthesis request per queued chunk and yield audio directly."""
        if start_frame := await self._ensure_audio_context_started(context_id):
            yield start_frame

        logger.debug(f"{self}: Generating TTS [{text}]")

        chunks = [
            chunk for chunk in self._split_text_into_chunks(text) if any(c.isalnum() for c in chunk)
        ]
        if not chunks:
            return

        state = await self._ensure_synthesis_state(
            context_id, NvidiaTTSSynthesisMode.PER_SENTENCE
        )
        request_count = 0
        for chunk in chunks:
            state.text_queue.put(chunk)
            state.text_queue.put(_RECONNECT)
            request_count += 1

        try:
            await self.start_tts_usage_metrics(text)
            completed_requests = 0
            while completed_requests < request_count:
                item = await state.response_queue.get()
                if item is _REQUEST_DONE:
                    completed_requests += 1
                    continue
                if item is None:
                    if state.stop_event.is_set():
                        break
                    raise RuntimeError("Synthesis stream closed unexpectedly")
                if self._synthesis_state is not state:
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
            if self._synthesis_state is state and state.mode == NvidiaTTSSynthesisMode.PER_SENTENCE:
                await self._abort_all_synthesis()
