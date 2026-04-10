#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Nemotron Speech text-to-speech service implementation.

This module provides integration with NVIDIA Nemotron Speech's TTS services through
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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Mapping, Optional

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
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    import grpc
    import riva.client
    import riva.client.proto.riva_tts_pb2 as rtts
    from riva.client.proto.riva_audio_pb2 import AudioEncoding
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use NVIDIA Nemotron Speech TTS, you need to `pip install pipecat-ai[nvidia]`."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class NvidiaTTSSettings(TTSSettings):
    """Settings for NvidiaTTSService.

    Parameters:
        quality: Audio quality setting (0-100).
    """

    quality: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _SynthesisStreamState:
    """Runtime state for one active synthesis stream."""

    context_id: str
    text_queue: queue.Queue
    response_queue: asyncio.Queue
    stop_event: threading.Event
    rpc_call: Any = None
    synth_task: Optional[asyncio.Task] = None
    response_task: Optional[asyncio.Task] = None


class NvidiaTTSService(TTSService):
    """NVIDIA Nemotron Speech text-to-speech service.

    Provides high-quality text-to-speech synthesis using NVIDIA Nemotron Speech's
    cloud-based TTS models. Supports multiple voices, languages, and
    configurable quality settings.
    """

    Settings = NvidiaTTSSettings
    _settings: Settings
    _MAX_CHUNK_LEN = 200

    class InputParams(BaseModel):
        """Input parameters for Nemotron Speech TTS configuration.

        .. deprecated:: 0.0.105
            Use ``NvidiaTTSService.Settings`` directly via the ``settings`` parameter instead.

        Parameters:
            language: Language code for synthesis. Defaults to US English.
            quality: Audio quality setting (0-100). Defaults to 20.
        """

        language: Optional[Language] = Language.EN_US
        quality: Optional[int] = 20

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        model_function_map: Mapping[str, str] = {
            "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
            "model_name": "magpie-tts-multilingual",
        },
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        use_ssl: bool = True,
        custom_dictionary: Optional[dict] = None,
        encoding: Optional[AudioEncoding] = AudioEncoding.LINEAR_PCM,
        **kwargs,
    ):
        """Initialize the NVIDIA Nemotron Speech TTS service.

        Args:
            api_key: NVIDIA API key for authentication. Required when using the
                cloud endpoint. Not needed for local deployments.
            server: gRPC server endpoint. Defaults to NVIDIA's cloud endpoint.
                For local deployments, pass the local address (e.g. ``localhost:50051``).
            voice_id: Voice model identifier. Defaults to multilingual Aria voice.

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaTTSService.Settings(voice=...)`` instead.

            sample_rate: Audio sample rate. If None, uses service default.
            model_function_map: Dictionary containing function_id and model_name for the TTS model.
            params: Additional configuration parameters for TTS synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            use_ssl: Whether to use SSL for the gRPC connection. Defaults to True
                for the NVIDIA cloud endpoint. Set to False for local deployments.
            custom_dictionary: Custom pronunciation dictionary mapping words
                (graphemes) to IPA phonetic representations (phonemes),
                e.g. ``{"NVIDIA": "ɛn.vɪ.diː.ʌ"}``. See
                https://docs.nvidia.com/nim/speech/latest/tts/phoneme-support.html
                for the list of supported IPA phonemes.
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

        self._custom_dictionary: Optional[str] = None
        if custom_dictionary:
            entries = [f"{k}  {v}" for k, v in custom_dictionary.items()]
            self._custom_dictionary = ",".join(entries)
        self._encoding = encoding

        self._service = None
        self._config = None

        # Runtime state for the active streaming turn.
        self._stream_state: Optional[_SynthesisStreamState] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True as this service supports metric generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the TTS model.

        .. deprecated:: 0.0.104
            Model cannot be changed after initialization for NVIDIA Nemotron Speech TTS.
            Set model and function id in the constructor instead.

            Example::

                NvidiaTTSService(
                    api_key=...,
                    model_function_map={"function_id": "<UUID>", "model_name": "<model_name>"},
                )

        Args:
            model: The model name to set.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'set_model' is deprecated. Model cannot be changed after initialization"
                " for NVIDIA Nemotron Speech TTS. Set model and function id in the constructor"
                " instead, e.g.: NvidiaTTSService(api_key=..., model_function_map="
                "{'function_id': '<UUID>', 'model_name': '<model_name>'})",
                DeprecationWarning,
                stacklevel=2,
            )

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

        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)

        self._service = riva.client.SpeechSynthesisService(auth)

    def _create_synthesis_config(self):
        if not self._service:
            return

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
            return None

    async def start(self, frame: StartFrame):
        """Start the NVIDIA Nemotron Speech TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_synthesis_config()
        logger.debug(f"Initialized NvidiaTTSService with model: {self._settings.model}")

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA Nemotron Speech TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_synthesis_stream()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA Nemotron Speech TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_synthesis_stream()

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

    async def _process_responses(self, state: _SynthesisStreamState):
        """Consume gRPC responses and append audio to the active audio context."""
        while True:
            item = await state.response_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                # Ignore stale exceptions from interrupted streams.
                if self._stream_state is state:
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

    async def _close_synthesis_stream(self):
        """Close the active gRPC synthesis stream gracefully.

        Sends a sentinel to end the request generator, waits for the
        synthesis task to finish producing all remaining audio, then lets
        the response task drain naturally before cleaning up.
        """
        state = self._stream_state
        if state is None:
            return

        self._signal_synthesis_close(state)

        if state.synth_task is not None:
            try:
                await state.synth_task
            except asyncio.CancelledError:
                pass
            state.synth_task = None

        if state.response_task is not None:
            try:
                await state.response_task
            except asyncio.CancelledError:
                pass
            state.response_task = None

        if self._stream_state is state:
            self._stream_state = None

    async def _wait_for_synthesis_close_interruptibly(self, state: _SynthesisStreamState):
        """Wait for synthesis close unless interruption preempts this stream."""
        while True:
            if self._stream_state is not state or state.stop_event.is_set():
                # Interruption took ownership of stream shutdown.
                return

            synth_done = state.synth_task is None or state.synth_task.done()
            response_done = state.response_task is None or state.response_task.done()
            if synth_done and response_done:
                break

            # Poll in short intervals to keep this wait interruptible.
            await asyncio.sleep(0.05)

        if state.synth_task is not None:
            try:
                await state.synth_task
            except asyncio.CancelledError:
                pass
            state.synth_task = None

        if state.response_task is not None:
            try:
                await state.response_task
            except asyncio.CancelledError:
                pass
            state.response_task = None

        if self._stream_state is state:
            self._stream_state = None

    async def _abort_synthesis_stream(self):
        """Abort the active gRPC synthesis stream immediately.

        Cancels the response task first to stop delivering audio, then
        drains the text queue and signals the synthesis handler to stop.
        Unlike ``_close_synthesis_stream``, pending audio is discarded.
        """
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

    async def flush_audio(self, context_id: Optional[str] = None):
        """Flush any pending audio and finalize the current context.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        state = self._stream_state
        if state is not None:
            self._signal_synthesis_close(state)
            await self._wait_for_synthesis_close_interruptibly(state)
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
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NVIDIA Nemotron Speech TTS.

        On the first call for a turn, starts a persistent ``synthesize_online``
        gRPC stream. Subsequent calls within the same turn feed text into the
        existing stream, enabling Magpie's cross-sentence stitching.

        Text is split into chunks respecting Magpie's per-request limits. Each chunk becomes
        a separate request in the gRPC stream, stitched seamlessly by Magpie.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            None on success. Audio is delivered asynchronously via the
                response consumer. ErrorFrame on failure.
        """
        text = text.strip()
        if not text or not any(c.isalnum() for c in text):
            return

        try:
            assert self._service is not None, "TTS service not initialized"
            assert self._config is not None, "Synthesis configuration not created"

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