#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Nemotron Speech text-to-speech service implementation.

This module provides integration with NVIDIA Nemotron Speech's TTS services through
gRPC API for high-quality speech synthesis.

Refer to the NVIDIA TTS NIM documentation for usage, customization,
and local deployment steps:
https://docs.nvidia.com/nim/speech/latest/tts/

For zero-shot voice cloning, request access to the Magpie TTS Zero-Shot model:
https://developer.nvidia.com/riva-tts-zeroshot-models

Local or private cloud deployment is recommended for best zero-shot performance.
"""

import asyncio
import os
import queue
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
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
        zero_shot_audio_prompt_file: Optional[Path] = None,
        audio_prompt_encoding: Optional[AudioEncoding] = AudioEncoding.ENCODING_UNSPECIFIED,
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
            zero_shot_audio_prompt_file: Path to audio prompt file for zero-shot voice
                cloning. Audio length should be between 3-10 seconds. The file
                is read once at init time and cached in memory. Requires the
                Magpie TTS Zero-Shot model. See
                https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html
            audio_prompt_encoding: Encoding of the zero-shot audio prompt file.
                Defaults to ``AudioEncoding.ENCODING_UNSPECIFIED``.
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
        self._audio_prompt_encoding = audio_prompt_encoding

        self._zero_shot_audio_prompt_file = zero_shot_audio_prompt_file
        self._zero_shot_audio_prompt: Optional[bytes] = None
        if self._zero_shot_audio_prompt_file is not None:
            if not self._zero_shot_audio_prompt_file.exists():
                raise FileNotFoundError(
                    f"Zero-shot audio prompt file not found: {self._zero_shot_audio_prompt_file}"
                )
            with self._zero_shot_audio_prompt_file.open("rb") as f:
                self._zero_shot_audio_prompt = f.read()
            logger.debug(
                f"Loaded zero-shot audio prompt from {self._zero_shot_audio_prompt_file} "
                f"({len(self._zero_shot_audio_prompt)} bytes)"
            )

        self._service = None
        self._config = None

        # Persistent gRPC stream state for cross-sentence stitching
        self._text_queue: Optional[queue.Queue] = None
        self._synth_thread: Optional[threading.Thread] = None
        self._response_task: Optional[asyncio.Task] = None
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._active_context_id: Optional[str] = None

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
        """Stop the NVIDIA Nemotron Speech TTS service and clean up resources.

        Args:
            frame: EndFrame indicating pipeline stop.
        """
        await super().stop(frame)
        await self._close_synthesis_stream()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA Nemotron Speech TTS service operation.

        Args:
            frame: CancelFrame indicating operation cancellation.
        """
        await super().cancel(frame)
        await self._close_synthesis_stream()

    def _start_synthesis_stream(self, context_id: str):
        """Start a persistent gRPC synthesis stream for the current turn.

        Creates a queue-backed generator that feeds text to
        ``synthesize_online``. The gRPC stream stays open until a ``None``
        sentinel is pushed into the queue.
        """
        self._text_queue = queue.Queue()
        self._active_context_id = context_id
        self._response_queue = asyncio.Queue()

        self._synth_thread = threading.Thread(
            target=self._synthesis_thread_handler,
            daemon=True,
            name="nvidia-tts-synth",
        )
        self._synth_thread.start()
        self._response_task = self.create_task(
            self._response_consumer(), name="nvidia-tts-response"
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
        if self._zero_shot_audio_prompt is not None:
            req.zero_shot_data.audio_prompt = self._zero_shot_audio_prompt
            req.zero_shot_data.encoding = self._audio_prompt_encoding
            req.zero_shot_data.quality = self._settings.quality
        if self._custom_dictionary:
            req.custom_dictionary = self._custom_dictionary
        return req

    def _synthesis_thread_handler(self):
        """Run ``SynthesizeOnline`` gRPC stream in a background thread.

        Builds request objects directly to avoid a Python 3.12 compatibility
        bug in ``riva.client.SpeechSynthesisService.synthesize_online``.
        """
        base_req = self._build_base_request()

        def request_generator():
            while True:
                text = self._text_queue.get()
                if text is None:
                    break
                base_req.text = text
                yield base_req

        try:
            responses = self._service.stub.SynthesizeOnline(
                request_generator(),
                metadata=self._service.auth.get_auth_metadata(),
            )
            for resp in responses:
                asyncio.run_coroutine_threadsafe(
                    self._response_queue.put(resp), self.get_event_loop()
                )
        except Exception as e:
            logger.error(f"{self} gRPC synthesis stream error: {e}")
            asyncio.run_coroutine_threadsafe(self._response_queue.put(e), self.get_event_loop())
        finally:
            asyncio.run_coroutine_threadsafe(self._response_queue.put(None), self.get_event_loop())

    async def _response_consumer(self):
        """Consume gRPC responses and append audio to the active audio context."""
        while True:
            item = await self._response_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                await self.push_error(f"{self} synthesis error: {item}")
                break
            await self.stop_ttfb_metrics()
            frame = TTSAudioRawFrame(
                audio=item.audio,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=self._active_context_id,
            )
            await self.append_to_audio_context(self._active_context_id, frame)

    async def _close_synthesis_stream(self):
        """Close the active gRPC synthesis stream.

        Sends a sentinel to end the request generator, waits for the gRPC
        thread to finish producing all remaining audio, then lets the
        response consumer drain naturally before cleaning up.
        """
        if self._text_queue is not None:
            self._text_queue.put(None)

        if self._synth_thread is not None:
            await asyncio.to_thread(self._synth_thread.join)
            self._synth_thread = None

        self._text_queue = None

        if self._response_task is not None:
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
            self._response_task = None

        self._active_context_id = None

    async def flush_audio(self, context_id: Optional[str] = None):
        """Flush the synthesis stream at the end of an LLM turn.

        Sends a sentinel to the gRPC stream, waits for remaining audio,
        then delegates to the base class for audio context cleanup.

        Args:
            context_id: The audio context to flush.
        """
        await self._close_synthesis_stream()
        await super().flush_audio(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Handle interruption by closing the active synthesis stream."""
        await self.stop_all_metrics()
        await self._close_synthesis_stream()

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

            for chunk in self._split_text_into_chunks(text):
                if any(c.isalnum() for c in chunk):
                    self._text_queue.put(chunk)

            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")