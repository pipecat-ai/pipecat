#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Vertex AI Text-to-Speech service implementations.

This module provides two concrete TTS services via Vertex AI:

- ``VertexTTSService`` — streaming synthesis for Chirp 3 HD / Journey voices,
  using ``google.cloud.texttospeech_v1`` routed to the regional Cloud TTS endpoint
  ``{location}-texttospeech.googleapis.com``.

- ``VertexGeminiTTSService`` — Gemini TTS models (e.g. gemini-2.5-flash-preview-tts)
  routed through ``{location}-aiplatform.googleapis.com`` via
  ``google.genai.Client(vertexai=True)``.  This is the BAA-compliant path for
  healthcare workloads covered under a Vertex AI agreement.

Both services require a GCP project ID, a region, and service-account credentials
(JSON string, file path, or ADC).
"""

import json
import os

from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress gRPC fork warnings on forked processes
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.google.tts import (
    language_to_gemini_tts_language,
    language_to_google_tts_language,
)
from pipecat.services.settings import (
    NOT_GIVEN,
    TTSSettings,
    _NotGiven,
    is_given,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    from google.api_core.client_options import ClientOptions
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request
    from google.cloud import texttospeech_v1
    from google.genai import Client as GenAIClient
    from google.genai import types as genai_types
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google Vertex AI TTS, you need to `pip install pipecat-ai[google]`. "
        "Also ensure Google Cloud credentials are configured."
    )
    raise Exception(f"Missing module: {e}")


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VertexTTSSettings(TTSSettings):
    """Settings for VertexTTSService.

    Parameters:
        speaking_rate: The speaking rate, in the range [0.25, 2.0].
    """

    speaking_rate: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class VertexGeminiTTSSettings(TTSSettings):
    """Settings for VertexGeminiTTSService.

    Parameters:
        prompt: Optional style instructions forwarded as the system instruction.
        multi_speaker: Whether to enable multi-speaker synthesis.
        speaker_configs: Speaker definitions for multi-speaker mode.
            Each entry: ``{"speaker_alias": str, "speaker_id": str}``.
    """

    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    multi_speaker: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_configs: list[dict[str, Any]] | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


# ---------------------------------------------------------------------------
# Base class for Cloud-TTS-backed services (VertexTTSService)
# ---------------------------------------------------------------------------


class GoogleVertexBaseTTSService(TTSService):
    """Base class for Vertex AI TTS services backed by google.cloud.texttospeech_v1.

    Handles authentication and shared streaming synthesis logic for
    Chirp / Journey voice services.  Not used by VertexGeminiTTSService,
    which uses the aiplatform endpoint instead.
    """

    def __init__(
        self,
        *,
        credentials: str | None = None,
        credentials_path: str | None = None,
        project_id: str,
        location: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._project_id = project_id
        self._location = location
        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    def _create_client(
        self, credentials: str | None, credentials_path: str | None
    ) -> texttospeech_v1.TextToSpeechAsyncClient:
        creds = self._get_credentials(credentials, credentials_path)
        client_options = ClientOptions(
            api_endpoint=f"{self._location}-texttospeech.googleapis.com"
        )
        return texttospeech_v1.TextToSpeechAsyncClient(
            credentials=creds, client_options=client_options
        )

    @staticmethod
    def _get_credentials(
        credentials: str | None, credentials_path: str | None
    ) -> service_account.Credentials:
        creds: service_account.Credentials | None = None

        if credentials:
            creds = service_account.Credentials.from_service_account_info(
                json.loads(credentials),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            try:
                creds, _ = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except GoogleAuthError:
                pass

        if not creds:
            raise ValueError("No valid credentials provided.")

        creds.refresh(Request())
        return creds

    def can_generate_metrics(self) -> bool:
        return True

    async def _stream_tts(
        self,
        streaming_config: texttospeech_v1.StreamingSynthesizeConfig,
        text: str,
        context_id: str,
        prompt: str | None = None,
    ) -> AsyncGenerator[Frame, None]:
        config_request = texttospeech_v1.StreamingSynthesizeRequest(
            streaming_config=streaming_config
        )

        async def request_generator():
            yield config_request
            synthesis_input_params = {"text": text}
            if prompt is not None:
                synthesis_input_params["prompt"] = prompt
            yield texttospeech_v1.StreamingSynthesizeRequest(
                input=texttospeech_v1.StreamingSynthesisInput(**synthesis_input_params)
            )

        streaming_responses = await self._client.streaming_synthesize(request_generator())
        await self.start_tts_usage_metrics(text)

        audio_buffer = b""
        first_chunk_seen = False
        CHUNK_SIZE = self.chunk_size

        async for response in streaming_responses:
            chunk = response.audio_content
            if not chunk:
                continue

            if not first_chunk_seen:
                await self.stop_ttfb_metrics()
                first_chunk_seen = True

            audio_buffer += chunk
            while len(audio_buffer) >= CHUNK_SIZE:
                piece = audio_buffer[:CHUNK_SIZE]
                audio_buffer = audio_buffer[CHUNK_SIZE:]
                yield TTSAudioRawFrame(piece, self.sample_rate, 1, context_id=context_id)

        if audio_buffer:
            yield TTSAudioRawFrame(audio_buffer, self.sample_rate, 1, context_id=context_id)


# ---------------------------------------------------------------------------
# VertexTTSService — Chirp / Journey voices via regional Cloud TTS endpoint
# ---------------------------------------------------------------------------


class VertexTTSService(GoogleVertexBaseTTSService):
    """Google Vertex AI Text-to-Speech streaming service for Chirp and Journey voices.

    Synthesises audio via ``{location}-texttospeech.googleapis.com`` using the
    Cloud TTS streaming API.  Suitable for Chirp 3 HD and Journey voices.

    For Gemini TTS models (BAA-compliant aiplatform path) use
    ``VertexGeminiTTSService`` instead.

    Example::

        tts = VertexTTSService(
            credentials_path="/path/to/service-account.json",
            project_id="my-gcp-project",
            location="us-central1",
            settings=VertexTTSService.Settings(
                voice="en-US-Chirp3-HD-Charon",
                language=Language.EN_US,
            )
        )
    """

    Settings = VertexTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Vertex TTS configuration.

        .. deprecated:: 0.0.105
            Use ``VertexTTSService.Settings`` directly via the ``settings`` parameter instead.
        """

        language: Language | None = Language.EN
        speaking_rate: float | None = None

    def __init__(
        self,
        *,
        credentials: str | None = None,
        credentials_path: str | None = None,
        project_id: str,
        location: str,
        voice_id: str | None = None,
        voice_cloning_key: str | None = None,
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        # 1. Hardcoded defaults
        default_settings = self.Settings(
            model=None,
            voice="en-US-Chirp3-HD-Charon",
            language="en-US",
            speaking_rate=None,
        )

        # 2. Deprecated direct arg overrides
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Deprecated params overrides (only if settings not provided)
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.speaking_rate is not None:
                    default_settings.speaking_rate = params.speaking_rate

        # 4. Settings delta (always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        self._voice_cloning_key = voice_cloning_key

        super().__init__(
            credentials=credentials,
            credentials_path=credentials_path,
            project_id=project_id,
            location=location,
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_google_tts_language(language)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        if isinstance(delta, self.Settings) and is_given(delta.speaking_rate):
            rate_value = float(delta.speaking_rate)
            if not (0.25 <= rate_value <= 2.0):
                logger.warning(
                    f"Invalid speaking_rate value: {rate_value}. Must be between 0.25 and 2.0"
                )
                delta.speaking_rate = NOT_GIVEN
        return await super()._update_settings(delta)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if self._voice_cloning_key:
                voice_clone_params = texttospeech_v1.VoiceCloneParams(
                    voice_cloning_key=self._voice_cloning_key
                )
                voice = texttospeech_v1.VoiceSelectionParams(
                    language_code=self._settings.language, voice_clone=voice_clone_params
                )
            else:
                voice = texttospeech_v1.VoiceSelectionParams(
                    language_code=self._settings.language, name=self._settings.voice
                )

            streaming_config = texttospeech_v1.StreamingSynthesizeConfig(
                voice=voice,
                streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                    audio_encoding=texttospeech_v1.AudioEncoding.PCM,
                    sample_rate_hertz=self.sample_rate,
                    speaking_rate=self._settings.speaking_rate,
                ),
            )

            async for frame in self._stream_tts(streaming_config, text, context_id):
                yield frame

        except Exception as e:
            await self.push_error(error_msg=f"Vertex TTS generation error: {str(e)}", exception=e)


# ---------------------------------------------------------------------------
# VertexGeminiTTSService — Gemini TTS via aiplatform.googleapis.com (BAA path)
# ---------------------------------------------------------------------------


class VertexGeminiTTSService(TTSService):
    """Gemini Text-to-Speech via Vertex AI aiplatform endpoint.

    Routes all synthesis requests through
    ``{location}-aiplatform.googleapis.com`` using
    ``google.genai.Client(vertexai=True)``.  This is the BAA-compliant path
    for healthcare workloads covered under a Vertex AI agreement, unlike
    ``GeminiTTSService`` which uses the global Cloud TTS endpoint.

    Authentication follows the same pattern as ``GeminiLiveVertexLLMService``:
    service-account JSON string, file path, or Application Default Credentials,
    with an automatic token refresh before each client is created.

    Example::

        tts = VertexGeminiTTSService(
            credentials_path="/path/to/service-account.json",
            project_id="my-gcp-project",
            location="us-central1",
            settings=VertexGeminiTTSService.Settings(
                model="gemini-2.5-flash-preview-tts",
                voice="Kore",
                language=Language.EN_US,
            )
        )
    """

    Settings = VertexGeminiTTSSettings
    _settings: Settings

    GOOGLE_SAMPLE_RATE = 24000

    AVAILABLE_VOICES = [
        "Achernar", "Achird", "Algenib", "Algieba", "Alnilam",
        "Aoede", "Autonoe", "Callirhoe", "Charon", "Despina",
        "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus",
        "Kore", "Laomedeia", "Leda", "Orus", "Puck",
        "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar",
        "Sulafar", "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi",
    ]

    class InputParams(BaseModel):
        """Input parameters for Vertex Gemini TTS configuration.

        .. deprecated:: 0.0.105
            Use ``VertexGeminiTTSService.Settings`` directly via the ``settings`` parameter instead.
        """

        language: Language | None = Language.EN
        prompt: str | None = None
        multi_speaker: bool = False
        speaker_configs: list[dict] | None = None

    def __init__(
        self,
        *,
        credentials: str | None = None,
        credentials_path: str | None = None,
        project_id: str,
        location: str,
        model: str | None = None,
        voice_id: str | None = None,
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialise the Vertex AI Gemini TTS service.

        Args:
            credentials: JSON string of a Google Cloud service account.
            credentials_path: Path to a service-account JSON file.
            project_id: GCP project ID (required for Vertex AI).
            location: GCP region, e.g. ``"us-central1"``.
            model: Gemini TTS model name.

                .. deprecated:: 0.0.105
                    Use ``settings=VertexGeminiTTSService.Settings(model=...)`` instead.

            voice_id: Voice name from ``AVAILABLE_VOICES``.

                .. deprecated:: 0.0.105
                    Use ``settings=VertexGeminiTTSService.Settings(voice=...)`` instead.

            sample_rate: Output sample rate in Hz. Gemini TTS outputs 24 kHz;
                passing a different value will trigger a warning.
            params: Deprecated configuration object.

                .. deprecated:: 0.0.105
                    Use ``settings=VertexGeminiTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings delta.
            **kwargs: Forwarded to ``TTSService``.
        """
        if sample_rate and sample_rate != self.GOOGLE_SAMPLE_RATE:
            logger.warning(
                f"VertexGeminiTTSService: Gemini TTS outputs at {self.GOOGLE_SAMPLE_RATE} Hz. "
                f"Requested sample_rate={sample_rate} Hz may cause audio distortion."
            )

        # 1. Hardcoded defaults
        default_settings = self.Settings(
            model="gemini-2.5-flash-preview-tts",
            voice="Kore",
            language="en-US",
            prompt=None,
            multi_speaker=False,
            speaker_configs=None,
        )

        # 2. Deprecated direct arg overrides
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        if default_settings.voice not in self.AVAILABLE_VOICES:
            logger.warning(
                f"VertexGeminiTTSService: voice '{default_settings.voice}' not in known list — "
                "using anyway, but verify it is available on Vertex AI."
            )

        # 3. Deprecated params overrides (only if settings not provided)
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.prompt is not None:
                    default_settings.prompt = params.prompt
                if params.multi_speaker is not None:
                    default_settings.multi_speaker = params.multi_speaker
                if params.speaker_configs is not None:
                    default_settings.speaker_configs = params.speaker_configs

        # 4. Settings delta (always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Store Vertex AI coordinates before creating the client, as _create_genai_client needs them.
        self._project_id = project_id
        self._location = location
        self._genai_client: GenAIClient = self._create_genai_client(credentials, credentials_path)

        logger.info(
            f"VertexGeminiTTSService: initialised — project={project_id}, location={location}, "
            f"model={default_settings.model}, voice={default_settings.voice}, "
            f"language={default_settings.language}"
        )

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Auth + client
    # ------------------------------------------------------------------

    def _create_genai_client(
        self, credentials: str | None, credentials_path: str | None
    ) -> GenAIClient:
        """Create a google.genai Client routed to the Vertex AI aiplatform endpoint."""
        creds = self._get_credentials(credentials, credentials_path)
        logger.debug(
            f"VertexGeminiTTSService: creating GenAI client — "
            f"project={self._project_id}, location={self._location}"
        )
        return GenAIClient(
            vertexai=True,
            credentials=creds,
            project=self._project_id,
            location=self._location,
        )

    @staticmethod
    def _get_credentials(
        credentials: str | None, credentials_path: str | None
    ) -> service_account.Credentials:
        """Load and refresh Google service-account credentials.

        Supports JSON string, file path, and Application Default Credentials.

        Raises:
            ValueError: If no credentials are found via any method.
        """
        creds: service_account.Credentials | None = None

        if credentials:
            logger.debug("VertexGeminiTTSService: loading credentials from JSON string")
            creds = service_account.Credentials.from_service_account_info(
                json.loads(credentials),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif credentials_path:
            logger.debug(f"VertexGeminiTTSService: loading credentials from file: {credentials_path}")
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            logger.debug("VertexGeminiTTSService: attempting Application Default Credentials")
            try:
                creds, _ = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except GoogleAuthError as auth_err:
                logger.warning(f"VertexGeminiTTSService: ADC failed: {auth_err}")

        if not creds:
            raise ValueError(
                "VertexGeminiTTSService: no valid credentials found. "
                "Provide credentials=, credentials_path=, or set GOOGLE_APPLICATION_CREDENTIALS."
            )

        logger.debug("VertexGeminiTTSService: refreshing credentials token")
        creds.refresh(Request())
        return creds

    # ------------------------------------------------------------------
    # TTSService interface
    # ------------------------------------------------------------------

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_gemini_tts_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self.sample_rate != self.GOOGLE_SAMPLE_RATE:
            logger.warning(
                f"VertexGeminiTTSService: pipeline sample_rate={self.sample_rate} Hz differs from "
                f"Gemini TTS output rate={self.GOOGLE_SAMPLE_RATE} Hz — audio may be distorted."
            )
        logger.info(
            f"VertexGeminiTTSService: started — endpoint=https://{self._location}-aiplatform.googleapis.com"
        )

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        if is_given(delta.voice) and delta.voice not in self.AVAILABLE_VOICES:
            logger.warning(
                f"VertexGeminiTTSService: voice '{delta.voice}' not in known list — "
                "using anyway."
            )
        return await super()._update_settings(delta)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesise text via Vertex AI aiplatform using google.genai streaming.

        Args:
            text: Text to synthesise. Supports Gemini expressive markup
                (e.g. ``[sigh]``, ``[laughing]``).
            context_id: Frame context identifier for downstream routing.

        Yields:
            TTSAudioRawFrame: PCM audio chunks at ``self.sample_rate`` Hz, mono.
        """
        logger.debug(
            f"VertexGeminiTTSService: run_tts called — "
            f"model={self._settings.model}, voice={self._settings.voice}, "
            f"language={self._settings.language}, text_len={len(text)}, text=[{text}]"
        )

        try:
            # Build speech config — single-speaker or multi-speaker
            if self._settings.multi_speaker and self._settings.speaker_configs:
                logger.debug(
                    f"VertexGeminiTTSService: multi-speaker mode — "
                    f"{len(self._settings.speaker_configs)} speakers"
                )
                speaker_voice_configs = [
                    genai_types.SpeakerVoiceConfig(
                        speaker=cfg["speaker_alias"],
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=cfg.get("speaker_id", self._settings.voice)
                            )
                        ),
                    )
                    for cfg in self._settings.speaker_configs
                ]
                speech_config = genai_types.SpeechConfig(
                    multi_speaker_voice_config=genai_types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=speaker_voice_configs
                    )
                )
            else:
                logger.debug(
                    f"VertexGeminiTTSService: single-speaker mode — voice={self._settings.voice}"
                )
                speech_config = genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=self._settings.voice
                        )
                    )
                )

            genai_config = genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
                system_instruction=self._settings.prompt or None,
            )

            logger.debug(
                f"VertexGeminiTTSService: sending request to aiplatform — "
                f"model={self._settings.model}, has_prompt={bool(self._settings.prompt)}"
            )

            await self.start_tts_usage_metrics(text)

            audio_buffer = b""
            first_chunk_seen = False
            chunk_count = 0
            total_audio_bytes = 0
            CHUNK_SIZE = self.chunk_size

            async for response in self._genai_client.aio.models.stream_generate_content(
                model=self._settings.model,
                contents=text,
                config=genai_config,
            ):
                if not response.candidates:
                    logger.debug("VertexGeminiTTSService: received response with no candidates, skipping")
                    continue

                candidate = response.candidates[0]

                if not candidate.content or not candidate.content.parts:
                    if candidate.finish_reason:
                        logger.debug(
                            f"VertexGeminiTTSService: candidate finish_reason={candidate.finish_reason}"
                        )
                    continue

                for part in candidate.content.parts:
                    if not part.inline_data:
                        if part.text:
                            logger.debug(
                                f"VertexGeminiTTSService: received unexpected text part: [{part.text}]"
                            )
                        continue

                    audio_data: bytes = part.inline_data.data
                    if not audio_data:
                        logger.debug("VertexGeminiTTSService: received empty inline_data.data, skipping")
                        continue

                    chunk_count += 1
                    total_audio_bytes += len(audio_data)

                    if not first_chunk_seen:
                        logger.debug(
                            f"VertexGeminiTTSService: first audio chunk received — "
                            f"mime_type={part.inline_data.mime_type}, bytes={len(audio_data)}"
                        )
                        await self.stop_ttfb_metrics()
                        first_chunk_seen = True

                    audio_buffer += audio_data
                    while len(audio_buffer) >= CHUNK_SIZE:
                        piece = audio_buffer[:CHUNK_SIZE]
                        audio_buffer = audio_buffer[CHUNK_SIZE:]
                        yield TTSAudioRawFrame(piece, self.sample_rate, 1, context_id=context_id)

            # Flush remainder
            if audio_buffer:
                logger.debug(
                    f"VertexGeminiTTSService: flushing final {len(audio_buffer)} bytes"
                )
                yield TTSAudioRawFrame(audio_buffer, self.sample_rate, 1, context_id=context_id)

            if not first_chunk_seen:
                logger.warning(
                    f"VertexGeminiTTSService: no audio returned for text=[{text}] — "
                    "check model name availability on Vertex AI and region."
                )
            else:
                logger.debug(
                    f"VertexGeminiTTSService: synthesis complete — "
                    f"{chunk_count} API chunks, {total_audio_bytes} total bytes"
                )

        except Exception as e:
            logger.error(
                f"VertexGeminiTTSService: error synthesising [{text}]: {e}",
                exc_info=True,
            )
            await self.push_error(
                error_msg=f"Vertex Gemini TTS generation error: {str(e)}", exception=e
            )
