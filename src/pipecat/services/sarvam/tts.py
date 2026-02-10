#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI text-to-speech service implementation.

This module provides TTS services using Sarvam AI's API with support for multiple
Indian languages and two model variants:

**Model Variants:**

- **bulbul:v2** (default): Standard TTS model
    - Supports: pitch, loudness, pace (0.3-3.0)
    - Default sample rate: 22050 Hz
    - Speakers: anushka (default), abhilash, manisha, vidya, arya, karun, hitesh

- **bulbul:v3-beta**: Advanced TTS model with temperature control
    - Does NOT support: pitch, loudness
    - Supports: pace (0.5-2.0), temperature (0.01-1.0)
    - Default sample rate: 24000 Hz
    - Preprocessing is always enabled
    - Speakers: aditya (default), ritu, priya, neha, rahul, pooja, rohan, simran,
      kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir,
      aayan, shubh, ashutosh, advait, amelia, sophia

- **bulbul:v3**: Advanced TTS model with temperature control
    - Does NOT support: pitch, loudness
    - Supports: pace (0.5-2.0), temperature (0.01-1.0)
    - Default sample rate: 24000 Hz
    - Preprocessing is always enabled
    - Speakers: aditya (default), ritu, priya, neha, rahul, pooja, rohan, simran,
      kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir,
      aayan, shubh, ashutosh, advait, amelia, sophia

See https://docs.sarvam.ai/api-reference-docs/text-to-speech/stream for full API details.
"""

import asyncio
import base64
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Tuple

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam, you need to `pip install pipecat-ai[sarvam]`.")
    raise Exception(f"Missing module: {e}")


class SarvamTTSModel(str, Enum):
    """Available Sarvam TTS models.

    Attributes:
        BULBUL_V2: Standard TTS model with pitch/loudness control.
            - Supports pitch, loudness, pace (0.3-3.0)
            - Default sample rate: 22050 Hz
        BULBUL_V3_BETA: Advanced model with temperature control.
            - Does NOT support pitch/loudness
            - Pace range: 0.5-2.0
            - Supports temperature parameter
            - Default sample rate: 24000 Hz
            - Preprocessing is always enabled
    """

    BULBUL_V2 = "bulbul:v2"
    BULBUL_V3_BETA = "bulbul:v3-beta"
    BULBUL_V3 = "bulbul:v3"


class SarvamTTSSpeakerV2(str, Enum):
    """Available speakers for bulbul:v2 model.

    Female voices: anushka, manisha, vidya, arya
    Male voices: abhilash, karun, hitesh
    """

    ANUSHKA = "anushka"
    ABHILASH = "abhilash"
    MANISHA = "manisha"
    VIDYA = "vidya"
    ARYA = "arya"
    KARUN = "karun"
    HITESH = "hitesh"


class SarvamTTSSpeakerV3(str, Enum):
    """Available speakers for bulbul:v3-beta model.

    Includes a wider variety of voices with different characteristics.
    """

    ADITYA = "aditya"
    RITU = "ritu"
    PRIYA = "priya"
    NEHA = "neha"
    RAHUL = "rahul"
    POOJA = "pooja"
    ROHAN = "rohan"
    SIMRAN = "simran"
    KAVYA = "kavya"
    AMIT = "amit"
    DEV = "dev"
    ISHITA = "ishita"
    SHREYA = "shreya"
    RATAN = "ratan"
    VARUN = "varun"
    MANAN = "manan"
    SUMIT = "sumit"
    ROOPA = "roopa"
    KABIR = "kabir"
    AAYAN = "aayan"
    SHUBH = "shubh"
    ASHUTOSH = "ashutosh"
    ADVAIT = "advait"
    AMELIA = "amelia"
    SOPHIA = "sophia"


@dataclass(frozen=True)
class TTSModelConfig:
    """Immutable configuration for a Sarvam TTS model.

    Attributes:
        supports_pitch: Whether the model accepts pitch parameter.
        supports_loudness: Whether the model accepts loudness parameter.
        supports_temperature: Whether the model accepts temperature parameter.
        default_sample_rate: Default audio sample rate in Hz.
        default_speaker: Default speaker voice ID.
        pace_range: Valid range for pace parameter (min, max).
        preprocessing_always_enabled: Whether preprocessing is always enabled.
        speakers: Tuple of available speaker names for this model.
    """

    supports_pitch: bool
    supports_loudness: bool
    supports_temperature: bool
    default_sample_rate: int
    default_speaker: str
    pace_range: Tuple[float, float]
    preprocessing_always_enabled: bool
    speakers: Tuple[str, ...]


TTS_MODEL_CONFIGS: Dict[str, TTSModelConfig] = {
    "bulbul:v2": TTSModelConfig(
        supports_pitch=True,
        supports_loudness=True,
        supports_temperature=False,
        default_sample_rate=22050,
        default_speaker="anushka",
        pace_range=(0.3, 3.0),
        preprocessing_always_enabled=False,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV2),
    ),
    "bulbul:v3-beta": TTSModelConfig(
        supports_pitch=False,
        supports_loudness=False,
        supports_temperature=True,
        default_sample_rate=24000,
        default_speaker="shubh",
        pace_range=(0.5, 2.0),
        preprocessing_always_enabled=True,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV3),
    ),
    "bulbul:v3": TTSModelConfig(
        supports_pitch=False,
        supports_loudness=False,
        supports_temperature=True,
        default_sample_rate=24000,
        default_speaker="shubh",
        pace_range=(0.5, 2.0),
        preprocessing_always_enabled=True,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV3),
    ),
}


def get_speakers_for_model(model: str) -> List[str]:
    """Get the list of available speakers for a given model.

    Args:
        model: The model name (e.g., "bulbul:v2" or "bulbul:v3-beta").

    Returns:
        List of speaker names available for the model.
    """
    if model in TTS_MODEL_CONFIGS:
        return list(TTS_MODEL_CONFIGS[model].speakers)
    # Default to v2 speakers for unknown models
    return list(TTS_MODEL_CONFIGS["bulbul:v2"].speakers)


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Sarvam AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Sarvam AI language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.BN: "bn-IN",  # Bengali
        Language.EN: "en-IN",  # English (India)
        Language.GU: "gu-IN",  # Gujarati
        Language.HI: "hi-IN",  # Hindi
        Language.KN: "kn-IN",  # Kannada
        Language.ML: "ml-IN",  # Malayalam
        Language.MR: "mr-IN",  # Marathi
        Language.OR: "od-IN",  # Odia
        Language.PA: "pa-IN",  # Punjabi
        Language.TA: "ta-IN",  # Tamil
        Language.TE: "te-IN",  # Telugu
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class SarvamHttpTTSService(TTSService):
    """Text-to-Speech service using Sarvam AI's API.

    Converts text to speech using Sarvam AI's TTS models with support for multiple
    Indian languages. Provides control over voice characteristics.

    **Model Differences:**

    - **bulbul:v2** (default):
        - Supports: pitch (-0.75 to 0.75), loudness (0.3 to 3.0), pace (0.3 to 3.0)
        - Default sample rate: 22050 Hz
        - Speakers: anushka, abhilash, manisha, vidya, arya, karun, hitesh

    - **bulbul:v3-beta**:
        - Does NOT support: pitch, loudness (will be ignored)
        - Supports: pace (0.5 to 2.0), temperature (0.01 to 1.0)
        - Default sample rate: 24000 Hz
        - Preprocessing is always enabled
        - Speakers: aditya, ritu, priya, neha, rahul, pooja, rohan, simran, kavya,
          amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir,
          aayan, shubh, ashutosh, advait, amelia, sophia

    Example::

        # Using bulbul:v2 (default)
        tts = SarvamHttpTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            aiohttp_session=session,
            params=SarvamHttpTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2,
                loudness=1.5
            )
        )

        # Using bulbul:v3-beta with temperature control
        tts_v3 = SarvamHttpTTSService(
            api_key="your-api-key",
            voice_id="aditya",  # Use v3 speaker
            model="bulbul:v3-beta",
            aiohttp_session=session,
            params=SarvamHttpTTSService.InputParams(
                language=Language.HI,
                pace=1.2,  # Range: 0.5-2.0 for v3
                temperature=0.8
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Sarvam TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English (India).
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            pace: Speech pace multiplier. Defaults to 1.0.
                - bulbul:v2: Range 0.3 to 3.0
                - bulbul:v3-beta: Range 0.5 to 2.0
            loudness: Volume multiplier (0.3 to 3.0). Defaults to 1.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            enable_preprocessing: Whether to enable text preprocessing. Defaults to False.
                **Note:** Always enabled for bulbul:v3-beta (cannot be disabled).
            temperature: Controls output randomness for bulbul:v3-beta (0.01 to 1.0).
                Lower values = more deterministic, higher = more random. Defaults to 0.6.
                **Note:** Only supported for bulbul:v3-beta. Ignored for v2.
        """

        language: Optional[Language] = Language.EN
        pitch: Optional[float] = Field(
            default=0.0,
            ge=-0.75,
            le=0.75,
            description="Voice pitch adjustment. Only for bulbul:v2.",
        )
        pace: Optional[float] = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Speech pace. v2: 0.3-3.0, v3: 0.5-2.0.",
        )
        loudness: Optional[float] = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Volume multiplier. Only for bulbul:v2.",
        )
        enable_preprocessing: Optional[bool] = Field(
            default=False,
            description="Enable text preprocessing. Always enabled for v3-beta model.",
        )
        temperature: Optional[float] = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Output randomness for bulbul:v3-beta only. Range: 0.01-1.0.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: Optional[str] = None,
        model: str = "bulbul:v2",
        base_url: str = "https://api.sarvam.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service.

        Args:
            api_key: Sarvam AI API subscription key.
            aiohttp_session: Shared aiohttp session for making requests.
            voice_id: Speaker voice ID. If None, uses model-appropriate default.
            model: TTS model to use. Options:
                - "bulbul:v2" (default): Standard model with pitch/loudness support
                - "bulbul:v3-beta": Advanced model with temperature control
            base_url: Sarvam AI API base URL. Defaults to "https://api.sarvam.ai".
            sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000).
                If None, uses model-specific default.
            params: Additional voice and preprocessing parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # Get model configuration (validates model exists)
        if model not in TTS_MODEL_CONFIGS:
            allowed = ", ".join(sorted(TTS_MODEL_CONFIGS.keys()))
            raise ValueError(f"Unsupported model '{model}'. Allowed values: {allowed}.")

        self._config = TTS_MODEL_CONFIGS[model]

        # Set default sample rate based on model if not specified
        if sample_rate is None:
            sample_rate = self._config.default_sample_rate

        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or SarvamHttpTTSService.InputParams()

        # Set default voice based on model if not specified
        if voice_id is None:
            voice_id = self._config.default_speaker

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        # Validate and clamp pace to model's valid range
        pace = params.pace
        pace_min, pace_max = self._config.pace_range
        if pace is not None and (pace < pace_min or pace > pace_max):
            logger.warning(f"Pace {pace} is outside model range ({pace_min}-{pace_max}). Clamping.")
            pace = max(pace_min, min(pace_max, pace))

        # Build base settings
        self._settings = {
            "language": (
                self.language_to_service_language(params.language) if params.language else "en-IN"
            ),
            "enable_preprocessing": (
                True if self._config.preprocessing_always_enabled else params.enable_preprocessing
            ),
            "pace": pace,
            "model": model,
        }

        # Add parameters based on model support
        if self._config.supports_pitch:
            self._settings["pitch"] = params.pitch
        elif params.pitch != 0.0:
            logger.warning(f"pitch parameter is ignored for {model}")

        if self._config.supports_loudness:
            self._settings["loudness"] = params.loudness
        elif params.loudness != 1.0:
            logger.warning(f"loudness parameter is ignored for {model}")

        if self._config.supports_temperature:
            self._settings["temperature"] = params.temperature
        elif params.temperature != 0.6:
            logger.warning(f"temperature parameter is ignored for {model}")

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Sarvam AI's API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Build payload with common parameters
            payload = {
                "text": text,
                "target_language_code": self._settings["language"],
                "speaker": self._voice_id,
                "sample_rate": self.sample_rate,
                "enable_preprocessing": self._settings["enable_preprocessing"],
                "model": self._model_name,
                "pace": self._settings.get("pace", 1.0),
            }

            # Add model-specific parameters based on config
            if self._config.supports_pitch:
                payload["pitch"] = self._settings.get("pitch", 0.0)
            if self._config.supports_loudness:
                payload["loudness"] = self._settings.get("loudness", 1.0)
            if self._config.supports_temperature:
                payload["temperature"] = self._settings.get("temperature", 0.6)

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
                **sdk_headers(),
            }

            url = f"{self._base_url}/text-to-speech"

            yield TTSStartedFrame(context_id=context_id)

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Sarvam API error: {error_text}")
                    return

                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            # Decode base64 audio data
            if "audios" not in response_data or not response_data["audios"]:
                yield ErrorFrame(error="No audio data received")
                return

            # Get the first audio (there should be only one for single text input)
            base64_audio = response_data["audios"][0]
            audio_data = base64.b64decode(base64_audio)

            # Strip WAV header (first 44 bytes) if present
            if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
                logger.debug("Stripping WAV header from Sarvam audio data")
                audio_data = audio_data[44:]

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

            yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)


class SarvamTTSService(InterruptibleTTSService):
    """WebSocket-based text-to-speech service using Sarvam AI.

    Provides streaming TTS with real-time audio generation for multiple Indian languages.
    Uses WebSocket for low-latency streaming audio synthesis.

    **Model Differences:**

    - **bulbul:v2** (default):
        - Supports: pitch (-0.75 to 0.75), loudness (0.3 to 3.0), pace (0.3 to 3.0)
        - Default sample rate: 22050 Hz
        - Speakers: anushka, abhilash, manisha, vidya, arya, karun, hitesh

    - **bulbul:v3-beta** / **bulbul:v3**:
        - Does NOT support: pitch, loudness (will be ignored)
        - Supports: pace (0.5 to 2.0), temperature (0.01 to 1.0)
        - Default sample rate: 24000 Hz
        - Preprocessing is always enabled
        - Speakers: aditya, ritu, priya, neha, rahul, pooja, rohan, simran, kavya,
          amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa, kabir,
          aayan, shubh, ashutosh, advait, amelia, sophia

    **WebSocket Protocol:**
    The service uses a WebSocket connection for real-time streaming. Messages include:
    - config: Initial configuration with voice settings
    - text: Text chunks for synthesis
    - flush: Signal to process remaining buffered text
    - ping: Keepalive signal

    Example::

        # Using bulbul:v2 (default)
        tts = SarvamTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2,
                loudness=1.5
            )
        )

        # Using bulbul:v3-beta with temperature control
        tts_v3 = SarvamTTSService(
            api_key="your-api-key",
            voice_id="aditya",  # Use v3 speaker
            model="bulbul:v3-beta",
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                pace=1.2,  # Range: 0.5-2.0 for v3
                temperature=0.8
            )
        )

    See https://docs.sarvam.ai/api-reference-docs/text-to-speech/stream for API details.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam TTS WebSocket service.

        Parameters:
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            pace: Speech pace multiplier. Defaults to 1.0.
                - bulbul:v2: Range 0.3 to 3.0
                - bulbul:v3-beta: Range 0.5 to 2.0
            loudness: Volume multiplier (0.3 to 3.0). Defaults to 1.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            enable_preprocessing: Enable text preprocessing. Defaults to False.
                **Note:** Always enabled for bulbul:v3-beta.
            min_buffer_size: Minimum characters to buffer before generating audio.
                Lower values reduce latency but may affect quality. Defaults to 50.
            max_chunk_length: Maximum characters processed in a single chunk.
                Controls memory usage and processing efficiency. Defaults to 150.
            output_audio_codec: Audio codec format. Options: linear16, mulaw, alaw,
                opus, flac, aac, wav, mp3. Defaults to "linear16".
            output_audio_bitrate: Audio bitrate (32k, 64k, 96k, 128k, 192k).
                Defaults to "128k".
            language: Target language for synthesis. Supports Indian languages.
            temperature: Controls output randomness for bulbul:v3-beta (0.01 to 1.0).
                Lower = more deterministic, higher = more random. Defaults to 0.6.
                **Note:** Only supported for bulbul:v3-beta. Ignored for v2.

        **Speakers by Model:**

        bulbul:v2:
            - Female: anushka (default), manisha, vidya, arya
            - Male: abhilash, karun, hitesh

        bulbul:v3-beta:
            - aditya (default), ritu, priya, neha, rahul, pooja, rohan, simran,
              kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit,
              roopa, kabir, aayan, shubh, ashutosh, advait, amelia, sophia
        """

        pitch: Optional[float] = Field(
            default=0.0,
            ge=-0.75,
            le=0.75,
            description="Voice pitch adjustment. Only for bulbul:v2.",
        )
        pace: Optional[float] = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Speech pace. v2: 0.3-3.0, v3: 0.5-2.0.",
        )
        loudness: Optional[float] = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Volume multiplier. Only for bulbul:v2.",
        )
        enable_preprocessing: Optional[bool] = Field(
            default=False,
            description="Enable text preprocessing. Always enabled for v3 models.",
        )
        min_buffer_size: Optional[int] = Field(
            default=50,
            description="Minimum characters to buffer before TTS processing.",
        )
        max_chunk_length: Optional[int] = Field(
            default=150,
            description="Maximum length for sentence splitting.",
        )
        output_audio_codec: Optional[str] = Field(
            default="linear16",
            description="Audio codec: linear16, mulaw, alaw, opus, flac, aac, wav, mp3.",
        )
        output_audio_bitrate: Optional[str] = Field(
            default="128k",
            description="Audio bitrate: 32k, 64k, 96k, 128k, 192k.",
        )
        language: Optional[Language] = Language.EN
        temperature: Optional[float] = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Output randomness for bulbul:v3-beta only. Range: 0.01-1.0.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "bulbul:v2",
        voice_id: Optional[str] = None,
        url: str = "wss://api.sarvam.ai/text-to-speech/ws",
        aggregate_sentences: Optional[bool] = True,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service with voice and transport configuration.

        Args:
            api_key: Sarvam API key for authenticating TTS requests.
            model: TTS model to use. Options:
                - "bulbul:v2" (default): Standard model with pitch/loudness support
                - "bulbul:v3-beta": Advanced model with temperature control
            voice_id: Speaker voice ID. If None, uses model-appropriate default.
            url: WebSocket URL for the TTS backend (default production URL).
            aggregate_sentences: Merge multiple sentences into one audio chunk (default True).
            sample_rate: Output audio sample rate in Hz (8000, 16000, 22050, 24000).
                If None, uses model-specific default.
            params: Optional input parameters to override defaults.
            **kwargs: Arguments forwarded to InterruptibleTTSService.

        See https://docs.sarvam.ai/api-reference-docs/text-to-speech/stream
        """
        # Get model configuration (validates model exists)
        if model not in TTS_MODEL_CONFIGS:
            allowed = ", ".join(sorted(TTS_MODEL_CONFIGS.keys()))
            raise ValueError(f"Unsupported model '{model}'. Allowed values: {allowed}.")

        self._config = TTS_MODEL_CONFIGS[model]

        # Set default sample rate based on model if not specified
        if sample_rate is None:
            sample_rate = self._config.default_sample_rate

        # Initialize parent class first
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )
        params = params or SarvamTTSService.InputParams()

        # Set default voice based on model if not specified
        if voice_id is None:
            voice_id = self._config.default_speaker

        # WebSocket endpoint URL with model query parameter
        self._websocket_url = f"{url}?model={model}"
        self._api_key = api_key
        self.set_model_name(model)
        self.set_voice(voice_id)

        # Validate and clamp pace to model's valid range
        pace = params.pace
        pace_min, pace_max = self._config.pace_range
        if pace is not None and (pace < pace_min or pace > pace_max):
            logger.warning(f"Pace {pace} is outside model range ({pace_min}-{pace_max}). Clamping.")
            pace = max(pace_min, min(pace_max, pace))

        # Build base settings
        self._settings = {
            "target_language_code": (
                self.language_to_service_language(params.language) if params.language else "en-IN"
            ),
            "speaker": voice_id,
            "speech_sample_rate": str(sample_rate),
            "enable_preprocessing": (
                True if self._config.preprocessing_always_enabled else params.enable_preprocessing
            ),
            "min_buffer_size": params.min_buffer_size,
            "max_chunk_length": params.max_chunk_length,
            "output_audio_codec": params.output_audio_codec,
            "output_audio_bitrate": params.output_audio_bitrate,
            "pace": pace,
            "model": model,
        }

        # Add parameters based on model support
        if self._config.supports_pitch:
            self._settings["pitch"] = params.pitch
        elif params.pitch != 0.0:
            logger.warning(f"pitch parameter is ignored for {model}")

        if self._config.supports_loudness:
            self._settings["loudness"] = params.loudness
        elif params.loudness != 1.0:
            logger.warning(f"loudness parameter is ignored for {model}")

        if self._config.supports_temperature:
            self._settings["temperature"] = params.temperature
        elif params.temperature != 0.6:
            logger.warning(f"temperature parameter is ignored for {model}")

        self._receive_task = None
        self._keepalive_task = None
        self._context_id: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        # WebSocket API expects sample rate as string
        self._settings["speech_sample_rate"] = str(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending flush command."""
        try:
            if self._websocket:
                msg = {"type": "flush"}
                await self._websocket.send(json.dumps(msg))
        except Exception as e:
            await self.push_error(error_msg=f"Error sending flush to Sarvam: {e}", exception=e)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and flush audio if it's the end of a full response."""
        await super().process_frame(frame, direction)

        # When the LLM finishes responding, flush any remaining text in Sarvam's buffer
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if voice changed."""
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")
            await self._send_config()

    async def _connect(self):
        """Connect to Sarvam WebSocket and start background tasks."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(
                self._keepalive_task_handler(),
            )

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Sarvam API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={
                    "api-subscription-key": self._api_key,
                    **sdk_headers(),
                },
            )
            logger.debug("Connected to Sarvam TTS Websocket")
            await self._send_config()

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Sarvam TTS Websocket: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _send_config(self):
        """Send initial configuration message."""
        if not self._websocket:
            raise Exception("WebSocket not connected")
        self._settings["speaker"] = self._voice_id
        logger.debug(f"Config being sent is {self._settings}")
        config_message = {"type": "config", "data": self._settings}

        try:
            await self._websocket.send(json.dumps(config_message))
            logger.debug("Configuration sent successfully")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Sarvam")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._context_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from Sarvam WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("type") == "audio":
                    # Check for interruption before processing audio
                    await self.stop_ttfb_metrics()
                    audio = base64.b64decode(msg["data"]["audio"])
                    frame = TTSAudioRawFrame(
                        audio, self.sample_rate, 1, context_id=self._context_id
                    )
                    await self.push_frame(frame)
                elif msg.get("type") == "error":
                    error_msg = msg["data"]["message"]
                    await self.push_error(error_msg=f"TTS Error: {error_msg}")

                    # If it's a timeout error, the connection might need to be reset
                    if "too long" in error_msg.lower() or "timeout" in error_msg.lower():
                        logger.warning("Connection timeout detected, service may need restart")

                    await self.push_frame(ErrorFrame(error=f"TTS Error: {error_msg}"))

    async def _keepalive_task_handler(self):
        """Handle keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 20
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive message to maintain connection."""
        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "ping"}
            await self._websocket.send(json.dumps(msg))

    async def _send_text(self, text: str):
        """Send text to Sarvam WebSocket for synthesis."""
        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "text", "data": {"text": text}}
            await self._websocket.send(json.dumps(msg))
        else:
            logger.warning("WebSocket not ready, cannot send text")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech audio frames from input text using Sarvam TTS.

        Sends text over WebSocket for synthesis and yields corresponding audio or status frames.

        Args:
            text: The text input to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame objects including TTSStartedFrame, TTSAudioRawFrame(s, context_id=context_id), or TTSStoppedFrame.
        """
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self.start_ttfb_metrics()
                # Store context_id for use in _receive_messages
                self._context_id = context_id
                yield TTSStartedFrame(context_id=context_id)
                await self._send_text(text)
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
