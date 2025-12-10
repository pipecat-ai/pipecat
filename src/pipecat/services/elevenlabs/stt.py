#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs speech-to-text service implementation.

This module provides integration with ElevenLabs' Speech-to-Text API for transcription
using segmented audio processing. The service uploads audio files and receives
transcription results directly.
"""

import base64
import io
import json
from enum import Enum
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService, WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use ElevenLabs Realtime STT, you need to `pip install pipecat-ai[elevenlabs]`."
    )
    raise Exception(f"Missing module: {e}")


def language_to_elevenlabs_language(language: Language) -> Optional[str]:
    """Convert a Language enum to ElevenLabs language code.

    Source:
        https://elevenlabs.io/docs/capabilities/speech-to-text

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding ElevenLabs language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AF: "afr",  # Afrikaans
        Language.AM: "amh",  # Amharic
        Language.AR: "ara",  # Arabic
        Language.HY: "hye",  # Armenian
        Language.AS: "asm",  # Assamese
        Language.AST: "ast",  # Asturian
        Language.AZ: "aze",  # Azerbaijani
        Language.BE: "bel",  # Belarusian
        Language.BN: "ben",  # Bengali
        Language.BS: "bos",  # Bosnian
        Language.BG: "bul",  # Bulgarian
        Language.MY: "mya",  # Burmese
        Language.YUE: "yue",  # Cantonese
        Language.CA: "cat",  # Catalan
        Language.CEB: "ceb",  # Cebuano
        Language.NY: "nya",  # Chichewa
        Language.HR: "hrv",  # Croatian
        Language.CS: "ces",  # Czech
        Language.DA: "dan",  # Danish
        Language.NL: "nld",  # Dutch
        Language.EN: "eng",  # English
        Language.ET: "est",  # Estonian
        Language.FIL: "fil",  # Filipino
        Language.FI: "fin",  # Finnish
        Language.FR: "fra",  # French
        Language.FF: "ful",  # Fulah
        Language.GL: "glg",  # Galician
        Language.LG: "lug",  # Ganda
        Language.KA: "kat",  # Georgian
        Language.DE: "deu",  # German
        Language.EL: "ell",  # Greek
        Language.GU: "guj",  # Gujarati
        Language.HA: "hau",  # Hausa
        Language.HE: "heb",  # Hebrew
        Language.HI: "hin",  # Hindi
        Language.HU: "hun",  # Hungarian
        Language.IS: "isl",  # Icelandic
        Language.IG: "ibo",  # Igbo
        Language.ID: "ind",  # Indonesian
        Language.GA: "gle",  # Irish
        Language.IT: "ita",  # Italian
        Language.JA: "jpn",  # Japanese
        Language.JV: "jav",  # Javanese
        Language.KEA: "kea",  # Kabuverdianu
        Language.KN: "kan",  # Kannada
        Language.KK: "kaz",  # Kazakh
        Language.KM: "khm",  # Khmer
        Language.KO: "kor",  # Korean
        Language.KU: "kur",  # Kurdish
        Language.KY: "kir",  # Kyrgyz
        Language.LO: "lao",  # Lao
        Language.LV: "lav",  # Latvian
        Language.LN: "lin",  # Lingala
        Language.LT: "lit",  # Lithuanian
        Language.LUO: "luo",  # Luo
        Language.LB: "ltz",  # Luxembourgish
        Language.MK: "mkd",  # Macedonian
        Language.MS: "msa",  # Malay
        Language.ML: "mal",  # Malayalam
        Language.MT: "mlt",  # Maltese
        Language.ZH: "zho",  # Mandarin Chinese
        Language.MI: "mri",  # Māori
        Language.MR: "mar",  # Marathi
        Language.MN: "mon",  # Mongolian
        Language.NE: "nep",  # Nepali
        Language.NSO: "nso",  # Northern Sotho
        Language.NO: "nor",  # Norwegian
        Language.OC: "oci",  # Occitan
        Language.OR: "ori",  # Odia
        Language.PS: "pus",  # Pashto
        Language.FA: "fas",  # Persian
        Language.PL: "pol",  # Polish
        Language.PT: "por",  # Portuguese
        Language.PA: "pan",  # Punjabi
        Language.RO: "ron",  # Romanian
        Language.RU: "rus",  # Russian
        Language.SR: "srp",  # Serbian
        Language.SN: "sna",  # Shona
        Language.SD: "snd",  # Sindhi
        Language.SK: "slk",  # Slovak
        Language.SL: "slv",  # Slovenian
        Language.SO: "som",  # Somali
        Language.ES: "spa",  # Spanish
        Language.SW: "swa",  # Swahili
        Language.SV: "swe",  # Swedish
        Language.TA: "tam",  # Tamil
        Language.TG: "tgk",  # Tajik
        Language.TE: "tel",  # Telugu
        Language.TH: "tha",  # Thai
        Language.TR: "tur",  # Turkish
        Language.UK: "ukr",  # Ukrainian
        Language.UMB: "umb",  # Umbundu
        Language.UR: "urd",  # Urdu
        Language.UZ: "uzb",  # Uzbek
        Language.VI: "vie",  # Vietnamese
        Language.CY: "cym",  # Welsh
        Language.WO: "wol",  # Wolof
        Language.XH: "xho",  # Xhosa
        Language.ZU: "zul",  # Zulu
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class ElevenLabsSTTService(SegmentedSTTService):
    """Speech-to-text service using ElevenLabs' file-based API.

    This service uses ElevenLabs' Speech-to-Text API to perform transcription on audio
    segments. It inherits from SegmentedSTTService to handle audio buffering and speech detection.
    The service uploads audio files to ElevenLabs and receives transcription results directly.
    """

    class InputParams(BaseModel):
        """Configuration parameters for ElevenLabs STT API.

        Parameters:
            language: Target language for transcription.
            tag_audio_events: Whether to include audio events like (laughter), (coughing), in the transcription.
        """

        language: Optional[Language] = None
        tag_audio_events: bool = True

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.elevenlabs.io",
        model: str = "scribe_v1",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs STT service.

        Args:
            api_key: ElevenLabs API key for authentication.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            base_url: Base URL for ElevenLabs API.
            model: Model ID for transcription. Defaults to "scribe_v1".
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            params: Configuration parameters for the STT service.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or ElevenLabsSTTService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._model_id = model
        self._tag_audio_events = params.tag_audio_events

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "eng",
        }

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as ElevenLabs STT service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to ElevenLabs service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The ElevenLabs-specific language code, or None if not supported.
        """
        return language_to_elevenlabs_language(language)

    async def set_language(self, language: Language):
        """Set the transcription language.

        Args:
            language: The language to use for speech-to-text transcription.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = self.language_to_service_language(language)

    async def set_model(self, model: str):
        """Set the STT model.

        Args:
            model: The model name to use for transcription.

        Note:
            ElevenLabs STT API does not currently support model selection.
            This method is provided for interface compatibility.
        """
        await super().set_model(model)
        logger.info(f"Model setting [{model}] noted, but ElevenLabs STT uses default model")

    async def _transcribe_audio(self, audio_data: bytes) -> dict:
        """Upload audio data to ElevenLabs and get transcription result.

        Args:
            audio_data: Raw audio bytes in WAV format.

        Returns:
            The transcription result data.

        Raises:
            Exception: If transcription fails or returns an error.
        """
        url = f"{self._base_url}/v1/speech-to-text"
        headers = {"xi-api-key": self._api_key}

        # Create form data with the audio file
        data = aiohttp.FormData()
        data.add_field(
            "file",
            io.BytesIO(audio_data),
            filename="audio.wav",
            content_type="audio/x-wav",
        )

        # Add required model_id, language_code, and tag_audio_events
        data.add_field("model_id", self._model_id)
        data.add_field("language_code", self._settings["language"])
        data.add_field("tag_audio_events", str(self._tag_audio_events).lower())

        async with self._session.post(url, data=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"ElevenLabs transcription error: {error_text}")
                raise Exception(f"Transcription failed with status {response.status}: {error_text}")

            result = await response.json()
            return result

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment using ElevenLabs' STT API.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text, or ErrorFrame on failure.

        Note:
            The audio is already in WAV format from the SegmentedSTTService.
            Only non-empty transcriptions are yielded.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Upload audio and get transcription result directly
            result = await self._transcribe_audio(audio)

            # Extract transcription text
            text = result.get("text", "").strip()
            if text:
                # Use the language_code returned by the API
                detected_language = result.get("language_code", "eng")

                await self._handle_transcription(text, True, detected_language)
                logger.debug(f"Transcription: [{text}]")

                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    detected_language,
                    result=result,
                )

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


def audio_format_from_sample_rate(sample_rate: int) -> str:
    """Get the appropriate audio format string for a given sample rate.

    Args:
        sample_rate: The audio sample rate in Hz.

    Returns:
        The ElevenLabs audio format string.
    """
    match sample_rate:
        case 8000:
            return "pcm_8000"
        case 16000:
            return "pcm_16000"
        case 22050:
            return "pcm_22050"
        case 24000:
            return "pcm_24000"
        case 44100:
            return "pcm_44100"
        case 48000:
            return "pcm_48000"
    logger.warning(
        f"ElevenLabsRealtimeSTTService: No audio format available for {sample_rate} sample rate, using pcm_16000"
    )
    return "pcm_16000"


class CommitStrategy(str, Enum):
    """Commit strategies for transcript segmentation."""

    MANUAL = "manual"
    VAD = "vad"


class ElevenLabsRealtimeSTTService(WebsocketSTTService):
    """Speech-to-text service using ElevenLabs' Realtime WebSocket API.

    This service uses ElevenLabs' Realtime Speech-to-Text API to perform transcription
    with ultra-low latency. It supports both partial (interim) and committed (final)
    transcripts, and can use either manual commit control or automatic Voice Activity
    Detection (VAD) for segment boundaries.

    By default, uses manual commit strategy where Pipecat's VAD controls when to
    commit transcript segments, providing consistency with other STT services.
    """

    class InputParams(BaseModel):
        """Configuration parameters for ElevenLabs Realtime STT API.

        Parameters:
            language_code: ISO-639-1 or ISO-639-3 language code. Leave None for auto-detection.
            commit_strategy: How to segment speech - manual (Pipecat VAD) or vad (ElevenLabs VAD).
            vad_silence_threshold_secs: Seconds of silence before VAD commits (0.3-3.0).
                Only used when commit_strategy is VAD. None uses ElevenLabs default.
            vad_threshold: VAD sensitivity (0.1-0.9, lower is more sensitive).
                Only used when commit_strategy is VAD. None uses ElevenLabs default.
            min_speech_duration_ms: Minimum speech duration for VAD (50-2000ms).
                Only used when commit_strategy is VAD. None uses ElevenLabs default.
            min_silence_duration_ms: Minimum silence duration for VAD (50-2000ms).
                Only used when commit_strategy is VAD. None uses ElevenLabs default.
            include_timestamps: Whether to include word-level timestamps in transcripts.
            enable_logging: Whether to enable logging on ElevenLabs' side.
        """

        language_code: Optional[str] = None
        commit_strategy: CommitStrategy = CommitStrategy.MANUAL
        vad_silence_threshold_secs: Optional[float] = None
        vad_threshold: Optional[float] = None
        min_speech_duration_ms: Optional[int] = None
        min_silence_duration_ms: Optional[int] = None
        include_timestamps: bool = False
        enable_logging: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "api.elevenlabs.io",
        model: str = "scribe_v2_realtime",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs Realtime STT service.

        Args:
            api_key: ElevenLabs API key for authentication.
            base_url: Base URL for ElevenLabs WebSocket API.
            model: Model ID for transcription. Defaults to "scribe_v2_realtime".
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            params: Configuration parameters for the STT service.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or ElevenLabsRealtimeSTTService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._model_id = model
        self._params = params
        self._audio_format = ""  # initialized in start()
        self._receive_task = None

        self._settings = {"language": params.language_code}

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as ElevenLabs Realtime STT service supports metrics generation.
        """
        return True

    async def set_language(self, language: Language):
        """Set the transcription language.

        Args:
            language: The language to use for speech-to-text transcription.

        Note:
            Changing language requires reconnecting to the WebSocket.
        """
        logger.info(f"Switching STT language to: [{language}]")
        new_language = (
            language_to_elevenlabs_language(language)
            if isinstance(language, Language)
            else language
        )
        self._params.language_code = new_language
        self._settings["language"] = new_language
        # Reconnect with new settings
        await self._disconnect()
        await self._connect()

    async def set_model(self, model: str):
        """Set the STT model.

        Args:
            model: The model name to use for transcription.

        Note:
            Changing model requires reconnecting to the WebSocket.
        """
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._model_id = model
        # Reconnect with new settings
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the STT service and establish WebSocket connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        self._audio_format = audio_format_from_sample_rate(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close WebSocket connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close WebSocket connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Start metrics when user starts speaking
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Send commit when user stops speaking (manual commit mode)
            if self._params.commit_strategy == CommitStrategy.MANUAL:
                if self._websocket and self._websocket.state is State.OPEN:
                    try:
                        commit_message = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": "",
                            "commit": True,
                            "sample_rate": self.sample_rate,
                        }
                        await self._websocket.send(json.dumps(commit_message))
                        logger.trace("Sent manual commit to ElevenLabs")
                    except Exception as e:
                        logger.warning(f"Failed to send commit: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket responses.
        """
        # Reconnect if connection is closed
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                # Encode audio as base64
                audio_base64 = base64.b64encode(audio).decode("utf-8")

                # Send audio chunk
                message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": audio_base64,
                    "commit": False,
                    "sample_rate": self.sample_rate,
                }
                await self._websocket.send(json.dumps(message))
            except Exception as e:
                yield ErrorFrame(f"ElevenLabs Realtime STT error: {str(e)}")

        yield None

    async def _connect(self):
        """Establish WebSocket connection to ElevenLabs Realtime STT."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close WebSocket connection and cleanup tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to ElevenLabs Realtime STT WebSocket endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to ElevenLabs Realtime STT")

            # Build query parameters
            params = [f"model_id={self._model_id}"]

            if self._params.language_code:
                params.append(f"language_code={self._params.language_code}")

            params.append(f"audio_format={self._audio_format}")
            params.append(f"commit_strategy={self._params.commit_strategy.value}")

            # Add optional parameters
            if self._params.include_timestamps:
                params.append(f"include_timestamps={str(self._params.include_timestamps).lower()}")

            if self._params.enable_logging:
                params.append(f"enable_logging={str(self._params.enable_logging).lower()}")

            # Add VAD parameters if using VAD commit strategy and values are specified
            if self._params.commit_strategy == CommitStrategy.VAD:
                if self._params.vad_silence_threshold_secs is not None:
                    params.append(
                        f"vad_silence_threshold_secs={self._params.vad_silence_threshold_secs}"
                    )
                if self._params.vad_threshold is not None:
                    params.append(f"vad_threshold={self._params.vad_threshold}")
                if self._params.min_speech_duration_ms is not None:
                    params.append(f"min_speech_duration_ms={self._params.min_speech_duration_ms}")
                if self._params.min_silence_duration_ms is not None:
                    params.append(f"min_silence_duration_ms={self._params.min_silence_duration_ms}")

            ws_url = f"wss://{self._base_url}/v1/speech-to-text/realtime?{'&'.join(params)}"

            headers = {"xi-api-key": self._api_key}

            self._websocket = await websocket_connect(ws_url, additional_headers=headers)
            await self._call_event_handler("on_connected")
            logger.debug("Connected to ElevenLabs Realtime STT")
        except Exception as e:
            await self.push_error(
                error_msg=f"Unable to connect to ElevenLabs Realtime STT: {e}", exception=e
            )

    async def _disconnect_websocket(self):
        """Disconnect from ElevenLabs Realtime STT WebSocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from ElevenLabs Realtime STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _process_messages(self):
        """Process incoming WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _receive_messages(self):
        """Continuously receive and process WebSocket messages."""
        try:
            await self._process_messages()
        except Exception as e:
            logger.warning(f"{self} WebSocket connection closed: {e}")
            # Connection closed, will reconnect on next audio chunk

    async def _process_response(self, data: dict):
        """Process a response message from ElevenLabs.

        Args:
            data: Parsed JSON response data.
        """
        message_type = data.get("message_type")

        if message_type == "session_started":
            logger.debug(f"ElevenLabs session started: {data}")

        elif message_type == "partial_transcript":
            await self._on_partial_transcript(data)

        elif message_type == "committed_transcript":
            await self._on_committed_transcript(data)

        elif message_type == "committed_transcript_with_timestamps":
            await self._on_committed_transcript_with_timestamps(data)

        elif message_type == "error":
            error_msg = data.get("error", "Unknown error")
            logger.error(f"ElevenLabs error: {error_msg}")
            await self.push_error(error_msg=f"Error: {error_msg}")

        elif message_type == "auth_error":
            error_msg = data.get("error", "Authentication error")
            logger.error(f"ElevenLabs auth error: {error_msg}")
            await self.push_error(error_msg=f"Auth error: {error_msg}")

        elif message_type == "quota_exceeded_error":
            error_msg = data.get("error", "Quota exceeded")
            logger.error(f"ElevenLabs quota exceeded: {error_msg}")
            await self.push_error(error_msg=f"Quota exceeded: {error_msg}")

        else:
            logger.debug(f"Unknown message type: {message_type}")

    async def _on_partial_transcript(self, data: dict):
        """Handle partial transcript (interim results).

        Args:
            data: Partial transcript data.
        """
        text = data.get("text", "").strip()
        if not text:
            return

        await self.stop_ttfb_metrics()

        # Get language if provided
        language = data.get("language_code")

        logger.trace(f"Partial transcript: [{text}]")

        await self.push_frame(
            InterimTranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=data,
            )
        )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_committed_transcript(self, data: dict):
        """Handle committed transcript (final results).

        Args:
            data: Committed transcript data.
        """
        # If timestamps are enabled, skip this message and wait for the
        # committed_transcript_with_timestamps message which contains all the data
        if self._params.include_timestamps:
            return

        text = data.get("text", "").strip()
        if not text:
            return

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        # Get language if provided
        language = data.get("language_code")

        logger.debug(f"Committed transcript: [{text}]")

        await self._handle_transcription(text, True, language)

        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=data,
            )
        )

    async def _on_committed_transcript_with_timestamps(self, data: dict):
        """Handle committed transcript with word-level timestamps.

        This message is sent when include_timestamps=true. The result data includes:
        - text: The transcribed text
        - language_code: Detected language (if available)
        - words: Array of word objects with timing information:
            - text: The word text
            - start: Start time in seconds
            - end: End time in seconds
            - type: "word" or "spacing"
            - speaker_id: Speaker identifier (if available)
            - logprob: Log probability score (if available)
            - characters: Array of character strings (if available)

        Args:
            data: Committed transcript data with timestamps.
        """
        text = data.get("text", "").strip()
        if not text:
            return

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        # Get language if provided
        language = data.get("language_code")

        logger.debug(f"Committed transcript with timestamps: [{text}]")

        await self._handle_transcription(text, True, language)

        # This message is sent after committed_transcript when include_timestamps=true.
        # It contains the full transcript data including text and word-level timestamps.
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=data,
            )
        )
