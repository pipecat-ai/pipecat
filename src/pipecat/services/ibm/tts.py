#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""IBM Watson Text-to-Speech service implementation.

This module provides integration with IBM Watson's Text-to-Speech API using
the WebSocket streaming interface for low-latency audio synthesis.

Watson TTS WebSocket API reference:
    https://cloud.ibm.com/apidocs/text-to-speech#synthesizewebsocket

Supported audio formats (accept parameter):
    - audio/wav          (default, with optional ;rate=<rate>)
    - audio/ogg          (with optional ;codecs=opus or ;codecs=vorbis)
    - audio/mp3 / audio/mpeg
    - audio/flac
    - audio/l16          (requires ;rate=<rate> and optionally ;channels=<n>;endianness=<big-endian|little-endian>)
    - audio/mulaw        (requires ;rate=<rate>)
    - audio/alaw         (requires ;rate=<rate>)
    - audio/basic        (8 kHz, 8-bit u-law)
    - audio/webm         (with optional ;codecs=opus or ;codecs=vorbis)

Usage example::

    tts = WatsonTTSService(
        api_key="your-api-key",
        url="https://api.us-south.text-to-speech.watson.cloud.ibm.com",
        params=WatsonTTSService.InputParams(
            voice="en-US_MichaelV3Voice",
            accept="audio/wav;rate=16000",
        ),
    )
"""

import json
import time
from typing import AsyncGenerator, Optional

import aiohttp
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
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use IBM Watson TTS, you need to `pip install pipecat-ai[ibm]`."
    )
    raise Exception(f"Missing module: {e}")


# ---------------------------------------------------------------------------
# Voice → language helpers
# ---------------------------------------------------------------------------

# Map from pipecat Language enum to a default Watson TTS voice name.
# Full voice list: https://cloud.ibm.com/docs/text-to-speech?topic=text-to-speech-voices
_LANGUAGE_TO_DEFAULT_VOICE = {
    Language.EN: "en-US_EllieNatural",
    Language.EN_US: "en-US_EllieNatural",
    Language.EN_GB: "en-GB_ChloeNatural",
    Language.DE: "de-DE_BirgitV3Voice",
    Language.ES: "es-LA_DanielaNatural",
    Language.ES_ES: "es-ES_EnriqueV3Voice",
    Language.FR: "fr-FR_ReneeV3Voice",
    Language.IT: "it-IT_FrancescaV3Voice",
    Language.JA: "ja-JP_EmiV3Voice",
    Language.PT: "pt-BR_IsabelaV3Voice",
    Language.ZH: "zh-CN_LiNaVoice",
    Language.KO: "ko-KR_YoungmiVoice",
    Language.NL: "nl-NL_EmmaVoice",
    Language.AR: "ar-MS_OmarVoice",
}


def language_to_watson_tts_voice(language: Language) -> Optional[str]:
    """Return a default Watson TTS voice name for the given language.

    Args:
        language: A pipecat :class:`Language` enum value.

    Returns:
        A Watson voice name string, or ``None`` if the language is not mapped.
    """
    return _LANGUAGE_TO_DEFAULT_VOICE.get(language)


# ---------------------------------------------------------------------------
# WatsonTTSService
# ---------------------------------------------------------------------------


class WatsonTTSService(TTSService):
    """IBM Watson Text-to-Speech WebSocket service.

    Streams synthesized audio from Watson TTS over a WebSocket connection.
    Each call to :meth:`run_tts` opens a new WebSocket, sends the text
    message, collects all binary audio chunks, and closes the connection.

    Watson TTS WebSocket protocol summary:

    1. Connect to ``wss://<host>/v1/synthesize?voice=<voice>&access_token=<token>``
    2. Send a JSON *open* message: ``{"text": "<text>", "accept": "<mime-type>"}``
    3. Receive binary audio chunks until the server closes the connection.
    4. Any JSON message from the server indicates an error.

    Authentication uses IBM IAM tokens obtained from the API key.
    Tokens are cached and refreshed automatically before expiry.
    """

    class InputParams(BaseModel):
        """Watson TTS synthesis parameters.

        Parameters:
            voice: Watson voice name (e.g. ``"en-US_MichaelV3Voice"``).
                   If *None*, a default voice is chosen based on *language*.
            language: Pipecat language used to pick a default voice when
                      *voice* is not specified.
            accept: MIME type for the synthesized audio.
                    Defaults to ``"audio/wav;rate=16000"``.
                    Examples: ``"audio/ogg;codecs=opus"``,
                    ``"audio/l16;rate=22050"``.
            rate_percentage: Speaking-rate adjustment as a signed percentage
                             relative to the default rate (e.g. ``10`` for
                             10 % faster, ``-10`` for 10 % slower).
                             Range: ``-100`` to ``+100``.  ``None`` = default.
            pitch_percentage: Pitch adjustment as a signed percentage relative
                              to the default pitch.
                              Range: ``-100`` to ``+100``.  ``None`` = default.
            spell_out_mode: Controls how individual characters are spoken.
                            ``"default"`` (default), ``"all"``, or ``"none"``.
            customization_id: GUID of a custom voice model to use.
            x_watson_learning_opt_out: Set to ``True`` to opt out of Watson
                                       request logging.
        """

        voice: Optional[str] = None
        language: Optional[Language] = Language.EN_US
        accept: str = "audio/wav;rate=16000"
        rate_percentage: Optional[int] = None
        pitch_percentage: Optional[int] = None
        spell_out_mode: Optional[str] = None
        customization_id: Optional[str] = None
        x_watson_learning_opt_out: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        params: Optional["WatsonTTSService.InputParams"] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialise the Watson TTS service.

        Args:
            api_key: IBM Cloud API key used to obtain IAM access tokens.
            url: Watson TTS service URL, e.g.
                 ``"https://api.us-south.text-to-speech.watson.cloud.ibm.com"``.
            params: Optional :class:`InputParams` for voice and synthesis
                    configuration.
            sample_rate: Desired output sample rate in Hz.  When *None* the
                         rate embedded in the *accept* MIME type is used, or
                         16 000 Hz as a fallback.
            **kwargs: Additional keyword arguments forwarded to
                      :class:`~pipecat.services.tts_service.TTSService`.
        """
        self._api_key = api_key
        # Normalise URL: strip trailing slash, ensure no trailing path
        self._base_url = url.rstrip("/")

        self._params = params or WatsonTTSService.InputParams()

        # Resolve voice: explicit > language-derived > hard-coded fallback
        voice = self._params.voice or language_to_watson_tts_voice(
            self._params.language or Language.EN_US
        ) or "en-US_MichaelV3Voice"

        # Initialize with settings
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=TTSSettings(
                model=None,  # Watson TTS doesn't expose model selection
                voice=voice,
                language=self._params.language,
            ),
            **kwargs,
        )

        # Set voice ID (parent class sets self._voice_id from settings.voice)
        self._voice_id = voice

        # IAM token cache
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Latency tracking
        self._synthesis_start_time: Optional[float] = None
        self._first_audio_time: Optional[float] = None

        # Derive sample rate from the accept MIME type when not provided
        if sample_rate is None:
            self._sample_rate = self._parse_rate_from_accept(self._params.accept)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rate_from_accept(accept: str) -> int:
        """Extract the sample rate from an *accept* MIME type string.

        For example ``"audio/wav;rate=16000"`` → ``16000``.
        Falls back to ``16000`` when no rate parameter is present.

        Args:
            accept: MIME type string, possibly with parameters.

        Returns:
            Sample rate as an integer.
        """
        for part in accept.split(";"):
            part = part.strip()
            if part.startswith("rate="):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    pass
        return 16000

    async def _get_access_token(self) -> str:
        """Obtain (or return a cached) IBM IAM access token.

        Tokens are valid for one hour; this method refreshes them five
        minutes before expiry.

        Returns:
            A valid IAM bearer token string.

        Raises:
            Exception: If the IAM token request fails.
        """
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token

        iam_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self._api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(iam_url, headers=headers, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to obtain IAM token: {error_text}")

                result = await response.json()
                token: str = result["access_token"]
                self._access_token = token
                # Refresh 5 minutes before the token expires
                self._token_expiry = time.time() + result.get("expires_in", 3600) - 300
                logger.debug("Obtained new IAM access token for Watson TTS")
                return token

    def _build_ws_url(self, token: str) -> str:
        """Build the Watson TTS WebSocket URL.

        Args:
            token: A valid IAM bearer token.

        Returns:
            A fully-qualified ``wss://`` URL string.
        """
        ws_host = (
            self._base_url.replace("https://", "").replace("http://", "")
        )
        url = f"wss://{ws_host}/v1/synthesize"

        query_parts = [
            f"voice={self._voice_id}",
            f"access_token={token}",
        ]
        if self._params.customization_id:
            query_parts.append(f"customization_id={self._params.customization_id}")

        return url + "?" + "&".join(query_parts)

    def _build_open_message(self, text: str) -> str:
        """Build the JSON *open* message sent to Watson TTS.

        Args:
            text: The text to synthesise.

        Returns:
            A JSON-encoded string ready to send over the WebSocket.
        """
        msg: dict = {
            "text": text,
            "accept": self._params.accept,
        }

        # Optional SSML-equivalent parameters expressed as Watson JSON fields
        if self._params.rate_percentage is not None:
            msg["rate_percentage"] = self._params.rate_percentage
        if self._params.pitch_percentage is not None:
            msg["pitch_percentage"] = self._params.pitch_percentage
        if self._params.spell_out_mode is not None:
            msg["spell_out_mode"] = self._params.spell_out_mode

        return json.dumps(msg)

    # ------------------------------------------------------------------
    # TTSService interface
    # ------------------------------------------------------------------

    def can_generate_metrics(self) -> bool:
        """Return ``True``; Watson TTS supports TTFB and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a pipecat :class:`Language` to a Watson voice name.

        Args:
            language: The language to convert.

        Returns:
            A Watson voice name string, or ``None`` if not mapped.
        """
        return language_to_watson_tts_voice(language)

    async def set_voice(self, voice: str):
        """Change the Watson voice used for synthesis.

        Args:
            voice: Watson voice name, e.g. ``"en-US_AllisonV3Voice"``.
        """
        logger.debug(f"{self}: Setting Watson TTS voice to [{voice}]")
        self._voice_id = voice

    async def set_model(self, model: str):
        """Set the model name (maps to Watson voice for compatibility).

        Args:
            model: Watson voice name.
        """
        await super().set_model(model)
        await self.set_voice(model)

    async def start(self, frame: StartFrame):
        """Start the TTS service.

        Args:
            frame: The :class:`~pipecat.frames.frames.StartFrame` that
                   triggers service startup.
        """
        await super().start(frame)
        logger.debug(f"{self}: Watson TTS service started (voice={self._voice_id})")

    async def stop(self, frame: EndFrame):
        """Stop the TTS service.

        Args:
            frame: The :class:`~pipecat.frames.frames.EndFrame` that
                   triggers service shutdown.
        """
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.

        Args:
            frame: The :class:`~pipecat.frames.frames.CancelFrame` that
                   triggers service cancellation.
        """
        await super().cancel(frame)

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesise *text* using the Watson TTS WebSocket API.

        Opens a new WebSocket connection for each utterance, sends the text,
        collects all binary audio chunks, and closes the connection.

        Watson TTS WebSocket protocol:

        1. Connect to ``wss://<host>/v1/synthesize?voice=<voice>&access_token=<token>``
        2. Send JSON: ``{"text": "<text>", "accept": "<mime-type>", ...}``
        3. Receive binary audio frames until the server closes the socket.
        4. Any JSON text message from the server signals an error.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context (used for
                        frame tagging and metrics).

        Yields:
            :class:`~pipecat.frames.frames.TTSStartedFrame`,
            :class:`~pipecat.frames.frames.TTSAudioRawFrame` (one or more),
            :class:`~pipecat.frames.frames.TTSStoppedFrame`, or
            :class:`~pipecat.frames.frames.ErrorFrame` on failure.
        """
        logger.debug(f"{self}: Generating Watson TTS [{text}]")

        try:
            token = await self._get_access_token()
        except Exception as e:
            yield ErrorFrame(error=f"Watson TTS auth error: {e}")
            return

        ws_url = self._build_ws_url(token)
        open_message = self._build_open_message(text)

        # Optional opt-out header
        extra_headers = {}
        if self._params.x_watson_learning_opt_out:
            extra_headers["X-Watson-Learning-Opt-Out"] = "true"

        try:
            # Start latency tracking
            self._synthesis_start_time = time.time()
            self._first_audio_time = None
            
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame(context_id=context_id)

            async with websocket_connect(
                ws_url,
                additional_headers=extra_headers if extra_headers else None,
            ) as websocket:
                # Log WebSocket connection time
                if self._synthesis_start_time:
                    connect_latency = (time.time() - self._synthesis_start_time) * 1000
                    logger.info(f"Watson TTS WebSocket connected in {connect_latency:.2f}ms")
                
                await self._call_event_handler("on_connected")

                # Send the synthesis request
                await websocket.send(open_message)
                logger.debug(f"{self}: Sent Watson TTS open message")

                # Receive audio chunks until the server closes the connection
                async for message in websocket:
                    if isinstance(message, bytes):
                        # Binary message → audio data
                        if message:
                            # Log time to first audio chunk
                            if self._first_audio_time is None and self._synthesis_start_time:
                                self._first_audio_time = time.time()
                                first_audio_latency = (self._first_audio_time - self._synthesis_start_time) * 1000
                                logger.info(f"Watson TTS first audio chunk received in {first_audio_latency:.2f}ms")
                            
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(
                                audio=message,
                                sample_rate=self._sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )
                    else:
                        # Text message → Watson error or status JSON
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"{self}: Received non-JSON text message: {message}"
                            )
                            continue

                        if "error" in data:
                            error_msg = data["error"]
                            logger.error(f"{self}: Watson TTS error: {error_msg}")
                            yield ErrorFrame(error=f"Watson TTS error: {error_msg}")
                            return
                        elif "warnings" in data:
                            for warning in data["warnings"]:
                                logger.warning(f"{self}: Watson TTS warning: {warning}")
                        else:
                            logger.debug(
                                f"{self}: Watson TTS text message: {message}"
                            )

                await self._call_event_handler("on_disconnected")

        except Exception as e:
            logger.error(f"{self}: Watson TTS exception: {e}")
            yield ErrorFrame(error=f"Watson TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)

# Made with Bob
