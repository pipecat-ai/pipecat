#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, model_validator

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService, WordTTSService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language

# See .env.example for ElevenLabs configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use ElevenLabs, you need to `pip install pipecat-ai[elevenlabs]`. Also, set `ELEVENLABS_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

ElevenLabsOutputFormat = Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"]

# Models that support language codes
# eleven_multilingual_v2 doesn't support language codes, so it's excluded
ELEVENLABS_MULTILINGUAL_MODELS = {
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
}


def language_to_elevenlabs_language(language: Language) -> str | None:
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BG: "bg",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SV: "sv",
        Language.TA: "ta",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


def sample_rate_from_output_format(output_format: str) -> int:
    match output_format:
        case "pcm_16000":
            return 16000
        case "pcm_22050":
            return 22050
        case "pcm_24000":
            return 24000
        case "pcm_44100":
            return 44100
    return 16000


def calculate_word_times(
    alignment_info: Mapping[str, Any], cumulative_time: float
) -> List[Tuple[str, float]]:
    zipped_times = list(zip(alignment_info["chars"], alignment_info["charStartTimesMs"]))

    words = "".join(alignment_info["chars"]).split(" ")

    # Calculate start time for each word. We do this by finding a space character
    # and using the previous word time, also taking into account there might not
    # be a space at the end.
    times = []
    for i, (a, b) in enumerate(zipped_times):
        if a == " " or i == len(zipped_times) - 1:
            t = cumulative_time + (zipped_times[i - 1][1] / 1000.0)
            times.append(t)

    word_times = list(zip(words, times))

    return word_times


class ElevenLabsTTSService(WordTTSService, WebsocketService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        optimize_streaming_latency: Optional[str] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        auto_mode: Optional[bool] = True

        @model_validator(mode="after")
        def validate_voice_settings(self):
            stability = self.stability
            similarity_boost = self.similarity_boost
            if (stability is None) != (similarity_boost is None):
                raise ValueError(
                    "Both 'stability' and 'similarity_boost' must be provided when using voice settings"
                )
            return self

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        url: str = "wss://api.elevenlabs.io",
        output_format: ElevenLabsOutputFormat = "pcm_24000",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. ElevenLabs gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        #
        # Finally, ElevenLabs doesn't provide information on when the bot stops
        # speaking for a while, so we want the parent class to send TTSStopFrame
        # after a short period not receiving any audio.
        WordTTSService.__init__(
            self,
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            stop_frame_timeout_s=2.0,
            sample_rate=sample_rate_from_output_format(output_format),
            **kwargs,
        )
        WebsocketService.__init__(self)

        self._api_key = api_key
        self._url = url
        self._settings = {
            "sample_rate": sample_rate_from_output_format(output_format),
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
            "output_format": output_format,
            "optimize_streaming_latency": params.optimize_streaming_latency,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
            "auto_mode": str(params.auto_mode).lower(),
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._voice_settings = self._set_voice_settings()

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_elevenlabs_language(language)

    def _set_voice_settings(self):
        voice_settings = {}
        if (
            self._settings["stability"] is not None
            and self._settings["similarity_boost"] is not None
        ):
            voice_settings["stability"] = self._settings["stability"]
            voice_settings["similarity_boost"] = self._settings["similarity_boost"]
            if self._settings["style"] is not None:
                voice_settings["style"] = self._settings["style"]
            if self._settings["use_speaker_boost"] is not None:
                voice_settings["use_speaker_boost"] = self._settings["use_speaker_boost"]
        else:
            if self._settings["style"] is not None:
                logger.warning(
                    "'style' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
                )
            if self._settings["use_speaker_boost"] is not None:
                logger.warning(
                    "'use_speaker_boost' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
                )

        return voice_settings or None

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        await self._disconnect()
        await self._connect()

    async def _update_settings(self, settings: Dict[str, Any]):
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            await self._disconnect()
            await self._connect()
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._websocket:
            msg = {"text": " ", "flush": True}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("LLMFullResponseEndFrame", 0), ("Reset", 0)])

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # If we received a TTSSpeakFrame and the LLM response included text (it
        # might be that it's only a function calling response) we pause
        # processing more frames until we receive a BotStoppedSpeakingFrame.
        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._started:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _connect(self):
        await self._connect_websocket()

        self._receive_task = self.get_event_loop().create_task(
            self._receive_task_handler(self.push_error)
        )
        self._keepalive_task = self.get_event_loop().create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
            await self._receive_task
            self._receive_task = None

        if self._keepalive_task:
            self._keepalive_task.cancel()
            await self._keepalive_task
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            logger.debug("Connecting to ElevenLabs")

            voice_id = self._voice_id
            model = self.model_name
            output_format = self._settings["output_format"]
            url = f"{self._url}/v1/text-to-speech/{voice_id}/stream-input?model_id={model}&output_format={output_format}&auto_mode={self._settings['auto_mode']}"

            if self._settings["optimize_streaming_latency"]:
                url += f"&optimize_streaming_latency={self._settings['optimize_streaming_latency']}"

            # Language can only be used with the ELEVENLABS_MULTILINGUAL_MODELS
            language = self._settings["language"]
            if model in ELEVENLABS_MULTILINGUAL_MODELS:
                url += f"&language_code={language}"
            else:
                logger.warning(
                    f"Language code [{language}] not applied. Language codes can only be used with multilingual models: {', '.join(sorted(ELEVENLABS_MULTILINGUAL_MODELS))}"
                )

            self._websocket = await websockets.connect(url)

            # According to ElevenLabs, we should always start with a single space.
            msg: Dict[str, Any] = {
                "text": " ",
                "xi_api_key": self._api_key,
            }
            if self._voice_settings:
                msg["voice_settings"] = self._voice_settings
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from ElevenLabs")
                await self._websocket.send(json.dumps({"text": ""}))
                await self._websocket.close()
                self._websocket = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    async def _receive_messages(self):
        async for message in self._websocket:
            msg = json.loads(message)
            if msg.get("audio"):
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()

                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self._settings["sample_rate"], 1)
                await self.push_frame(frame)
            if msg.get("alignment"):
                word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                await self.add_word_timestamps(word_times)
                self._cumulative_time = word_times[-1][1]

    async def _keepalive_task_handler(self):
        while True:
            try:
                await asyncio.sleep(10)
                await self._send_text("")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self} exception: {e}")

    async def _send_text(self, text: str):
        if self._websocket:
            msg = {"text": text + " "}
            await self._websocket.send(json.dumps(msg))

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0

                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")


class ElevenLabsHttpTTSService(TTSService):
    """ElevenLabs Text-to-Speech service using HTTP streaming.

    Args:
        api_key: ElevenLabs API key
        voice_id: ID of the voice to use
        aiohttp_session: aiohttp ClientSession
        model: Model ID (default: "eleven_flash_v2_5" for low latency)
        base_url: API base URL
        output_format: Audio output format (PCM)
        params: Additional parameters for voice configuration
    """

    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        optimize_streaming_latency: Optional[int] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "eleven_flash_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        output_format: ElevenLabsOutputFormat = "pcm_24000",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate_from_output_format(output_format), **kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._output_format = output_format
        self._params = params
        self._session = aiohttp_session

        self._settings = {
            "sample_rate": sample_rate_from_output_format(output_format),
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
            "output_format": output_format,
            "optimize_streaming_latency": params.optimize_streaming_latency,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._voice_settings = self._set_voice_settings()

    def can_generate_metrics(self) -> bool:
        return True

    def _set_voice_settings(self) -> Optional[Dict[str, Union[float, bool]]]:
        """Configure voice settings if stability and similarity_boost are provided.

        Returns:
            Dictionary of voice settings or None if required parameters are missing.
        """
        voice_settings: Dict[str, Union[float, bool]] = {}
        if (
            self._settings["stability"] is not None
            and self._settings["similarity_boost"] is not None
        ):
            voice_settings["stability"] = float(self._settings["stability"])
            voice_settings["similarity_boost"] = float(self._settings["similarity_boost"])
            if self._settings["style"] is not None:
                voice_settings["style"] = float(self._settings["style"])
            if self._settings["use_speaker_boost"] is not None:
                voice_settings["use_speaker_boost"] = bool(self._settings["use_speaker_boost"])
        else:
            if self._settings["style"] is not None:
                logger.warning(
                    "'style' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
                )
            if self._settings["use_speaker_boost"] is not None:
                logger.warning(
                    "'use_speaker_boost' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
                )

        return voice_settings or None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ElevenLabs streaming API.

        Args:
            text: The text to convert to speech

        Yields:
            Frames containing audio data and status information
        """
        logger.debug(f"Generating TTS: [{text}]")

        url = f"{self._base_url}/v1/text-to-speech/{self._voice_id}/stream"

        payload: Dict[str, Union[str, Dict[str, Union[float, bool]]]] = {
            "text": text,
            "model_id": self._model_name,
        }

        if self._voice_settings:
            payload["voice_settings"] = self._voice_settings

        if self._settings["language"]:
            payload["language_code"] = self._settings["language"]

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        # Build query parameters
        params = {
            "output_format": self._output_format,
        }
        if self._settings["optimize_streaming_latency"] is not None:
            params["optimize_streaming_latency"] = self._settings["optimize_streaming_latency"]

        logger.debug(f"ElevenLabs request - payload: {payload}, params: {params}")

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                url, json=payload, headers=headers, params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self} error: {error_text}")
                    yield ErrorFrame(error=f"ElevenLabs API error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                async for chunk in response.content:
                    if chunk:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)

                yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))

        finally:
            yield TTSStoppedFrame()
