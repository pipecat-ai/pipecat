#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, model_validator

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import WordTTSService
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


class ElevenLabsTTSService(WordTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        optimize_streaming_latency: Optional[str] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None

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
        model: str = "eleven_turbo_v2_5",
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
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            stop_frame_timeout_s=2.0,
            sample_rate=sample_rate_from_output_format(output_format),
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._settings = {
            "sample_rate": sample_rate_from_output_format(output_format),
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN,
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

        # Websocket connection to ElevenLabs.
        self._websocket = None
        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        match language:
            case Language.BG:
                return "bg"
            case Language.ZH:
                return "zh"
            case Language.CS:
                return "cs"
            case Language.DA:
                return "da"
            case Language.NL:
                return "nl"
            case (
                Language.EN
                | Language.EN_US
                | Language.EN_AU
                | Language.EN_GB
                | Language.EN_NZ
                | Language.EN_IN
            ):
                return "en"
            case Language.FI:
                return "fi"
            case Language.FR | Language.FR_CA:
                return "fr"
            case Language.DE | Language.DE_CH:
                return "de"
            case Language.EL:
                return "el"
            case Language.HI:
                return "hi"
            case Language.HU:
                return "hu"
            case Language.ID:
                return "id"
            case Language.IT:
                return "it"
            case Language.JA:
                return "ja"
            case Language.KO:
                return "ko"
            case Language.MS:
                return "ms"
            case Language.NO:
                return "no"
            case Language.PL:
                return "pl"
            case Language.PT:
                return "pt-PT"
            case Language.PT_BR:
                return "pt-BR"
            case Language.RO:
                return "ro"
            case Language.RU:
                return "ru"
            case Language.SK:
                return "sk"
            case Language.ES:
                return "es"
            case Language.SV:
                return "sv"
            case Language.TR:
                return "tr"
            case Language.UK:
                return "uk"
            case Language.VI:
                return "vi"
        return None

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
                await self.add_word_timestamps([("LLMFullResponseEndFrame", 0)])

    async def _connect(self):
        try:
            voice_id = self._voice_id
            model = self.model_name
            output_format = self._settings["output_format"]
            url = f"{self._url}/v1/text-to-speech/{voice_id}/stream-input?model_id={model}&output_format={output_format}"

            if self._settings["optimize_streaming_latency"]:
                url += f"&optimize_streaming_latency={self._settings['optimize_streaming_latency']}"

            # Language can only be used with the 'eleven_turbo_v2_5' model
            language = self._settings["language"]
            if model == "eleven_turbo_v2_5":
                url += f"&language_code={language}"
            else:
                logger.warning(
                    f"Language code [{language}] not applied. Language codes can only be used with the 'eleven_turbo_v2_5' model."
                )

            self._websocket = await websockets.connect(url)
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
            self._keepalive_task = self.get_event_loop().create_task(self._keepalive_task_handler())

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

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                await self._websocket.send(json.dumps({"text": ""}))
                await self._websocket.close()
                self._websocket = None

            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None

            if self._keepalive_task:
                self._keepalive_task.cancel()
                await self._keepalive_task
                self._keepalive_task = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    async def _receive_task_handler(self):
        try:
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
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception: {e}")

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
