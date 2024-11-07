#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    URLImageRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.ai_services import ImageGenService, STTService, TTSService
from pipecat.services.openai import (
    BaseOpenAILLMService,
    OpenAIAssistantContextAggregator,
    OpenAIContextAggregatorPair,
    OpenAIUserContextAggregator,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

# See .env.example for Azure configuration needed
try:
    from azure.cognitiveservices.speech import (
        CancellationReason,
        ResultReason,
        SpeechConfig,
        SpeechRecognizer,
        SpeechSynthesisOutputFormat,
        SpeechSynthesizer,
    )
    from azure.cognitiveservices.speech.audio import (
        AudioStreamFormat,
        PushAudioInputStream,
    )
    from azure.cognitiveservices.speech.dialog import AudioConfig
    from openai import AsyncAzureOpenAI
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Azure, you need to `pip install pipecat-ai[azure]`. Also, set `AZURE_SPEECH_API_KEY` and `AZURE_SPEECH_REGION` environment variables."
    )
    raise Exception(f"Missing module: {e}")


class AzureLLMService(BaseOpenAILLMService):
    def __init__(
        self, *, api_key: str, endpoint: str, model: str, api_version: str = "2023-12-01-preview"
    ):
        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, model=model)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> OpenAIContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = OpenAIAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


def sample_rate_to_output_format(sample_rate: int) -> SpeechSynthesisOutputFormat:
    match sample_rate:
        case 8000:
            return SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm
        case 16000:
            return SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        case 22050:
            return SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm
        case 24000:
            return SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        case 44100:
            return SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm
        case 48000:
            return SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
    return SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm


class AzureTTSService(TTSService):
    class InputParams(BaseModel):
        emphasis: Optional[str] = None
        language: Optional[Language] = Language.EN_US
        pitch: Optional[str] = None
        rate: Optional[str] = "1.05"
        role: Optional[str] = None
        style: Optional[str] = None
        style_degree: Optional[str] = None
        volume: Optional[str] = None

    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        voice="en-US-SaraNeural",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "sample_rate": sample_rate,
            "emphasis": params.emphasis,
            "language": self.language_to_service_language(params.language)
            if params.language
            else Language.EN_US,
            "pitch": params.pitch,
            "rate": params.rate,
            "role": params.role,
            "style": params.style,
            "style_degree": params.style_degree,
            "volume": params.volume,
        }

        speech_config = SpeechConfig(
            subscription=api_key,
            region=region,
            speech_recognition_language=self._settings["language"],
        )
        speech_config.set_speech_synthesis_output_format(sample_rate_to_output_format(sample_rate))

        self._speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        match language:
            case Language.BG:
                return "bg-BG"
            case Language.CA:
                return "ca-ES"
            case Language.ZH:
                return "zh-CN"
            case Language.ZH_TW:
                return "zh-TW"
            case Language.CS:
                return "cs-CZ"
            case Language.DA:
                return "da-DK"
            case Language.NL:
                return "nl-NL"
            case Language.EN | Language.EN_US:
                return "en-US"
            case Language.EN_AU:
                return "en-AU"
            case Language.EN_GB:
                return "en-GB"
            case Language.EN_NZ:
                return "en-NZ"
            case Language.EN_IN:
                return "en-IN"
            case Language.ET:
                return "et-EE"
            case Language.FI:
                return "fi-FI"
            case Language.NL_BE:
                return "nl-BE"
            case Language.FR:
                return "fr-FR"
            case Language.FR_CA:
                return "fr-CA"
            case Language.DE:
                return "de-DE"
            case Language.DE_CH:
                return "de-CH"
            case Language.EL:
                return "el-GR"
            case Language.HI:
                return "hi-IN"
            case Language.HU:
                return "hu-HU"
            case Language.ID:
                return "id-ID"
            case Language.IT:
                return "it-IT"
            case Language.JA:
                return "ja-JP"
            case Language.KO:
                return "ko-KR"
            case Language.LV:
                return "lv-LV"
            case Language.LT:
                return "lt-LT"
            case Language.MS:
                return "ms-MY"
            case Language.NO:
                return "nb-NO"
            case Language.PL:
                return "pl-PL"
            case Language.PT:
                return "pt-PT"
            case Language.PT_BR:
                return "pt-BR"
            case Language.RO:
                return "ro-RO"
            case Language.RU:
                return "ru-RU"
            case Language.SK:
                return "sk-SK"
            case Language.ES:
                return "es-ES"
            case Language.SV:
                return "sv-SE"
            case Language.TH:
                return "th-TH"
            case Language.TR:
                return "tr-TR"
            case Language.UK:
                return "uk-UA"
            case Language.VI:
                return "vi-VN"
        return None

    def _construct_ssml(self, text: str) -> str:
        language = self._settings["language"]
        ssml = (
            f"<speak version='1.0' xml:lang='{language}' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self._voice_id}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
        )

        if self._settings["style"]:
            ssml += f"<mstts:express-as style='{self._settings['style']}'"
            if self._settings["style_degree"]:
                ssml += f" styledegree='{self._settings['style_degree']}'"
            if self._settings["role"]:
                ssml += f" role='{self._settings['role']}'"
            ssml += ">"

        prosody_attrs = []
        if self._settings["rate"]:
            prosody_attrs.append(f"rate='{self._settings['rate']}'")
        if self._settings["pitch"]:
            prosody_attrs.append(f"pitch='{self._settings['pitch']}'")
        if self._settings["volume"]:
            prosody_attrs.append(f"volume='{self._settings['volume']}'")

        ssml += f"<prosody {' '.join(prosody_attrs)}>"

        if self._settings["emphasis"]:
            ssml += f"<emphasis level='{self._settings['emphasis']}'>"

        ssml += text

        if self._settings["emphasis"]:
            ssml += "</emphasis>"

        ssml += "</prosody>"

        if self._settings["style"]:
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"

        return ssml

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()

        ssml = self._construct_ssml(text)

        result = await asyncio.to_thread(self._speech_synthesizer.speak_ssml, (ssml))

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            await self.start_tts_usage_metrics(text)
            await self.stop_ttfb_metrics()
            yield TTSStartedFrame()
            # Azure always sends a 44-byte header. Strip it off.
            yield TTSAudioRawFrame(
                audio=result.audio_data[44:],
                sample_rate=self._settings["sample_rate"],
                num_channels=1,
            )
            yield TTSStoppedFrame()
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.warning(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                logger.error(f"{self} error: {cancellation_details.error_details}")


class AzureSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        language=Language.EN_US,
        sample_rate=24000,
        channels=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(subscription=api_key, region=region)
        speech_config.speech_recognition_language = language

        stream_format = AudioStreamFormat(samples_per_second=sample_rate, channels=channels)
        self._audio_stream = PushAudioInputStream(stream_format)

        audio_config = AudioConfig(stream=self._audio_stream)
        self._speech_recognizer = SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        self._speech_recognizer.recognized.connect(self._on_handle_recognized)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        self._audio_stream.write(audio)
        await self.stop_processing_metrics()
        yield None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._speech_recognizer.start_continuous_recognition_async()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._speech_recognizer.stop_continuous_recognition_async()
        self._audio_stream.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._speech_recognizer.stop_continuous_recognition_async()
        self._audio_stream.close()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            frame = TranscriptionFrame(event.result.text, "", time_now_iso8601())
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())


class AzureImageGenServiceREST(ImageGenService):
    def __init__(
        self,
        *,
        image_size: str,
        api_key: str,
        endpoint: str,
        model: str,
        aiohttp_session: aiohttp.ClientSession,
        api_version="2023-06-01-preview",
    ):
        super().__init__()

        self._api_key = api_key
        self._azure_endpoint = endpoint
        self._api_version = api_version
        self.set_model_name(model)
        self._image_size = image_size
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"

        headers = {"api-key": self._api_key, "Content-Type": "application/json"}

        body = {
            # Enter your prompt text here
            "prompt": prompt,
            "size": self._image_size,
            "n": 1,
        }

        async with self._aiohttp_session.post(url, headers=headers, json=body) as submission:
            # We never get past this line, because this header isn't
            # defined on a 429 response, but something is eating our
            # exceptions!
            operation_location = submission.headers["operation-location"]
            status = ""
            attempts_left = 120
            json_response = None
            while status != "succeeded":
                attempts_left -= 1
                if attempts_left == 0:
                    logger.error(f"{self} error: image generation timed out")
                    yield ErrorFrame("Image generation timed out")
                    return

                await asyncio.sleep(1)

                response = await self._aiohttp_session.get(operation_location, headers=headers)

                json_response = await response.json()
                status = json_response["status"]

            image_url = json_response["result"]["data"][0]["url"] if json_response else None
            if not image_url:
                logger.error(f"{self} error: image generation failed")
                yield ErrorFrame("Image generation failed")
                return

            # Load the image from the url
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                frame = URLImageRawFrame(
                    url=image_url, image=image.tobytes(), size=image.size, format=image.format
                )
                yield frame
