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
        ServicePropertyChannel,
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


def language_to_azure_language(language: Language) -> str | None:
    language_map = {
        Language.BG: "bg-BG",
        Language.CA: "ca-ES",
        Language.ZH: "zh-CN",
        Language.ZH_TW: "zh-TW",
        Language.CS: "cs-CZ",
        Language.DA: "da-DK",
        Language.NL: "nl-NL",
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_NZ: "en-NZ",
        Language.EN_IN: "en-IN",
        Language.ET: "et-EE",
        Language.FI: "fi-FI",
        Language.NL_BE: "nl-BE",
        Language.FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.DE: "de-DE",
        Language.DE_CH: "de-CH",
        Language.EL: "el-GR",
        Language.HI: "hi-IN",
        Language.HU: "hu-HU",
        Language.ID: "id-ID",
        Language.IT: "it-IT",
        Language.JA: "ja-JP",
        Language.KO: "ko-KR",
        Language.LV: "lv-LV",
        Language.LT: "lt-LT",
        Language.MS: "ms-MY",
        Language.NO: "nb-NO",
        Language.PL: "pl-PL",
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.RO: "ro-RO",
        Language.RU: "ru-RU",
        Language.SK: "sk-SK",
        Language.ES: "es-ES",
        Language.SV: "sv-SE",
        Language.TH: "th-TH",
        Language.TR: "tr-TR",
        Language.UK: "uk-UA",
        Language.VI: "vi-VN",
    }
    return language_map.get(language)


def sample_rate_to_output_format(sample_rate: int) -> SpeechSynthesisOutputFormat:
    sample_rate_map = {
        8000: SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
        16000: SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
        22050: SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm,
        24000: SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
        44100: SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
        48000: SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
    }
    return sample_rate_map.get(sample_rate, SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)


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


class AzureBaseTTSService(TTSService):
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
            else "en-US",
            "pitch": params.pitch,
            "rate": params.rate,
            "role": params.role,
            "style": params.style,
            "style_degree": params.style_degree,
            "volume": params.volume,
        }

        self._api_key = api_key
        self._region = region
        self._voice_id = voice
        self._speech_synthesizer = None

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_azure_language(language)

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


class AzureTTSService(AzureBaseTTSService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(
            subscription=self._api_key,
            region=self._region,
            speech_recognition_language=self._settings["language"],
        )
        speech_config.set_speech_synthesis_output_format(
            sample_rate_to_output_format(self._settings["sample_rate"])
        )
        speech_config.set_service_property(
            "synthesizer.synthesis.connection.synthesisConnectionImpl",
            "websocket",
            ServicePropertyChannel.UriQueryParameter,
        )

        self._speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        # Set up event handlers
        self._audio_queue = asyncio.Queue()
        self._speech_synthesizer.synthesizing.connect(self._handle_synthesizing)
        self._speech_synthesizer.synthesis_completed.connect(self._handle_completed)
        self._speech_synthesizer.synthesis_canceled.connect(self._handle_canceled)

    def _handle_synthesizing(self, evt):
        """Handle audio chunks as they arrive"""
        if evt.result and evt.result.audio_data:
            self._audio_queue.put_nowait(evt.result.audio_data)

    def _handle_completed(self, evt):
        """Handle synthesis completion"""
        self._audio_queue.put_nowait(None)  # Signal completion

    def _handle_canceled(self, evt):
        """Handle synthesis cancellation"""
        logger.error(f"Speech synthesis canceled: {evt.result.cancellation_details.reason}")
        self._audio_queue.put_nowait(None)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            ssml = self._construct_ssml(text)

            # Start synthesis
            self._speech_synthesizer.speak_ssml_async(ssml)

            await self.start_tts_usage_metrics(text)

            # Stream audio chunks as they arrive
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:  # End of stream
                    break

                await self.stop_ttfb_metrics()

                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._settings["sample_rate"],
                    num_channels=1,
                )

            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(f"{self} error: {str(e)}")


class AzureHttpTTSService(AzureBaseTTSService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(
            subscription=self._api_key,
            region=self._region,
            speech_recognition_language=self._settings["language"],
        )
        speech_config.set_speech_synthesis_output_format(
            sample_rate_to_output_format(self._settings["sample_rate"])
        )

        self._speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()

        ssml = self._construct_ssml(text)

        result = await asyncio.to_thread(self._speech_synthesizer.speak_ssml, ssml)

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
