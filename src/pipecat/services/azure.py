#
# Copyright (c) 2024–2025, Daily
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
from pipecat.services.ai_services import ImageGenService, STTService, TTSService
from pipecat.services.openai import (
    OpenAILLMService,
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
        # Afrikaans
        Language.AF: "af-ZA",
        Language.AF_ZA: "af-ZA",
        # Amharic
        Language.AM: "am-ET",
        Language.AM_ET: "am-ET",
        # Arabic
        Language.AR: "ar-AE",  # Default to UAE Arabic
        Language.AR_AE: "ar-AE",
        Language.AR_BH: "ar-BH",
        Language.AR_DZ: "ar-DZ",
        Language.AR_EG: "ar-EG",
        Language.AR_IQ: "ar-IQ",
        Language.AR_JO: "ar-JO",
        Language.AR_KW: "ar-KW",
        Language.AR_LB: "ar-LB",
        Language.AR_LY: "ar-LY",
        Language.AR_MA: "ar-MA",
        Language.AR_OM: "ar-OM",
        Language.AR_QA: "ar-QA",
        Language.AR_SA: "ar-SA",
        Language.AR_SY: "ar-SY",
        Language.AR_TN: "ar-TN",
        Language.AR_YE: "ar-YE",
        # Assamese
        Language.AS: "as-IN",
        Language.AS_IN: "as-IN",
        # Azerbaijani
        Language.AZ: "az-AZ",
        Language.AZ_AZ: "az-AZ",
        # Bulgarian
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Bengali
        Language.BN: "bn-IN",  # Default to Indian Bengali
        Language.BN_BD: "bn-BD",
        Language.BN_IN: "bn-IN",
        # Bosnian
        Language.BS: "bs-BA",
        Language.BS_BA: "bs-BA",
        # Catalan
        Language.CA: "ca-ES",
        Language.CA_ES: "ca-ES",
        # Czech
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        # Welsh
        Language.CY: "cy-GB",
        Language.CY_GB: "cy-GB",
        # Danish
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        # German
        Language.DE: "de-DE",
        Language.DE_AT: "de-AT",
        Language.DE_CH: "de-CH",
        Language.DE_DE: "de-DE",
        # Greek
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        # English
        Language.EN: "en-US",  # Default to US English
        Language.EN_AU: "en-AU",
        Language.EN_CA: "en-CA",
        Language.EN_GB: "en-GB",
        Language.EN_HK: "en-HK",
        Language.EN_IE: "en-IE",
        Language.EN_IN: "en-IN",
        Language.EN_KE: "en-KE",
        Language.EN_NG: "en-NG",
        Language.EN_NZ: "en-NZ",
        Language.EN_PH: "en-PH",
        Language.EN_SG: "en-SG",
        Language.EN_TZ: "en-TZ",
        Language.EN_US: "en-US",
        Language.EN_ZA: "en-ZA",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain Spanish
        Language.ES_AR: "es-AR",
        Language.ES_BO: "es-BO",
        Language.ES_CL: "es-CL",
        Language.ES_CO: "es-CO",
        Language.ES_CR: "es-CR",
        Language.ES_CU: "es-CU",
        Language.ES_DO: "es-DO",
        Language.ES_EC: "es-EC",
        Language.ES_ES: "es-ES",
        Language.ES_GQ: "es-GQ",
        Language.ES_GT: "es-GT",
        Language.ES_HN: "es-HN",
        Language.ES_MX: "es-MX",
        Language.ES_NI: "es-NI",
        Language.ES_PA: "es-PA",
        Language.ES_PE: "es-PE",
        Language.ES_PR: "es-PR",
        Language.ES_PY: "es-PY",
        Language.ES_SV: "es-SV",
        Language.ES_US: "es-US",
        Language.ES_UY: "es-UY",
        Language.ES_VE: "es-VE",
        # Estonian
        Language.ET: "et-EE",
        Language.ET_EE: "et-EE",
        # Basque
        Language.EU: "eu-ES",
        Language.EU_ES: "eu-ES",
        # Persian
        Language.FA: "fa-IR",
        Language.FA_IR: "fa-IR",
        # Finnish
        Language.FI: "fi-FI",
        Language.FI_FI: "fi-FI",
        # Filipino
        Language.FIL: "fil-PH",
        Language.FIL_PH: "fil-PH",
        # French
        Language.FR: "fr-FR",
        Language.FR_BE: "fr-BE",
        Language.FR_CA: "fr-CA",
        Language.FR_CH: "fr-CH",
        Language.FR_FR: "fr-FR",
        # Irish
        Language.GA: "ga-IE",
        Language.GA_IE: "ga-IE",
        # Galician
        Language.GL: "gl-ES",
        Language.GL_ES: "gl-ES",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hebrew
        Language.HE: "he-IL",
        Language.HE_IL: "he-IL",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Croatian
        Language.HR: "hr-HR",
        Language.HR_HR: "hr-HR",
        # Hungarian
        Language.HU: "hu-HU",
        Language.HU_HU: "hu-HU",
        # Armenian
        Language.HY: "hy-AM",
        Language.HY_AM: "hy-AM",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Icelandic
        Language.IS: "is-IS",
        Language.IS_IS: "is-IS",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Inuktitut
        Language.IU_CANS_CA: "iu-Cans-CA",
        Language.IU_LATN_CA: "iu-Latn-CA",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Javanese
        Language.JV: "jv-ID",
        Language.JV_ID: "jv-ID",
        # Georgian
        Language.KA: "ka-GE",
        Language.KA_GE: "ka-GE",
        # Kazakh
        Language.KK: "kk-KZ",
        Language.KK_KZ: "kk-KZ",
        # Khmer
        Language.KM: "km-KH",
        Language.KM_KH: "km-KH",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Lao
        Language.LO: "lo-LA",
        Language.LO_LA: "lo-LA",
        # Lithuanian
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        # Latvian
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        # Macedonian
        Language.MK: "mk-MK",
        Language.MK_MK: "mk-MK",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Mongolian
        Language.MN: "mn-MN",
        Language.MN_MN: "mn-MN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Malay
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        # Maltese
        Language.MT: "mt-MT",
        Language.MT_MT: "mt-MT",
        # Burmese
        Language.MY: "my-MM",
        Language.MY_MM: "my-MM",
        # Norwegian
        Language.NB: "nb-NO",
        Language.NB_NO: "nb-NO",
        Language.NO: "nb-NO",
        # Nepali
        Language.NE: "ne-NP",
        Language.NE_NP: "ne-NP",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_BE: "nl-BE",
        Language.NL_NL: "nl-NL",
        # Odia
        Language.OR: "or-IN",
        Language.OR_IN: "or-IN",
        # Punjabi
        Language.PA: "pa-IN",
        Language.PA_IN: "pa-IN",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Pashto
        Language.PS: "ps-AF",
        Language.PS_AF: "ps-AF",
        # Portuguese
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Romanian
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Sinhala
        Language.SI: "si-LK",
        Language.SI_LK: "si-LK",
        # Slovak
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        # Slovenian
        Language.SL: "sl-SI",
        Language.SL_SI: "sl-SI",
        # Somali
        Language.SO: "so-SO",
        Language.SO_SO: "so-SO",
        # Albanian
        Language.SQ: "sq-AL",
        Language.SQ_AL: "sq-AL",
        # Serbian
        Language.SR: "sr-RS",
        Language.SR_RS: "sr-RS",
        Language.SR_LATN: "sr-Latn-RS",
        Language.SR_LATN_RS: "sr-Latn-RS",
        # Sundanese
        Language.SU: "su-ID",
        Language.SU_ID: "su-ID",
        # Swedish
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
        # Swahili
        Language.SW: "sw-KE",
        Language.SW_KE: "sw-KE",
        Language.SW_TZ: "sw-TZ",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        Language.TA_LK: "ta-LK",
        Language.TA_MY: "ta-MY",
        Language.TA_SG: "ta-SG",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Ukrainian
        Language.UK: "uk-UA",
        Language.UK_UA: "uk-UA",
        # Urdu
        Language.UR: "ur-IN",
        Language.UR_IN: "ur-IN",
        Language.UR_PK: "ur-PK",
        # Uzbek
        Language.UZ: "uz-UZ",
        Language.UZ_UZ: "uz-UZ",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
        # Wu Chinese
        Language.WUU: "wuu-CN",
        Language.WUU_CN: "wuu-CN",
        # Yue Chinese
        Language.YUE: "yue-CN",
        Language.YUE_CN: "yue-CN",
        # Chinese
        Language.ZH: "zh-CN",
        Language.ZH_CN: "zh-CN",
        Language.ZH_CN_GUANGXI: "zh-CN-guangxi",
        Language.ZH_CN_HENAN: "zh-CN-henan",
        Language.ZH_CN_LIAONING: "zh-CN-liaoning",
        Language.ZH_CN_SHAANXI: "zh-CN-shaanxi",
        Language.ZH_CN_SHANDONG: "zh-CN-shandong",
        Language.ZH_CN_SICHUAN: "zh-CN-sichuan",
        Language.ZH_HK: "zh-HK",
        Language.ZH_TW: "zh-TW",
        # Zulu
        Language.ZU: "zu-ZA",
        Language.ZU_ZA: "zu-ZA",
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
    return sample_rate_map.get(sample_rate, SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)


class AzureLLMService(OpenAILLMService):
    """A service for interacting with Azure OpenAI using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Azure's OpenAI endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing Azure OpenAI
        endpoint (str): The Azure endpoint URL
        model (str): The model identifier to use
        api_version (str, optional): Azure API version. Defaults to "2024-09-01-preview"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model: str,
        api_version: str = "2024-09-01-preview",
        **kwargs,
    ):
        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Azure OpenAI endpoint."""
        logger.debug(f"Creating Azure OpenAI client with endpoint {self._endpoint}")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )


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
