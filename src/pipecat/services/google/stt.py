#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Cloud Speech-to-Text V2 service implementation for Pipecat.

This module provides a Google Cloud Speech-to-Text V2 service with streaming
support, enabling real-time speech recognition with features like automatic
punctuation, voice activity detection, and multi-language support.
"""

import asyncio
import json
import os
import time

from pipecat.utils.tracing.service_decorators import traced_stt

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from typing import AsyncGenerator, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from google.api_core.client_options import ClientOptions
    from google.auth import default
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import speech_v2
    from google.cloud.speech_v2.types import cloud_speech
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def language_to_google_stt_language(language: Language) -> Optional[str]:
    """Maps Language enum to Google Speech-to-Text V2 language codes.

    Args:
        language: Language enum value.

    Returns:
        Optional[str]: Google STT language code or None if not supported.
    """
    language_map = {
        # Afrikaans
        Language.AF: "af-ZA",
        Language.AF_ZA: "af-ZA",
        # Albanian
        Language.SQ: "sq-AL",
        Language.SQ_AL: "sq-AL",
        # Amharic
        Language.AM: "am-ET",
        Language.AM_ET: "am-ET",
        # Arabic
        Language.AR: "ar-EG",  # Default to Egypt
        Language.AR_AE: "ar-AE",
        Language.AR_BH: "ar-BH",
        Language.AR_DZ: "ar-DZ",
        Language.AR_EG: "ar-EG",
        Language.AR_IQ: "ar-IQ",
        Language.AR_JO: "ar-JO",
        Language.AR_KW: "ar-KW",
        Language.AR_LB: "ar-LB",
        Language.AR_MA: "ar-MA",
        Language.AR_OM: "ar-OM",
        Language.AR_QA: "ar-QA",
        Language.AR_SA: "ar-SA",
        Language.AR_SY: "ar-SY",
        Language.AR_TN: "ar-TN",
        Language.AR_YE: "ar-YE",
        # Armenian
        Language.HY: "hy-AM",
        Language.HY_AM: "hy-AM",
        # Azerbaijani
        Language.AZ: "az-AZ",
        Language.AZ_AZ: "az-AZ",
        # Basque
        Language.EU: "eu-ES",
        Language.EU_ES: "eu-ES",
        # Bengali
        Language.BN: "bn-IN",  # Default to India
        Language.BN_BD: "bn-BD",
        Language.BN_IN: "bn-IN",
        # Bosnian
        Language.BS: "bs-BA",
        Language.BS_BA: "bs-BA",
        # Bulgarian
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Burmese
        Language.MY: "my-MM",
        Language.MY_MM: "my-MM",
        # Catalan
        Language.CA: "ca-ES",
        Language.CA_ES: "ca-ES",
        # Chinese
        Language.ZH: "cmn-Hans-CN",  # Default to Simplified Chinese
        Language.ZH_CN: "cmn-Hans-CN",
        Language.ZH_HK: "cmn-Hans-HK",
        Language.ZH_TW: "cmn-Hant-TW",
        Language.YUE: "yue-Hant-HK",  # Cantonese
        Language.YUE_CN: "yue-Hant-HK",
        # Croatian
        Language.HR: "hr-HR",
        Language.HR_HR: "hr-HR",
        # Czech
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        # Danish
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        # Dutch
        Language.NL: "nl-NL",  # Default to Netherlands
        Language.NL_BE: "nl-BE",
        Language.NL_NL: "nl-NL",
        # English
        Language.EN: "en-US",  # Default to US
        Language.EN_AU: "en-AU",
        Language.EN_CA: "en-CA",
        Language.EN_GB: "en-GB",
        Language.EN_GH: "en-GH",
        Language.EN_HK: "en-HK",
        Language.EN_IN: "en-IN",
        Language.EN_IE: "en-IE",
        Language.EN_KE: "en-KE",
        Language.EN_NG: "en-NG",
        Language.EN_NZ: "en-NZ",
        Language.EN_PH: "en-PH",
        Language.EN_SG: "en-SG",
        Language.EN_TZ: "en-TZ",
        Language.EN_US: "en-US",
        Language.EN_ZA: "en-ZA",
        # Estonian
        Language.ET: "et-EE",
        Language.ET_EE: "et-EE",
        # Filipino
        Language.FIL: "fil-PH",
        Language.FIL_PH: "fil-PH",
        # Finnish
        Language.FI: "fi-FI",
        Language.FI_FI: "fi-FI",
        # French
        Language.FR: "fr-FR",  # Default to France
        Language.FR_BE: "fr-BE",
        Language.FR_CA: "fr-CA",
        Language.FR_CH: "fr-CH",
        Language.FR_FR: "fr-FR",
        # Galician
        Language.GL: "gl-ES",
        Language.GL_ES: "gl-ES",
        # Georgian
        Language.KA: "ka-GE",
        Language.KA_GE: "ka-GE",
        # German
        Language.DE: "de-DE",  # Default to Germany
        Language.DE_AT: "de-AT",
        Language.DE_CH: "de-CH",
        Language.DE_DE: "de-DE",
        # Greek
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hebrew
        Language.HE: "iw-IL",
        Language.HE_IL: "iw-IL",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Hungarian
        Language.HU: "hu-HU",
        Language.HU_HU: "hu-HU",
        # Icelandic
        Language.IS: "is-IS",
        Language.IS_IS: "is-IS",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        Language.IT_CH: "it-CH",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Javanese
        Language.JV: "jv-ID",
        Language.JV_ID: "jv-ID",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Kazakh
        Language.KK: "kk-KZ",
        Language.KK_KZ: "kk-KZ",
        # Khmer
        Language.KM: "km-KH",
        Language.KM_KH: "km-KH",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Lao
        Language.LO: "lo-LA",
        Language.LO_LA: "lo-LA",
        # Latvian
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        # Lithuanian
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        # Macedonian
        Language.MK: "mk-MK",
        Language.MK_MK: "mk-MK",
        # Malay
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Mongolian
        Language.MN: "mn-MN",
        Language.MN_MN: "mn-MN",
        # Nepali
        Language.NE: "ne-NP",
        Language.NE_NP: "ne-NP",
        # Norwegian
        Language.NO: "no-NO",
        Language.NB: "no-NO",
        Language.NB_NO: "no-NO",
        # Persian
        Language.FA: "fa-IR",
        Language.FA_IR: "fa-IR",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-PT",  # Default to Portugal
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Punjabi
        Language.PA: "pa-Guru-IN",
        Language.PA_IN: "pa-Guru-IN",
        # Romanian
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Serbian
        Language.SR: "sr-RS",
        Language.SR_RS: "sr-RS",
        # Sinhala
        Language.SI: "si-LK",
        Language.SI_LK: "si-LK",
        # Slovak
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        # Slovenian
        Language.SL: "sl-SI",
        Language.SL_SI: "sl-SI",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain
        Language.ES_AR: "es-AR",
        Language.ES_BO: "es-BO",
        Language.ES_CL: "es-CL",
        Language.ES_CO: "es-CO",
        Language.ES_CR: "es-CR",
        Language.ES_DO: "es-DO",
        Language.ES_EC: "es-EC",
        Language.ES_ES: "es-ES",
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
        # Sundanese
        Language.SU: "su-ID",
        Language.SU_ID: "su-ID",
        # Swahili
        Language.SW: "sw-TZ",  # Default to Tanzania
        Language.SW_KE: "sw-KE",
        Language.SW_TZ: "sw-TZ",
        # Swedish
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
        # Tamil
        Language.TA: "ta-IN",  # Default to India
        Language.TA_IN: "ta-IN",
        Language.TA_MY: "ta-MY",
        Language.TA_SG: "ta-SG",
        Language.TA_LK: "ta-LK",
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
        Language.UR: "ur-IN",  # Default to India
        Language.UR_IN: "ur-IN",
        Language.UR_PK: "ur-PK",
        # Uzbek
        Language.UZ: "uz-UZ",
        Language.UZ_UZ: "uz-UZ",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
        # Xhosa
        Language.XH: "xh-ZA",
        # Zulu
        Language.ZU: "zu-ZA",
        Language.ZU_ZA: "zu-ZA",
    }

    return language_map.get(language)


class GoogleSTTService(STTService):
    """Google Cloud Speech-to-Text V2 service implementation.

    Provides real-time speech recognition using Google Cloud's Speech-to-Text V2 API
    with streaming support. Handles audio transcription and optional voice activity detection.
    Implements automatic stream reconnection to handle Google's 4-minute streaming limit.

    Parameters:
        InputParams: Configuration parameters for the STT service.
        STREAMING_LIMIT: Google Cloud's streaming limit in milliseconds (4 minutes).

    Raises:
        ValueError: If neither credentials nor credentials_path is provided.
        ValueError: If project ID is not found in credentials.
    """

    # Google Cloud's STT service has a connection time limit of 5 minutes per stream.
    # They've shared an "endless streaming" example that guided this implementation:
    # https://cloud.google.com/speech-to-text/docs/transcribe-streaming-audio#endless-streaming

    STREAMING_LIMIT = 240000  # 4 minutes in milliseconds

    class InputParams(BaseModel):
        """Configuration parameters for Google Speech-to-Text.

        Parameters:
            languages: Single language or list of recognition languages. First language is primary.
            model: Speech recognition model to use.
            use_separate_recognition_per_channel: Process each audio channel separately.
            enable_automatic_punctuation: Add punctuation to transcripts.
            enable_spoken_punctuation: Include spoken punctuation in transcript.
            enable_spoken_emojis: Include spoken emojis in transcript.
            profanity_filter: Filter profanity from transcript.
            enable_word_time_offsets: Include timing information for each word.
            enable_word_confidence: Include confidence scores for each word.
            enable_interim_results: Stream partial recognition results.
            enable_voice_activity_events: Detect voice activity in audio.
        """

        languages: Union[Language, List[Language]] = Field(default_factory=lambda: [Language.EN_US])
        model: Optional[str] = "latest_long"
        use_separate_recognition_per_channel: Optional[bool] = False
        enable_automatic_punctuation: Optional[bool] = True
        enable_spoken_punctuation: Optional[bool] = False
        enable_spoken_emojis: Optional[bool] = False
        profanity_filter: Optional[bool] = False
        enable_word_time_offsets: Optional[bool] = False
        enable_word_confidence: Optional[bool] = False
        enable_interim_results: Optional[bool] = True
        enable_voice_activity_events: Optional[bool] = False

        @field_validator("languages", mode="before")
        @classmethod
        def validate_languages(cls, v) -> List[Language]:
            """Ensure languages is always a list.

            Args:
                v: Single Language enum or list of Language enums.

            Returns:
                List[Language]: List of configured languages.
            """
            if isinstance(v, Language):
                return [v]
            return v

        @property
        def language_list(self) -> List[Language]:
            """Get languages as a guaranteed list.

            Returns:
                List[Language]: List of configured languages.
            """
            assert isinstance(self.languages, list)
            return self.languages

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: str = "global",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Google STT service.

        Args:
            credentials: JSON string containing Google Cloud service account credentials.
            credentials_path: Path to service account credentials JSON file.
            location: Google Cloud location (e.g., "global", "us-central1").
            sample_rate: Audio sample rate in Hertz.
            params: Configuration parameters for the service.
            **kwargs: Additional arguments passed to STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GoogleSTTService.InputParams()

        self._location = location
        self._stream = None
        self._config = None
        self._streaming_task = None

        # Used for keep-alive logic
        self._stream_start_time = 0
        self._last_audio_input = []
        self._audio_input = []
        self._result_end_time = 0
        self._is_final_end_time = 0
        self._final_request_end_time = 0
        self._bridging_offset = 0
        self._last_transcript_was_final = False
        self._new_stream = True
        self._restart_counter = 0

        # Configure client options based on location
        client_options = None
        if self._location != "global":
            client_options = ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")

        # Extract project ID and create client
        creds: Optional[service_account.Credentials] = None
        if credentials:
            json_account_info = json.loads(credentials)
            self._project_id = json_account_info.get("project_id")
            creds = service_account.Credentials.from_service_account_info(json_account_info)
        elif credentials_path:
            with open(credentials_path) as f:
                json_account_info = json.load(f)
                self._project_id = json_account_info.get("project_id")
            creds = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            try:
                creds, project_id = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self._project_id = project_id
            except GoogleAuthError:
                pass

        if not creds:
            raise ValueError("No valid credentials provided.")

        if not self._project_id:
            raise ValueError("Project ID not found in credentials")

        self._client = speech_v2.SpeechAsyncClient(credentials=creds, client_options=client_options)

        self._settings = {
            "language_codes": [
                self.language_to_service_language(lang) for lang in params.language_list
            ],
            "model": params.model,
            "use_separate_recognition_per_channel": params.use_separate_recognition_per_channel,
            "enable_automatic_punctuation": params.enable_automatic_punctuation,
            "enable_spoken_punctuation": params.enable_spoken_punctuation,
            "enable_spoken_emojis": params.enable_spoken_emojis,
            "profanity_filter": params.profanity_filter,
            "enable_word_time_offsets": params.enable_word_time_offsets,
            "enable_word_confidence": params.enable_word_confidence,
            "enable_interim_results": params.enable_interim_results,
            "enable_voice_activity_events": params.enable_voice_activity_events,
        }

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: True, as this service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language | List[Language]) -> str | List[str]:
        """Convert Language enum(s) to Google STT language code(s).

        Args:
            language: Single Language enum or list of Language enums.

        Returns:
            str | List[str]: Google STT language code(s).
        """
        if isinstance(language, list):
            return [language_to_google_stt_language(lang) or "en-US" for lang in language]
        return language_to_google_stt_language(language) or "en-US"

    async def _reconnect_if_needed(self):
        """Reconnect the stream if it's currently active."""
        if self._streaming_task:
            logger.debug("Reconnecting stream due to configuration changes")
            await self._disconnect()
            await self._connect()

    async def set_language(self, language: Language):
        """Update the service's recognition language.

        A convenience method for setting a single language.

        Args:
            language: New language for recognition.
        """
        logger.debug(f"Switching STT language to: {language}")
        await self.set_languages([language])

    async def set_languages(self, languages: List[Language]):
        """Update the service's recognition languages.

        Args:
            languages: List of languages for recognition. First language is primary.
        """
        logger.debug(f"Switching STT languages to: {languages}")
        self._settings["language_codes"] = [
            self.language_to_service_language(lang) for lang in languages
        ]
        # Recreate stream with new languages
        await self._reconnect_if_needed()

    async def set_model(self, model: str):
        """Update the service's recognition model.

        Args:
            model: The new recognition model to use.
        """
        logger.debug(f"Switching STT model to: {model}")
        await super().set_model(model)
        self._settings["model"] = model
        # Recreate stream with new model
        await self._reconnect_if_needed()

    async def start(self, frame: StartFrame):
        """Start the STT service and establish connection.

        Args:
            frame: The start frame triggering the service start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and clean up resources.

        Args:
            frame: The end frame triggering the service stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and clean up resources.

        Args:
            frame: The cancel frame triggering the service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def update_options(
        self,
        *,
        languages: Optional[List[Language]] = None,
        model: Optional[str] = None,
        enable_automatic_punctuation: Optional[bool] = None,
        enable_spoken_punctuation: Optional[bool] = None,
        enable_spoken_emojis: Optional[bool] = None,
        profanity_filter: Optional[bool] = None,
        enable_word_time_offsets: Optional[bool] = None,
        enable_word_confidence: Optional[bool] = None,
        enable_interim_results: Optional[bool] = None,
        enable_voice_activity_events: Optional[bool] = None,
        location: Optional[str] = None,
    ) -> None:
        """Update service options dynamically.

        Args:
            languages: New list of recognition languages.
            model: New recognition model.
            enable_automatic_punctuation: Enable/disable automatic punctuation.
            enable_spoken_punctuation: Enable/disable spoken punctuation.
            enable_spoken_emojis: Enable/disable spoken emojis.
            profanity_filter: Enable/disable profanity filter.
            enable_word_time_offsets: Enable/disable word timing info.
            enable_word_confidence: Enable/disable word confidence scores.
            enable_interim_results: Enable/disable interim results.
            enable_voice_activity_events: Enable/disable voice activity detection.
            location: New Google Cloud location.

        Note:
            Changes that affect the streaming configuration will cause
            the stream to be reconnected.
        """
        # Update settings with new values
        if languages is not None:
            logger.debug(f"Updating language to: {languages}")
            self._settings["language_codes"] = [
                self.language_to_service_language(lang) for lang in languages
            ]

        if model is not None:
            logger.debug(f"Updating model to: {model}")
            self._settings["model"] = model

        if enable_automatic_punctuation is not None:
            logger.debug(f"Updating automatic punctuation to: {enable_automatic_punctuation}")
            self._settings["enable_automatic_punctuation"] = enable_automatic_punctuation

        if enable_spoken_punctuation is not None:
            logger.debug(f"Updating spoken punctuation to: {enable_spoken_punctuation}")
            self._settings["enable_spoken_punctuation"] = enable_spoken_punctuation

        if enable_spoken_emojis is not None:
            logger.debug(f"Updating spoken emojis to: {enable_spoken_emojis}")
            self._settings["enable_spoken_emojis"] = enable_spoken_emojis

        if profanity_filter is not None:
            logger.debug(f"Updating profanity filter to: {profanity_filter}")
            self._settings["profanity_filter"] = profanity_filter

        if enable_word_time_offsets is not None:
            logger.debug(f"Updating word time offsets to: {enable_word_time_offsets}")
            self._settings["enable_word_time_offsets"] = enable_word_time_offsets

        if enable_word_confidence is not None:
            logger.debug(f"Updating word confidence to: {enable_word_confidence}")
            self._settings["enable_word_confidence"] = enable_word_confidence

        if enable_interim_results is not None:
            logger.debug(f"Updating interim results to: {enable_interim_results}")
            self._settings["enable_interim_results"] = enable_interim_results

        if enable_voice_activity_events is not None:
            logger.debug(f"Updating voice activity events to: {enable_voice_activity_events}")
            self._settings["enable_voice_activity_events"] = enable_voice_activity_events

        if location is not None:
            logger.debug(f"Updating location to: {location}")
            self._location = location

        # Reconnect the stream for updates
        await self._reconnect_if_needed()

    async def _connect(self):
        """Initialize streaming recognition config and stream."""
        logger.debug("Connecting to Google Speech-to-Text")

        # Set stream start time
        self._stream_start_time = int(time.time() * 1000)
        self._new_stream = True

        self._config = cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=self._settings["language_codes"],
                model=self._settings["model"],
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=self._settings["enable_automatic_punctuation"],
                    enable_spoken_punctuation=self._settings["enable_spoken_punctuation"],
                    enable_spoken_emojis=self._settings["enable_spoken_emojis"],
                    profanity_filter=self._settings["profanity_filter"],
                    enable_word_time_offsets=self._settings["enable_word_time_offsets"],
                    enable_word_confidence=self._settings["enable_word_confidence"],
                ),
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                enable_voice_activity_events=self._settings["enable_voice_activity_events"],
                interim_results=self._settings["enable_interim_results"],
            ),
        )

        self._request_queue = asyncio.Queue()
        self._streaming_task = self.create_task(self._stream_audio())

    async def _disconnect(self):
        """Clean up streaming recognition resources."""
        if self._streaming_task:
            logger.debug("Disconnecting from Google Speech-to-Text")
            await self.cancel_task(self._streaming_task)
            self._streaming_task = None

    async def _request_generator(self):
        """Generates requests for the streaming recognize method."""
        recognizer_path = f"projects/{self._project_id}/locations/{self._location}/recognizers/_"
        logger.trace(f"Using recognizer path: {recognizer_path}")

        try:
            # Send initial config
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer_path,
                streaming_config=self._config,
            )

            while True:
                audio_data = await self._request_queue.get()

                self._request_queue.task_done()

                # Check streaming limit
                if (int(time.time() * 1000) - self._stream_start_time) > self.STREAMING_LIMIT:
                    logger.debug("Streaming limit reached, initiating graceful reconnection")
                    # Instead of immediate reconnection, we'll break and let the stream close naturally
                    self._last_audio_input = self._audio_input
                    self._audio_input = []
                    self._restart_counter += 1
                    # Put the current audio chunk back in the queue
                    await self._request_queue.put(audio_data)
                    break

                self._audio_input.append(audio_data)
                yield cloud_speech.StreamingRecognizeRequest(audio=audio_data)

        except Exception as e:
            logger.error(f"Error in request generator: {e}")
            raise

    async def _stream_audio(self):
        """Handle bi-directional streaming with Google STT."""
        try:
            while True:
                try:
                    if self._request_queue.empty():
                        # wait for 10ms in case we don't have audio
                        await asyncio.sleep(0.01)
                        continue

                    # Start bi-directional streaming
                    streaming_recognize = await self._client.streaming_recognize(
                        requests=self._request_generator()
                    )

                    # Process responses
                    await self._process_responses(streaming_recognize)

                    # If we're here, check if we need to reconnect
                    if (int(time.time() * 1000) - self._stream_start_time) > self.STREAMING_LIMIT:
                        logger.debug("Reconnecting stream after timeout")
                        # Reset stream start time
                        self._stream_start_time = int(time.time() * 1000)
                    else:
                        # Normal stream end
                        break

                except Exception as e:
                    logger.warning(f"{self} Reconnecting: {e}")

                    await asyncio.sleep(1)  # Brief delay before reconnecting
                    self._stream_start_time = int(time.time() * 1000)

        except Exception as e:
            logger.error(f"Error in streaming task: {e}")
            await self.push_frame(ErrorFrame(str(e)))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (actual transcription frames are pushed via internal processing).
        """
        if self._streaming_task:
            # Queue the audio data
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            await self._request_queue.put(audio)
        yield None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        pass

    async def _process_responses(self, streaming_recognize):
        """Process streaming recognition responses."""
        try:
            async for response in streaming_recognize:
                # Check streaming limit
                if (int(time.time() * 1000) - self._stream_start_time) > self.STREAMING_LIMIT:
                    logger.debug("Stream timeout reached in response processing")
                    break

                if not response.results:
                    continue

                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript
                    if not transcript:
                        continue

                    primary_language = self._settings["language_codes"][0]

                    if result.is_final:
                        self._last_transcript_was_final = True
                        await self.push_frame(
                            TranscriptionFrame(
                                transcript,
                                self._user_id,
                                time_now_iso8601(),
                                primary_language,
                                result=result,
                            )
                        )
                        await self.stop_processing_metrics()
                        await self._handle_transcription(
                            transcript,
                            is_final=True,
                            language=primary_language,
                        )
                    else:
                        self._last_transcript_was_final = False
                        await self.stop_ttfb_metrics()
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                transcript,
                                self._user_id,
                                time_now_iso8601(),
                                primary_language,
                                result=result,
                            )
                        )
        except Exception as e:
            logger.error(f"Error processing Google STT responses: {e}")
            # Re-raise the exception to let it propagate (e.g. in the case of a
            # timeout, propagate to _stream_audio to reconnect)
            raise
