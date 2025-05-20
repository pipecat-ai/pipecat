#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Author : Aneesh

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import WordTTSService
from pipecat.transcriptions.language import Language
# from pipecat.utils.tracing.service_decorators import traced_tts

# Sarvam AI Language Mapping
# Based on https://docs.sarvam.ai/api-reference-docs/text-to-speech/ap-is/overview
SARVAM_SUPPORTED_LANGUAGES = {
    Language.EN: "en-IN",      # English
    Language.HI: "hi-IN",      # Hindi
    Language.BN: "bn-IN",      # Bengali
    Language.MR: "mr-IN",      # Marathi
    Language.TE: "te-IN",      # Telugu
    Language.TA: "ta-IN",      # Tamil
    Language.GU: "gu-IN",      # Gujarati
    Language.KN: "kn-IN",      # Kannada
    Language.PA: "pa-IN",      # punjabi
    Language.OR: "or-IN",      # Odia
    Language.ML: "ml-IN",      # Malayalam
}


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Converts a pipecat.transcriptions.language.Language enum to a Sarvam AI language code string."""
    result = SARVAM_SUPPORTED_LANGUAGES.get(language)
    # logger.debug(f"result in sarvam language is : {result}")
    if not result:
        # Try to find the base language from a variant (e.g., en-US -> en)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Check if this base_code is one of the Sarvam supported language codes
        if base_code in SARVAM_SUPPORTED_LANGUAGES.values():
            result = base_code
        else:
            logger.warning(f"Unsupported language [{language}] for Sarvam AI. No mapping found for base code [{base_code}].")
    return result


class SarvamAITTSService(WordTTSService):
    """Sarvam AI Text-to-Speech service using HTTP streaming with word timestamps.

    This service converts text to speech using Sarvam AI's API, supporting
    streaming audio output and word-level timestamps.

    Args:
        api_key: Your Sarvam AI API key.
        voice: The voice ID to use for speech synthesis (e.g., "anushka").
        aiohttp_session: An `aiohttp.ClientSession` instance for making HTTP requests.
        model: Optional model identifier. If None, it defaults to the `voice` ID.
                  This is for future compatibility or specific model selection if Sarvam AI API evolves.
        base_url: The base URL for the Sarvam AI API. Defaults to "https://api.sarvam.ai".
        sample_rate: The desired audio sample rate in Hz. Sarvam AI supports 24000 and 48000 Hz.
                     Defaults to 24000 Hz.
        params: An `InputParams` Pydantic model instance for additional TTS configuration
                like language, speed, output format, and encoding.
    """

    class InputParams(BaseModel):
        language: Optional[Language] = "en-IN"  # Language of the input text.
        pitch: Optional[float] = 0.0  # Controls the pitch of the audio. Lower values result in a deeper voice, while higher values make it sharper. The suitable range is between -0.75 and 0.75. Default is 0.0.
        pace: Optional[float] = 1.0   # Controls the speed of the audio. Lower values result in slower speech, while higher values make it faster. The suitable range is between 0.5 and 2.0. Default is 1.0.
        loudness: Optional[float] = 1.0 # Controls the loudness of the audio. Lower values result in quieter audio, while higher values make it louder. The suitable range is between 0.3 and 3.0. Default is 1.0.
        enable_preprocessing: Optional[float] = False # Controls whether normalization of English words and numeric entities (e.g., numbers, dates) is performed. Set to true for better handling of mixed-language text. Default is false.

    def __init__(
        self,
        *,
        api_key: str,
        voice: str,
        aiohttp_session: aiohttp.ClientSession,
        model: Optional[str] = "bulbul:v2",
        base_url: str = "https://api.sarvam.ai/text-to-speech",
        speech_sample_rate: int = 22050, # Default to Sarvam's common sample rate
        params: InputParams = InputParams(),
        **kwargs,
    ):
        # Ensure sample_rate is valid for Sarvam AI before calling super().__init__
        # as WordTTSService stores it.
        if speech_sample_rate not in [8000, 16000, 22050, 24000]:
            logger.warning(
                f"SarvamAITTSService: Unsupported sample_rate {speech_sample_rate}. "
                f"Sarvam AI supports 8000, 16000, 22050, 24000. Defaulting to 22050."
            )
            speech_sample_rate = 22050

        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False, # We will generate text frames from word timestamps
            push_stop_frames=True,  # Let parent class handle TTSStoppedFrame logic
            sample_rate=speech_sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._voice = voice

        self._settings = {
            "language": self.language_to_service_language(params.language) if params.language else "en-IN",
            "pitch": params.pitch,
            "pace": params.pace,
            "loudness": params.loudness,
            "enable_preprocessing": params.enable_preprocessing
        }

        # In Sarvam, 'voice' often implicitly defines the model.
        # `model` could be used if Sarvam distinguishes them more explicitly in the future.
        self.set_voice(voice)
        self.set_model_name(model if model else None) # Use voice as model_name if no explicit model

        self._cumulative_time: float = 0.0
        self._started: bool = False

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Converts a pipecat Language enum to a Sarvam AI language code string."""
        return language_to_sarvam_language(language)

    def can_generate_metrics(self) -> bool:
        """Indicates that this service can generate usage metrics."""
        return True

    def _reset_state(self):
        """Resets internal state variables, typically on interruption or stop."""
        self._cumulative_time = 0.0
        self._started = False
        # logger.debug(f"{self}: Reset internal state.")

    async def start(self, frame: StartFrame):
        """Initializes the service upon receiving a StartFrame."""
        await super().start(frame)
        self._reset_state() # Ensure clean state at the beginning of a new session

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Handles incoming frames, resetting state on interruptions or stops."""
        await super().push_frame(frame, direction)
        if isinstance(frame, (StartInterruptionFrame, TTSStoppedFrame)):
            self._reset_state()
            if isinstance(frame, TTSStoppedFrame):
                # Signal reset of word timestamps on the client side if needed
                await self.add_word_timestamps([("Reset", 0.0)])

    # @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generates speech from text using Sarvam AI's streaming API with word timestamps.

        Args:
            text: The text to convert to speech.

        Yields:
            `TTSAudioRawFrame` containing audio data,
            and control frames like `TTSStartedFrame`.
        """
        logger.debug(
            f"{self}: Generating TTS for text [{text[:100]}{'...' if len(text) > 100 else ''}] "
            f"using voice [{self._voice_id}]"
        )

        url = f"{self._base_url}" # "https://api.sarvam.ai/text-to-speech" # Standard Sarvam TTS endpoint
        
        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
            # "Accept": "application/json", # Expecting JSON streaming response
        }

        # Construct the payload for Sarvam AI
        payload: Dict[str, Any] = {
            "text": text,
            "speaker": self._voice, # self._voice_id, # This is the "Voice ID" in Sarvam terms
            "target_language_code": self._settings["language"], # e.g., "en-IN", "hi-IN"
            "pitch": self._settings["pitch"],
            "pace": self._settings["pace"],
            "loudness": self._settings["loudness"],
            "enable_preprocessing": self._settings["enable_preprocessing"]
            # "output_format": self._settings["output_format"], # e.g., "pcm"
            # "sample_rate": self.sample_rate, # e.g., 24000
            # "word_timestamps": True, # Explicitly request word timestamps
        }
        # logger.debug(f"the payload for the sarvam ai is : {payload}")
        
        
        # Remove None values from payload as Sarvam API might not like nulls for optional fields
        final_payload = {k: v for k, v in payload.items() if v is not None}

        if "target_language_code" not in final_payload and self._settings["language"] is None:
            # Sarvam API usually requires language. Let's log a strong warning.
            # The user must configure a language through params or ensure the voice implies it.
             logger.error(
                 f"{self}: Language is not set. Sarvam AI API requires a language code. "
                 f"Request will likely fail. Please provide a language in InputParams."
            )
             yield ErrorFrame(error="Sarvam AI TTS: Language not specified.")
             return

        try:
            await self.start_ttfb_metrics() # Time To First Byte metric start

            async with self._session.post(url, json=final_payload, headers=headers) as response:
                # logger.debug(f"The response of the sarvam ai is {response}")
                # logger.debug(f"The response json of the sarvam ai is {await response.json()}")
                
                if response.status != 200:
                    yield ErrorFrame(error=f"Sarvam AI API error ({response.status}): {error_detail}")
                    self._reset_state() # Ensure state is reset on error
                    # logger.debug(f"response status is {response.status}")
                    return

                await self.start_tts_usage_metrics(text) # TTS usage metric start

                # Start TTS sequence if not already started for this series of run_tts calls
                if not self._started:
                    self.start_word_timestamps() # Initialize word timestamp tracking
                    yield TTSStartedFrame()
                    self._started = True
                # Process the streaming response (line by line, expecting JSON objects)
                response_audio = await response.json()
                for line in response_audio['audios']:                    
                    try:
                        data = line # Each line is a JSON object

                        await self.stop_ttfb_metrics() # Time To First Byte metric stop
                        audio_bytes = base64.b64decode(data)
                        # Sarvam PCM is typically 1 channel. If this assumption changes,
                        # num_channels might need to be configurable or detected.
                        yield TTSAudioRawFrame(audio_bytes, self.sample_rate, 1) # num_channels = 1

                    except Exception as e:
                        logger.error(f"{self}: Error processing chunk from Sarvam AI stream: {e}", exc_info=True)
                        continue # Skip problematic chunk but continue stream if possible

        except aiohttp.ClientConnectorError as e:
            logger.error(f"{self} connection error to Sarvam AI: {e}")
            yield ErrorFrame(error=f"Sarvam AI connection error: {e}")
            self._reset_state()
        except Exception as e:
            logger.error(f"{self} uncaught exception in run_tts: {e}", exc_info=True)
            yield ErrorFrame(error=f"Sarvam AI TTS error: {str(e)}")
            self._reset_state()
        finally:
            await self.stop_ttfb_metrics()
            # TTSStoppedFrame is handled by the parent class if push_stop_frames is True
            # and an aggregate_sentences timeout occurs or an LLMFullResponseEndFrame is seen.