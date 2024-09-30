#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from typing import AsyncGenerator, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VisionImageRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService, TTSService

try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
    from google.cloud import texttospeech_v1
    from google.oauth2 import service_account
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        gai.configure(api_key=api_key)
        self._create_client(model)

    def can_generate_metrics(self) -> bool:
        return True

    def _create_client(self, model: str):
        self.set_model_name(model)
        self._client = gai.GenerativeModel(model)

    def _get_messages_from_openai_context(self, context: OpenAILLMContext) -> List[glm.Content]:
        openai_messages = context.get_messages()
        google_messages = []

        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"

            parts = [glm.Part(text=content)]
            if "mime_type" in message:
                parts.append(
                    glm.Part(
                        inline_data=glm.Blob(
                            mime_type=message["mime_type"], data=message["data"].getvalue()
                        )
                    )
                )
            google_messages.append({"role": role, "parts": parts})

        return google_messages

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")

            messages = self._get_messages_from_openai_context(context)

            await self.start_ttfb_metrics()

            response = self._client.generate_content(messages, stream=True)

            await self.stop_ttfb_metrics()

            async for chunk in self._async_generator_wrapper(response):
                try:
                    text = chunk.text
                    await self.push_frame(TextFrame(text))
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logger.debug(
                            f"LLM refused to generate content for safety reasons - {messages}."
                        )
                    else:
                        logger.exception(f"{self} error: {e}")

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            if frame.model is not None:
                logger.debug(f"Switching LLM model to: [{frame.model}]")
                self.set_model_name(frame.model)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)


class GoogleTTSService(TTSService):
    class InputParams(BaseModel):
        pitch: Optional[str] = None
        rate: Optional[str] = None
        volume: Optional[str] = None
        emphasis: Optional[Literal["strong", "moderate", "reduced", "none"]] = None
        language: Optional[str] = "en-US"
        gender: Optional[Literal["male", "female", "neutral"]] = None
        google_style: Optional[Literal["apologetic", "calm", "empathetic", "firm", "lively"]] = None

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        voice_id: str = "en-US-Neural2-A",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._voice_id: str = voice_id
        self._params = params
        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    def _create_client(
        self, credentials: Optional[str], credentials_path: Optional[str]
    ) -> texttospeech_v1.TextToSpeechAsyncClient:
        creds: Optional[service_account.Credentials] = None

        # Create a Google Cloud service account for the Cloud Text-to-Speech API
        # Using either the provided credentials JSON string or the path to a service account JSON
        # file, create a Google Cloud service account and use it to authenticate with the API.
        if credentials:
            # Use provided credentials JSON string
            json_account_info = json.loads(credentials)
            creds = service_account.Credentials.from_service_account_info(json_account_info)
        elif credentials_path:
            # Use service account JSON file if provided
            creds = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            raise ValueError("Either 'credentials' or 'credentials_path' must be provided.")

        return texttospeech_v1.TextToSpeechAsyncClient(credentials=creds)

    def can_generate_metrics(self) -> bool:
        return True

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        # Voice tag
        voice_attrs = [f"name='{self._voice_id}'"]
        if self._params.language:
            voice_attrs.append(f"language='{self._params.language}'")
        if self._params.gender:
            voice_attrs.append(f"gender='{self._params.gender}'")
        ssml += f"<voice {' '.join(voice_attrs)}>"

        # Prosody tag
        prosody_attrs = []
        if self._params.pitch:
            prosody_attrs.append(f"pitch='{self._params.pitch}'")
        if self._params.rate:
            prosody_attrs.append(f"rate='{self._params.rate}'")
        if self._params.volume:
            prosody_attrs.append(f"volume='{self._params.volume}'")

        if prosody_attrs:
            ssml += f"<prosody {' '.join(prosody_attrs)}>"

        # Emphasis tag
        if self._params.emphasis:
            ssml += f"<emphasis level='{self._params.emphasis}'>"

        # Google style tag
        if self._params.google_style:
            ssml += f"<google:style name='{self._params.google_style}'>"

        ssml += text

        # Close tags
        if self._params.google_style:
            ssml += "</google:style>"
        if self._params.emphasis:
            ssml += "</emphasis>"
        if prosody_attrs:
            ssml += "</prosody>"
        ssml += "</voice></speak>"

        return ssml

    async def set_voice(self, voice: str) -> None:
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def set_language(self, language: str) -> None:
        logger.debug(f"Switching TTS language to: [{language}]")
        self._params.language = language

    async def set_pitch(self, pitch: str) -> None:
        logger.debug(f"Switching TTS pitch to: [{pitch}]")
        self._params.pitch = pitch

    async def set_rate(self, rate: str) -> None:
        logger.debug(f"Switching TTS rate to: [{rate}]")
        self._params.rate = rate

    async def set_volume(self, volume: str) -> None:
        logger.debug(f"Switching TTS volume to: [{volume}]")
        self._params.volume = volume

    async def set_emphasis(
        self, emphasis: Literal["strong", "moderate", "reduced", "none"]
    ) -> None:
        logger.debug(f"Switching TTS emphasis to: [{emphasis}]")
        self._params.emphasis = emphasis

    async def set_gender(self, gender: Literal["male", "female", "neutral"]) -> None:
        logger.debug(f"Switch TTS gender to [{gender}]")
        self._params.gender = gender

    async def google_style(
        self, google_style: Literal["apologetic", "calm", "empathetic", "firm", "lively"]
    ) -> None:
        logger.debug(f"Switching TTS google style to: [{google_style}]")
        self._params.google_style = google_style

    async def set_params(self, params: InputParams) -> None:
        logger.debug(f"Switching TTS params to: [{params}]")
        self._params = params

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            await self.start_ttfb_metrics()

            ssml = self._construct_ssml(text)
            synthesis_input = texttospeech_v1.SynthesisInput(ssml=ssml)
            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=self._params.language, name=self._voice_id
            )
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
            )

            request = texttospeech_v1.SynthesizeSpeechRequest(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            response = await self._client.synthesize_speech(request=request)

            await self.start_tts_usage_metrics(text)

            await self.push_frame(TTSStartedFrame())

            # Skip the first 44 bytes to remove the WAV header
            audio_content = response.audio_content[44:]

            # Read and yield audio data in chunks
            chunk_size = 8192
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i : i + chunk_size]
                if not chunk:
                    break
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                yield frame
                await asyncio.sleep(0)  # Allow other tasks to run

            await self.push_frame(TTSStoppedFrame())

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)
        finally:
            await self.push_frame(TTSStoppedFrame())
