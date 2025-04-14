#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Dict, Literal, Optional

from loguru import logger
from openai import AsyncOpenAI, BadRequestError

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

ValidVoice = Literal[
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"
]

VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "ash": "ash",
    "ballad": "ballad",
    "coral": "coral",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "sage": "sage",
    "shimmer": "shimmer",
    "verse": "verse",
}


class OpenAITTSService(TTSService):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.

    Args:
        api_key: OpenAI API key. Defaults to None.
        voice: Voice ID to use. Defaults to "alloy".
        model: TTS model to use. Defaults to "gpt-4o-mini-tts".
        sample_rate: Output audio sample rate in Hz. Defaults to None.
        **kwargs: Additional keyword arguments passed to TTSService.

    The service returns PCM-encoded audio at the specified sample rate.

    """

    OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
        sample_rate: Optional[int] = None,
        instructions: Optional[str] = None,
        **kwargs,
    ):
        if sample_rate and sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS only supports {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.set_model_name(model)
        self.set_voice(voice)
        self._instructions = instructions
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()

            # Setup extra body parameters
            extra_body = {}
            if self._instructions:
                extra_body["instructions"] = self._instructions

            async with self._client.audio.speech.with_streaming_response.create(
                input=text,
                model=self.model_name,
                voice=VALID_VOICES[self._voice_id],
                response_format="pcm",
                extra_body=extra_body,
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = 1024

                yield TTSStartedFrame()
                async for chunk in r.iter_bytes(CHUNK_SIZE):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        yield frame
                yield TTSStoppedFrame()
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")
