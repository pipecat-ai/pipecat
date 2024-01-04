import logging

from abc import abstractmethod
from collections.abc import AsyncGenerator

from dataclasses import dataclass
from typing import Generator
from PIL import Image


class AIService:
    def __init__(self):
        self.logger = logging.getLogger("dailyai")

    def close(self):
        pass

class LLMService(AIService):
    # Generate a set of responses to a prompt. Yields a list of responses.
    @abstractmethod
    async def run_llm_async(
        self, messages
    ) -> AsyncGenerator[str, None, None]:
        pass

    # Generate a responses to a prompt. Returns the response
    @abstractmethod
    async def run_llm(
        self, messages
    ) -> str or None:
        pass


class TTSService(AIService):
    # Some TTS services require a specific sample rate. We default to 16k
    def get_mic_sample_rate(self):
        return 16000

    # Converts the sentence to audio. Yields a list of audio frames that can
    # be sent to the microphone device
    @abstractmethod
    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None, None]:
        pass


class ImageGenService(AIService):
    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, sentence) -> tuple[str, Image.Image]:
        pass


@dataclass
class AIServiceConfig:
    tts: TTSService
    image: ImageGenService
    llm: LLMService
