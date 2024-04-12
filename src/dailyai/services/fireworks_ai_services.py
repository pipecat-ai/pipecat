import aiohttp
import os

from typing import Literal

from dailyai.services.ai_services import ImageGenService, VisionService
from dailyai.services.openai_api_llm_service import BaseOpenAILLMService
from dailyai.services.open_ai_services import OpenAIVisionService

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use Fireworks, you need to `pip install dailyai[fireworks]`. Also, set the `FIREWORKS_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class FireworksLLMService(BaseOpenAILLMService):
    def __init__(self, model="accounts/fireworks/models/firefunction-v1", *args, **kwargs):
        kwargs["base_url"] = "https://api.fireworks.ai/inference/v1"
        super().__init__(model, *args, **kwargs)



class FireworksVisionService(OpenAIVisionService):
    def __init__(self, *, api_key, model="accounts/fireworks/models/firellava-13b"):
        super().__init__(model=model, api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
