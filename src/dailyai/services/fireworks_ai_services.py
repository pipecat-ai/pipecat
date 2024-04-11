import os

from dailyai.services.openai_api_llm_service import BaseOpenAILLMService


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
