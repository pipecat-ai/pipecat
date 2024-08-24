import os

from .openai import BaseOpenAILLMService


class GroqLLMService(BaseOpenAILLMService):
    def __init__(self, *, model: str = "llama-3.1-8b-instant",
                 base_url: str = "https://api.groq.com/openai/v1",
                 api_key: str = os.get_env("GROQ_API_KEY")):
        super().__init__(model=model, base_url=base_url, api_key=api_key)
