import aiohttp
from dailyai.services.ai_services import LLMService


class GroqLLMService(LLMService):
    def __init__(self, *, api_key, model="llama2-70b-4096"):
        pass
