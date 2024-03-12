from dailyai.services.openai_api_llm_service import BaseOpenAILLMService


class OLLamaLLMService(BaseOpenAILLMService):

    def __init__(self, model="llama2", base_url="http://localhost:11434/v1"):
        super().__init__(model=model, base_url=base_url, api_key="ollama")
