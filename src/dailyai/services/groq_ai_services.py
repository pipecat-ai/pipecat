import os
import groq
from groq import AsyncGroq
from dailyai.services.ai_services import LLMService
from collections.abc import AsyncGenerator


class GroqLLMService(LLMService):
    def __init__(self, *, api_key, model="mixtral-8x7b-32768", context):
        super().__init__(context)
        self._model = model
        # os.environ["GROQ_SECRET_ACCESS_KEY"] = api_key
        
        self._client = AsyncGroq()

    async def run_llm_async(self, messages) -> AsyncGenerator[str, None]:
        try:
            resp = await self._client.chat.completions.create(messages=messages, model=self._model)

            if resp.choices[0].message.content:
                yield resp.choices[0].message.content
        except groq.APIConnectionError as e:
            self._logger.error("The server could not be reached")
            self._logger.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except groq.RateLimitError as e:
            self._logger.error("A 429 status code was received; we should back off a bit.")
        except groq.APIStatusError as e:
            self._logger.error("Another non-200-range status code was received")
            self._logger.error(e.status_code)
            self._logger.error(e.response)
    