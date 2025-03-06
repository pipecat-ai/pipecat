#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.deepseek import DeepSeekLLMService

load_dotenv(override=True)


class DeepSeekWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = DeepSeekLLMService(api_key=os.getenv("DEEPSEEK_API_KEY"), model="deepseek-chat")
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(DeepSeekWeatherBot().run())
