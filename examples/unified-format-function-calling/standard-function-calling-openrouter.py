#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.openrouter import OpenRouterLLMService

load_dotenv(override=True)


class OpenRouterWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = OpenRouterLLMService(
            api_key=os.getenv("OPENROUTER_API_KEY"), model="openai/gpt-4o-2024-11-20"
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(OpenRouterWeatherBot().run())
