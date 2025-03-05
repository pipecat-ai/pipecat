#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.together import TogetherLLMService

load_dotenv(override=True)


class TogetherWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = TogetherLLMService(
            api_key=os.getenv("TOGETHER_API_KEY"),
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(TogetherWeatherBot().run())
