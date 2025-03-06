#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.grok import GrokLLMService

load_dotenv(override=True)


class GrokWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = GrokLLMService(api_key=os.getenv("GROK_API_KEY"))
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(GrokWeatherBot().run())
