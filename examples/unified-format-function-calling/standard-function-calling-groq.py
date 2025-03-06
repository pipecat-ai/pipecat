#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.groq import GroqLLMService

load_dotenv(override=True)


class GroqWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(GroqWeatherBot().run())
