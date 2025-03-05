#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.fireworks import FireworksLLMService

load_dotenv(override=True)


class FireworksWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = FireworksLLMService(
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(FireworksWeatherBot().run())
