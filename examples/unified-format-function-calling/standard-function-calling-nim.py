#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.nim import NimLLMService

load_dotenv(override=True)


class NimWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = NimLLMService(
            api_key=os.getenv("NVIDIA_API_KEY"), model="meta/llama-3.3-70b-instruct"
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(NimWeatherBot().run())
