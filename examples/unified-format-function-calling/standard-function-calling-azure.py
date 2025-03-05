#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import WeatherBot
from dotenv import load_dotenv

from pipecat.services.azure import AzureLLMService

load_dotenv(override=True)


class AzureWeatherBot(WeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(AzureWeatherBot().run())
