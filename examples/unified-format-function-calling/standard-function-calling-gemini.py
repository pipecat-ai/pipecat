#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from base_function_calling import BaseFunctionCallingHandler
from dotenv import load_dotenv

from pipecat.services.google import GoogleLLMService

load_dotenv(override=True)


class WeatherBot(BaseFunctionCallingHandler):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(WeatherBot().run())
