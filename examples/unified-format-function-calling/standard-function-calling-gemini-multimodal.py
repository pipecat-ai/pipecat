#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from multimodal_base_function_calling import MultimodalBaseFunctionCallingHandler
from pipecat.adapters.schemas.tools_schema import AdapterType

from pipecat.services.gemini_multimodal_live import GeminiMultimodalLiveLLMService

load_dotenv(override=True)


class WeatherBot(MultimodalBaseFunctionCallingHandler):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        search_tool = {"google_search": {}}
        tools_def = MultimodalBaseFunctionCallingHandler.tools()
        tools_def.custom_tools = {AdapterType.GEMINI: [search_tool]}

        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            voice_id="Puck",
            transcribe_user_audio=True,
            transcribe_model_audio=True,
            tools=tools_def,
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(WeatherBot().run())
