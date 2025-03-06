#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from multimodal_base_function_calling import MultimodalWeatherBot

from pipecat.services.openai_realtime_beta import (
    InputAudioTranscription,
    OpenAIRealtimeBetaLLMService,
    SessionProperties,
    TurnDetection,
)

load_dotenv(override=True)


class OpenAiRealTimeWeatherBot(MultimodalWeatherBot):
    """Main class defining the LLM and passing it to the base handler."""

    def __init__(self):
        session_properties = SessionProperties(
            input_audio_transcription=InputAudioTranscription(),
            # Set openai TurnDetection parameters. Not setting this at all will turn it
            # on by default
            turn_detection=TurnDetection(silence_duration_ms=1000),
        )

        llm = OpenAIRealtimeBetaLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            session_properties=session_properties,
            start_audio_paused=False,
        )
        super().__init__(llm)


if __name__ == "__main__":
    asyncio.run(OpenAiRealTimeWeatherBot().run())
