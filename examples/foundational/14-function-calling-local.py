#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# This example demonstrates how to create an interruptible PURELY LOCAL audio pipeline using Pipecat 
# with the ability to call a function that fake checks the weather in a given location.
# It uses the Moonshine ASR for speech-to-text, Kokoro for text-to-speech, and Ollama for LLM.
# The pipeline is designed to be interruptible, allowing for real-time interaction with the user.
#
import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.moonshine.stt import MoonshineSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def fetch_weather(function_name, tool_call_id, args, llm, context, result_callback):
    """Get the current weather for a city using the wttr.in API.
    This function is registered with the LLM and will be called when the LLM
    generates a function call with the name "get_current_weather".

    Args:
        location: The location to get the weather for.

    Returns:
        A dictionary with the current conditions and temperature for the location.
    """    
    await llm.push_frame(TTSSpeakFrame("Let me check on that."))
    logger.info(f"Getting weather for {args['location']}")  
    location = args["location"]
    temperature = 20 
    conditions = "partly cloudy but mild" 
    logger.info(f"Conditions for {location}: {conditions}, temperature: {temperature}") 
    await result_callback({"conditions": conditions, "temperature": temperature})    

    
async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    stt = MoonshineSTTService(
        model_name="moonshine/tiny",
        language="en",
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )

    tts = KokoroTTSService(
        model_path="assets/kokoro-v1.0.onnx",
        voices_path="assets/voices-v1.0.bin",
        voice_id="am_fenrir",
    )

    # Initialize the LLM service
    # Make sure to have the Ollama server running locally
    # You can run it with the command: ollama serve --model llama3.1
    # Make sure to have the model downloaded with: ollama pull llama3.1
    llm = OLLamaLLMService(
        model="llama3.1",
        base_url="http://localhost:11434/v1",
    )

    # register the weather function with the LLM
    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("get_current_weather", fetch_weather)

    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location", "format"],
    )
    tools = ToolsSchema(standard_tools=[weather_function])                    

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    messages.append({"role": "system", "content": "Please introduce yourself to the user."})
    await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
