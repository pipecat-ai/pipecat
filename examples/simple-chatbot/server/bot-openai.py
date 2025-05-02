#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Bot Implementation.

This module implements a chatbot using OpenAI's GPT-4 model for natural language
processing. It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Text-to-speech using ElevenLabs
- Support for both English and Spanish

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotInterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMMessagesAppendFrame,
    OutputImageRawFrame,
    SpriteFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    ActionResult,
    RTVIAction,
    RTVIActionArgument,
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIService,
    RTVIServiceConfig,
    RTVIServiceOption,
    RTVIServiceOptionConfig,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

# Load sequential animation frames
for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


class RTVIArgument(BaseModel):
    name: str
    value: Union[str, bool]


class RTVIData(BaseModel):
    action: str
    service: str
    arguments: List[RTVIArgument]


class RTVIMessage(BaseModel):
    id: str
    type: str
    label: str
    data: RTVIData


async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with video/audio parameters
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_width=1024,
                video_out_height=576,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
            ),
        )

        # Initialize text-to-speech service
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            #
            # English
            #
            voice_id="pNInz6obpgDQGcFmaJgB",
            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",
        )

        # Initialize LLM service
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {
                "role": "system",
                #
                # English
                #
                "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
            },
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #

        rtvi = RTVIProcessor()

        rtvi_tts = RTVIService(
            name="tts",
            options=[],
        )

        async def action_tts_say_handler(
            rtvi: RTVIProcessor, service: str, arguments: Dict[str, Any]
        ) -> ActionResult:
            if "interrupt" in arguments and arguments["interrupt"]:
                # interrupting breaks function handling
                await rtvi.interrupt_bot()
            if "text" in arguments:
                save = arguments["save"] if "save" in arguments else False
                frame = TTSSpeakFrame(text=arguments["text"])
                await rtvi.push_frame(frame)
                if save:
                    llm_frame = LLMMessagesAppendFrame(
                        messages=[{"role": "assistant", "content": arguments["text"]}]
                    )
                    await rtvi.push_frame(llm_frame)

            return True

        action_tts_say = RTVIAction(
            service="tts",
            action="say",
            result="bool",
            arguments=[
                RTVIActionArgument(name="text", type="string"),
                RTVIActionArgument(name="save_in_context", type="bool"),
            ],
            handler=action_tts_say_handler,
        )

        async def action_tts_interrupt_handler(
            rtvi: RTVIProcessor, service: str, arguments: Dict[str, Any]
        ) -> ActionResult:
            await rtvi.interrupt_bot()
            return True

        action_tts_interrupt = RTVIAction(
            service="tts", action="interrupt", result="bool", handler=action_tts_interrupt_handler
        )

        rtvi.register_service(rtvi_tts)
        rtvi.register_action(action_tts_say)
        rtvi.register_action(action_tts_interrupt)

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                tts,
                ta,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )
        await task.queue_frame(quiet_frame)

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        # @transport.event_handler("on_app_message")
        # async def on_app_message(transport, message, sender: str):
        #     # Convert the incoming dictionary to an RTVIMessage object

        #     try:
        #         # Parse the dictionary into an RTVIMessage object
        #         rtvi_message = RTVIMessage.model_validate(message)

        #         # Log the parsed message
        #         logger.info(f"Message from {sender}: {rtvi_message.model_dump()}")

        #         if rtvi_message.data.action == "say":
        #             # Extract the text and interrupt values
        #             text = next(
        #                 (arg.value for arg in rtvi_message.data.arguments if arg.name == "text"),
        #                 None,
        #             )
        #             interrupt = next(
        #                 (
        #                     arg.value
        #                     for arg in rtvi_message.data.arguments
        #                     if arg.name == "interrupt"
        #                 ),
        #                 False,
        #             )

        #             if interrupt:
        #                 await task.queue_frame(BotInterruptionFrame())

        #             if text:
        #                 await task.queue_frame(TTSSpeakFrame(text=text))

        #     except Exception as e:
        #         logger.error(f"Failed to parse message: {e}")

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
