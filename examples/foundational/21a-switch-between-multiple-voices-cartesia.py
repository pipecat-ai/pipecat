import asyncio
import os
import sys
from enum import Enum

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class Character(Enum):
    GIMLI = "Gimli"
    LEGOLAS = "Legolas"


class CustomFrame(TextFrame):
    def __init__(self, text: str, character: Character, **kwargs):
        super().__init__(text=text, **kwargs)
        self.character = character


class FrameAndCharacterHandler(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.previous_frame_character = None
        self.frame_character = None
        self.voice_change_lock = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            custom_frame = self.create_custom_frame(frame)
            if custom_frame:
                async with self.voice_change_lock:
                    voice_changed = await self.process_frame_character(custom_frame)
                    if voice_changed:
                        # Wait a bit for the voice change to take effect
                        await asyncio.sleep(1)  # Adjust this delay as needed
                await self.push_frame(custom_frame, direction)
        else:
            await self.push_frame(frame, direction)

    def create_custom_frame(self, frame: TextFrame):
        if "#####" in frame.text:
            self.frame_character = "Gimli"
            frame.text = frame.text.replace("#####", "").strip()
        elif "-----" in frame.text:
            self.frame_character = "Legolas"
            frame.text = frame.text.replace("-----", "").strip()

        if self.frame_character and frame.text:
            return CustomFrame(text=frame.text, character=self.frame_character)
        return None

    async def process_frame_character(self, frame: CustomFrame):
        if self.previous_frame_character != frame.character:
            await self.change_tts_settings(frame.character)
            self.previous_frame_character = frame.character
            return True
        return False

    async def change_tts_settings(self, character: str):
        voice_id_mapping = {
            "Gimli": "79a125e8-cd45-4c13-8a67-188112f4dd22",
            "Legolas": "95856005-0332-41b0-935f-352e296aa0df",
        }
        voice_id = voice_id_mapping[character]
        if voice_id:
            settingsFrame = TTSUpdateSettingsFrame(settings={"voice": voice_id})
            await self.push_frame(settingsFrame)
            # Wait for the settings to be applied
            await asyncio.sleep(1)  # Adjust this delay as needed
        else:
            logger.warning(f"No voice ID found for character: {character}")


async def main():
    async with aiohttp.ClientSession() as session:
        room_url = "some url here"
        token = "some token here"

        transport = DailyTransport(
            room_url,
            token,
            "Pipecat",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        messages = [
            {
                "role": "system",
                "content": """You are writing a dialogue between two characters from Middle Earth: Gimli, a proud Dwarf, and Legolas, a skilled Elf. They are discussing who is better at craftsmanship: Dwarves or Elves. Make the conversation engaging and lively, with each character passionately defending their race's skills.

                Separate Gimli's parts/responses with ##### and Legolas's with -----. Do not add any other text than the tags and the responses. They should argue for at least 10 turns before both deciding that dealing with Sauron is more important than their craftsmanship debate.

                Example:
                ##### By Durin's beard, everyone knows that Dwarven craftsmanship is unmatched in all of Middle Earth! Our axes and armor are the stuff of legend!
                ----- While I respect your people's skills, Gimli, surely you jest. Elven craftsmanship has been refined over thousands of years. Our blades and bows are works of art!

                Begin the dialogue immediately, starting with Gimli's perspective.""",
            }]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)
        frame_and_character_handler = FrameAndCharacterHandler()

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=os.getenv("79a125e8-cd45-4c13-8a67-188112f4dd22"),  # Use Gimlis voice here
        )

        pipeline = Pipeline(
            [
                # transport.input(),            # Transport user input - I have commented this out because we don't want to capture user input. Just for this example. Add it back in if you want to capture user input.
                # context_aggregator.user(),    # User responses - I have commented this out because we don't want to capture user input. Just for this example. Add it back in if you want to capture user input.
                llm,
                frame_and_character_handler,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": "Please begin the dialogue between Gimli and Legolas immediately. Start with Gimli's perspective on Dwarven craftsmanship. Do not add any introductions or additional text; begin directly with Gimli's first statement.",
                }
            )
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
