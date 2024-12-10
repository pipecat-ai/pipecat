import argparse
import asyncio
import os
import sys

import aiohttp
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

logger.remove(0)
logger.add(sys.stderr, level="INFO")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


class DialoutBot:
    def __init__(self, room_url: str, token: str, callId: int, run_number: int, phone_number: str):
        self.recording_id = None
        self.room_url = room_url
        self.token = token
        self.callId = callId
        self.run_number = run_number
        self.phone_number = phone_number

    async def run(self):
        transport = DailyTransport(
            self.room_url,
            self.token,
            "Chatbot",
            DailyParams(
                api_url=daily_api_url,
                api_key=daily_api_key,
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by saying 'Oh, hello! Who dares dial me at this hour?!'.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        def get_phone_number(callId: int, run_number: int) -> str:
            if self.phone_number:
                return self.phone_number

            if run_number % 2 == 0:
                phone_numbers = [
                    "+12097135124",  # James
                    "+12097135125",  # James
                    "+19499870006",  # Varun
                ]
                return phone_numbers[callId % len(phone_numbers)]
            else:
                phone_numbers = [
                    "+14155204406",  # James
                    "+18187229086",  # James (Avoca)
                    "+16673870006",  # Varun
                ]
                return phone_numbers[callId % len(phone_numbers)]

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            logger.info(f"on_call_state_updated, state: {state}")
            # dialout_id = None

            if state == "joined":
                logger.info(f"on_call_state_updated {state}")

                backoff_time = 1  # Initial backoff time in seconds

                for _ in range(5):
                    try:
                        phone_number = get_phone_number(self.callId, self.run_number)
                        logger.info(f"Starting dialout to {phone_number}")
                        settings = {
                            "phoneNumber": phone_number,
                            "display_name": "Dialout User",
                        }
                        await transport.start_dialout(settings)
                        break  # Break out of the loop if start_dialout is successful
                    except Exception as e:
                        logger.error(f"Error starting dialout: {e}")
                        await asyncio.sleep(backoff_time)  # Wait for the current backoff time
                        backoff_time *= 2  # Double the backoff time for the next attempt

            if state == "left":
                logger.info(f"on_call_state_updated {state}")
                # await transport.stop_dialout(dialout_id)
                async with aiohttp.ClientSession() as aiohttp_session:
                    print(f"Deleting room: {self.room_url}")
                    rest = DailyRESTHelper(
                        daily_api_key=os.getenv("DAILY_API_KEY", ""),
                        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
                        aiohttp_session=aiohttp_session,
                    )
                    await rest.delete_room_by_url(self.room_url)

        # @transport.event_handler("on_first_participant_joined")
        # async def on_first_participant_joined(transport, participant):
        #     await transport.capture_participant_transcription(participant["id"])
        #     await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_dialout_answered")
        async def on_dialout_answered(transport, participant):
            logger.info(f"on_dialout_answered {participant["participantId"]}")
            streaming_settings = {
                "minIdleTimeOut": 10,
                "layout": {
                    "preset": "audio-only",
                },
            }

            backoff_time = 1  # Initial backoff time in seconds
            for _ in range(5):
                try:
                    await transport.start_recording(streaming_settings=streaming_settings)
                    break  # Break out of the loop if start_dialout is successful
                except Exception as e:
                    logger.error(f"Error starting recording: {e}")
                    await asyncio.sleep(backoff_time)  # Wait for the current backoff time
                    backoff_time *= 2  # Double the backoff time for the next attempt

            await transport.capture_participant_transcription(participant["participantId"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_recording_started")
        async def on_recording_started(transport, stream_id):
            self.recording_id = stream_id["streamId"]
            logger.info(f"Recording started: {self.recording_id}")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}, reason: {reason}")
            logger.info(f"Stopping recording: {self.recording_id}")
            await transport.stop_recording(self.recording_id)
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_recording_error")
        async def on_recording_error(transport, error):
            logger.error(f"Recording error: {error}")
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_dialout_error")
        async def on_dialout_error(transport, error):
            logger.error(f"Dialout error: {error}")
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Simple ChatBot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-r", type=str, help="Run Number")
    parser.add_argument("-p", type=str, help="Phone Number")
    config = parser.parse_args()

    bot = DialoutBot(config.u, config.t, int(config.i), int(config.r), config.p)

    try:
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"++++++++++++++ Error: {e}")
        sys.exit(1)
