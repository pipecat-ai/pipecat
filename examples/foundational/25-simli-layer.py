import asyncio
import aiohttp
import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.ai_services import AIService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams
from runner import configure
from loguru import logger
from dotenv import load_dotenv

from simli import SimliConfig
from pipecat.services.simli import SimliVideoService

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        _, token = await configure(session)
        print("Creating room")
        aiohttp_session = aiohttp.ClientSession()
        daily_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY", ""),
            daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
            aiohttp_session=aiohttp_session,
        )
        room = await daily_helper.create_room(DailyRoomParams())
        expiry_time: float = 60 * 60

        token = await daily_helper.get_token(room.url, expiry_time)
        print("Room created ", room.url)
        transport = DailyTransport(
            room.url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=512,
                camera_out_height=512,
                vad_enabled=True,
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

        # tts = ElevenLabsTTSService(
        #     api_key=os.getenv("ELEVENLABS_API_KEY"),
        #     voice_id="pNInz6obpgDQGcFmaJgB",
        # )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                #
                # English
                #
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
            },
        ]
        simli_ai = SimliVideoService(
            SimliConfig(os.getenv("SIMLI_API_KEY"), os.getenv("SIMLI_FACE_ID"))
        )
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                simli_ai,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
