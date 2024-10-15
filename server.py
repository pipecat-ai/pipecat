import asyncio
import aiohttp
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel



from pipecat.frames.frames import Frame, LLMMessagesFrame, MetricsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.azure import AzureTTSService, AzureLLMService
from pipecat.services.openai import OpenAITTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from openai.types.chat import ChatCompletionToolParam

from pipecat.frames.frames import LLMMessagesFrame, EndFrame

from loguru import logger
from dotenv import load_dotenv

from main.fetch_room import configure, ConfigRequest
from main.function_store import WeatherTool, WritingTool
from main.wss import ConnectionManager
from main.types.pydantic_types import BotRequest, BotResponse
from main.utils.prompts import agent_system_prompt
import random
import string

import logging

# Load environment variables
load_dotenv(override=True)


logger.add(sys.stderr, level="DEBUG")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Allows all origins, you can restrict it to specific domains
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


manager = ConnectionManager()

class RoomIdStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoomIdStore, cls).__new__(cls)
            cls._instance.room_id = ""
        return cls._instance

room_id_store = RoomIdStore()



class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):

        await self.push_frame(frame, direction)



# Async function to handle bot logic, similar to main function
async def run_bot(room_url, token, bot_name, room_id):
    async with aiohttp.ClientSession() as session:

        # Initialize tools with room_id passed
        weather_tool = WeatherTool(room_id=room_id)
        writing_tool = WritingTool(room_id=room_id)
        
        # Run the main function of the tools
        weather_tool.main()
        writing_tool.main()

        # Create the DailyTransport with the provided room URL, token, and bot name
        transport = DailyTransport(
            room_url,
            token,
            bot_name,
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2))
            )
        )

        # room_id = "test_room"

        async def my_callback(raw_text):
            print(f"Raw Text Received: {raw_text}")
            message_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            data = {"type": "message",  "content": raw_text,  "message_id": message_id}
            await manager.broadcast_json(data, room_id)
            return raw_text  # or process it as needed

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="en-US-JennyMultilingualNeural",
            callback=my_callback
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )

        llm.register_function(None, weather_tool.get_main_function(
        ), start_callback=weather_tool.get_start_callback_function())
        llm.register_function("Jotting_tool", writing_tool.get_main_function(
        ), start_callback=writing_tool.get_start_callback_function())
        tools = [
            weather_tool.get_function_definition(),
            writing_tool.get_function_definition()
        ]

        messages = [
            {
                "role": "system",
                "content": agent_system_prompt
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        # Initialize the pipeline with all necessary components
        pipeline = Pipeline([
            transport.input(),
            # LLMUserResponseAggregator(messages),
            context_aggregator.user(),
            llm,
            tts,
            MetricsLogger(),
            transport.output(),
            # LLMAssistantResponseAggregator(messages),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline, PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ))

        runner = PipelineRunner()

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logging.info(f"Participant left: {participant['id']}")
            await task.queue_frame(EndFrame())

        # Event handler for first participant joining the room
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            # participant_name = participant["info"]["userName"] or ""
            # Kick off the conversation.
            messages.append(
                {"role": "system", "content": "log the meaning of gravity on my notepad"}
            )
            await task.queue_frames([LLMMessagesFrame(messages)])

        # Run the pipeline task
        await runner.run(task)


# Assuming you have the bot initialization function called `start_bot`
active_bots = set()


@app.get("/")
async def home():
    return JSONResponse({"message": "Welcome to the Daily Storyteller API!"})


@app.get("/o/")
async def config():
    return JSONResponse({"message": "Welcome to the Daily Storyteller API!"})

# FastAPI route to start the bot and return room details


@app.post("/start-bot", response_model=BotResponse)
async def start_bot(request: BotRequest):
    try:
        if request.bot_name in active_bots:
            return {"error": f"Bot '{request.bot_name}' is already in the room"}

        active_bots.add(request.bot_name)
        # Logic to configure the room URL and token based on room ID
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)

        # Start the agent (run the bot asynchronously)
        asyncio.create_task(run_bot(room_url, token, request.bot_name, "test_room"))

        # Return the room URL and token
        return BotResponse(room_url=room_url, token=token)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to start the bot")


@app.post("/configure")
async def configure_endpoint(request: ConfigRequest):
    async with aiohttp.ClientSession() as session:
        # Call configure function
        data = await configure(session, request)
        print(data)
        asyncio.create_task(
            run_bot(data['room_url'], data['token'], "bot_name", request.room_id))
        return JSONResponse(data)


@app.websocket("/ws/{user_room_id}")
async def websocket_endpoint(websocket: WebSocket, user_room_id: str):
    
    await manager.connect(websocket, user_room_id)
    print(user_room_id)
    room_id_store.room_id = user_room_id
    try:
        while True:
            data = await websocket.receive_text()
            print(data, user_room_id)
            
            # This line is optional, depending on whether you want to echo back received messages
            await manager.broadcast(f"Message received: {data}", user_room_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_room_id)


# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
