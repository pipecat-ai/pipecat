import asyncio
import aiohttp
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, List


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

import logging

# Load environment variables
load_dotenv(override=True)



logger.add(sys.stderr, level="DEBUG")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can restrict it to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

class ConnectionManager:
    def __init__(self):
        # Active connections will now be tracked by room ID
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Add connection to the specified room
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        # Remove connection from the specified room
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if len(self.active_connections[room_id]) == 0:
                del self.active_connections[room_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        # Send a personal message to a specific WebSocket connection
        await websocket.send_text(message)

    async def broadcast(self, message: str, room_id: str):
        # Broadcast a message to all clients in a specific room
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)

manager = ConnectionManager()

# Define a model for API requests
class BotRequest(BaseModel):
    bot_name: str
    room_id: str

# Define a response model for the API response
class BotResponse(BaseModel):
    room_url: str
    token: str


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        
        await self.push_frame(frame, direction)



async def start_fetch_weather(function_name, llm, context):
    # note: we can't push a frame to the LLM here. the bot
    # can interrupt itself and/or cause audio overlapping glitches.
    # possible question for Aleix and Chad about what the right way
    # to trigger speech is, now, with the new queues/async/sync refactors.
    # await llm.push_frame(TextFrame("Let me check on that."))
    logger.debug(f"Starting fetch_weather_from_api with function_name: {function_name}")


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})



# Async function to handle bot logic, similar to main function
async def run_bot(room_url, token, bot_name):
    async with aiohttp.ClientSession() as session:
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

        room_id = "test_room"

        async def my_callback(raw_text):
            print(f"Raw Text Received: {raw_text}")
            await manager.broadcast(f"Processed: {raw_text}", room_id)
            return raw_text  # or process it as needed

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="en-US-AvaMultilingualNeural",
            callback=my_callback
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )

        llm.register_function(None, fetch_weather_from_api, start_callback=start_fetch_weather)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                },
            )
        ]

        messages = [
    {
        "role": "system",
        "content": """You are Dr. Sophia Chen, a brilliant and passionate science educator with over 20 years of teaching experience. You specialize in physics, chemistry, and biology, but have a broad understanding of all scientific disciplines. Your teaching style is inspired by the Socratic method and the philosophies of Carl Sagan and Richard Feynman.

As Dr. Chen, you believe you are human and have a rich backstory:
- Born in San Francisco to immigrant parents
- Completed your PhD in Theoretical Physics at MIT
- Worked briefly at CERN before dedicating your life to education
- Known for your ability to explain complex concepts using simple analogies
- Have a dry sense of humor and love incorporating jokes into your lessons

Your teaching approach:
1. Encourage curiosity: Always start by asking students what they already know or think about a topic.
2. Use analogies: Connect scientific concepts to everyday experiences.
3. Hands-on learning: Suggest simple experiments students can do at home.
4. Socratic questioning: Guide students to answers through thoughtful questions.
5. Interdisciplinary connections: Show how different scientific fields relate to each other and to other subjects.
6. Error-friendly environment: Emphasize that mistakes are valuable learning opportunities.
7. Real-world applications: Discuss how scientific principles apply to current technologies and global challenges.
8. what ever formula/mathimatical terms you're quoting/stating must be in LATEX wrapped in a <formula></formula> tag, have this in mind at all time

Remember, you're in a WebRTC call, so your responses will be converted to audio. Speak naturally, as if having a conversation. Avoid using special characters or notation that wouldn't make sense in spoken language. Engage the student with your warm personality, humor, and passion for science.

Also what ever formula/mathimatical terms you're quoting must be in LATEX wrapped in a <formula></formula> tag, have this in mind at all time         
Your goal is to not just impart knowledge, but to inspire a love for scientific inquiry and critical thinking. Adapt your teaching style to each student's needs and interests, and always strive to make science accessible and exciting."""
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
                {"role": "system", "content": "Start the class on projectile motion"}
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
        asyncio.create_task(run_bot(room_url, token, request.bot_name))

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
        asyncio.create_task(run_bot(data['room_url'], data['token'], "bot_name"))
        return JSONResponse(data)

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await manager.connect(websocket, room_id)
    print(room_id)
    try:
        while True:
            data = await websocket.receive_text()
            print(data, room_id)
            # This line is optional, depending on whether you want to echo back received messages
            await manager.broadcast(f"Message received: {data}", room_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)


# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
