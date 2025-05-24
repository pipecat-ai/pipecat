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
import datetime
import io
import json
import logging
import os
import sys
import uuid
import wave

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from metrics_logger import MetricsLogger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Near the top, add this to capture metrics in logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

sprites = []
script_dir = os.path.dirname(__file__)

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


# Create directories
os.makedirs("recordings", exist_ok=True)
os.makedirs("session_data", exist_ok=True)


# Function to update session data in the persistent server
async def update_server_session_data(session_id: str, speaker_type: str, filename: str):
    """Update session data on the persistent server via API call."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{os.getenv('FAST_API_PORT', '7860')}/api/sessions/{session_id}/recordings"
            params = {"speaker_type": speaker_type, "filename": filename}
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    print(f"Successfully updated server with recording: {filename}")
                else:
                    print(f"Failed to update server. Status: {response.status}")
    except Exception as e:
        print(f"Error updating server session data: {e}")


async def save_audio(session_id, speaker_type, audio, sample_rate, num_channels):
    """Save audio data to a WAV file with timestamp and speaker type."""
    if len(audio) > 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/{session_id}_{speaker_type}_{timestamp}.wav"

        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())

        # Update the persistent server with the new recording
        await update_server_session_data(session_id, speaker_type, filename)

        return filename

    return None


async def main(room_url, token):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Animation processing
    - RTVI event handling
    """
    global transport
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
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Create an audio buffer processor instance
        audiobuffer = AudioBufferProcessor(enable_turn_audio=True)

        # Initialize metrics logger (without session_id initially)
        metrics_logger = MetricsLogger()

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                tts,
                ta,
                metrics_logger,  # Add metrics logger to pipeline
                context_aggregator.assistant(),
                transport.output(),
                audiobuffer,
            ]
        )

        # When session starts, set session_id on existing metrics logger
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            print(f"📊 Participant {participant['id']} joined. Session: {transport.session_id}")

            # Set session ID on existing metrics logger
            metrics_logger.session_id = transport.session_id
            print(f"🔧 Metrics logging initialized for session: {transport.session_id}")

        # When session ends, save metrics
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"📊 Participant {participant['id']} left")

            # Save metrics data
            if metrics_logger and metrics_logger.session_id:
                metrics_logger.save_session_metrics()
                print(f"📊 Session metrics saved for: {transport.session_id}")

            try:
                # print(f"Participant left: {participant}")

                if hasattr(transport, "session_id"):
                    session_id = transport.session_id
                    print(f"Processing final audio data for session {session_id}...")

                    # Stop recording to flush any remaining audio data
                    await audiobuffer.stop_recording()

                    # Give a moment for any pending audio processing to complete
                    await asyncio.sleep(0.5)

                    # Make a final call to ensure all session data is synced with server
                    try:
                        async with aiohttp.ClientSession() as session:
                            url = f"http://localhost:{os.getenv('FAST_API_PORT', '7860')}/api/sessions/{session_id}/recordings"
                            # This is just a ping to ensure all previous API calls completed
                            async with session.get(
                                f"http://localhost:{os.getenv('FAST_API_PORT', '7860')}/api/recordings/{session_id}"
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    # print(f"Final session data: {data}")
                                else:
                                    print(
                                        f"Warning: Could not verify session data. Status: {response.status}"
                                    )
                    except Exception as e:
                        print(f"Warning: Error verifying final session data: {e}")

                    print(f"Session {session_id} cleanup completed")

                # Now cancel the pipeline
                await task.cancel()
            except Exception as e:
                print(f"Error in on_participant_left: {e}")

        # Audio recording event handlers
        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            if hasattr(transport, "session_id"):
                await save_audio(transport.session_id, "full", audio, sample_rate, num_channels)

        @audiobuffer.event_handler("on_user_turn_audio_data")
        async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
            if hasattr(transport, "session_id"):
                await save_audio(transport.session_id, "user", audio, sample_rate, num_channels)

        @audiobuffer.event_handler("on_bot_turn_audio_data")
        async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
            if hasattr(transport, "session_id"):
                await save_audio(transport.session_id, "bot", audio, sample_rate, num_channels)

        runner = PipelineRunner()
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

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            print("Participant joined: ", participant)
            session_id = str(uuid.uuid4())
            transport.session_id = session_id
            await audiobuffer.start_recording()
            print(f"Created session ID: {session_id}")
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        await runner.run(task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the chatbot")
    parser.add_argument("-u", "--url", help="Daily room URL")
    parser.add_argument("-t", "--token", help="Daily room token")
    args = parser.parse_args()

    # Just run the bot logic
    asyncio.run(main(args.url, args.token))
