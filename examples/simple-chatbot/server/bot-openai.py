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
import time
from datetime import datetime
from typing import Dict, Optional

import aiohttp
import sentry_sdk
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotInterruptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.metrics.sentry import SentryMetrics

# Load environment variables
load_dotenv(override=True)

# Initialize Sentry
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,  # Capture 100% of transactions for latency metrics
        profiles_sample_rate=1.0,  # Capture performance profiles
        integrations=[AsyncioIntegration()],  # Add asyncio integration
    )
    logger.info("Sentry initialized with DSN")
else:
    logger.warning("SENTRY_DSN not found in environment variables")

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


class LatencyTracker:
    """Tracks and calculates latencies between conversation events.
    
    Specifically tracks:
    1. Response Latency: Time from user stops talking to bot starts talking
    2. Interruption Latency: Time from user starts talking to bot stops talking (during bot speech)
    """
    
    def __init__(self):
        # Timestamps for different events
        self.user_started_speaking: Optional[float] = None
        self.user_stopped_speaking: Optional[float] = None
        self.bot_started_speaking: Optional[float] = None
        self.bot_stopped_speaking: Optional[float] = None
        
        # Conversation tracking
        self.conversation_id = f"conversation-{int(time.time())}"
        self.turn_count = 0
        
        # For tracking if user interrupted bot
        self.bot_is_speaking = False
        self.bot_was_interrupted = False
        self.interruption_timestamp = None
        self.interruption_transaction = None
        
        # Store active transactions
        self.response_latency_transaction = None
        
        # Store latency metrics
        self.response_latencies: Dict[datetime, float] = {}
        self.interrupt_latencies: Dict[datetime, float] = {}
    
    def user_start(self):
        """Record when user starts speaking."""
        timestamp = time.time()
        self.user_started_speaking = timestamp
        self.turn_count += 1
        logger.debug(f"User started speaking (Turn {self.turn_count})")
        
        # Check if this is an interruption (user starts while bot is speaking)
        if self.bot_is_speaking:
            self.bot_was_interrupted = True
            self.interruption_timestamp = timestamp
            logger.info(f"Detected interruption: User started speaking while bot was speaking (Turn {self.turn_count})")
            
            # Start interruption latency tracking
            if sentry_dsn:
                try:
                    # Create a transaction for this interruption
                    self.interruption_transaction = sentry_sdk.start_transaction(
                        op="interruption_latency",
                        name=f"Interruption Latency Turn {self.turn_count}",
                    )
                    self.interruption_transaction.set_tag("conversation_id", self.conversation_id)
                    self.interruption_transaction.set_tag("turn", self.turn_count)
                    self.interruption_transaction.set_tag("metric_type", "interruption_latency")
                    self.interruption_transaction.set_tag("interruption_type", "user_over_bot")
                    self.interruption_transaction.set_data("detection_method", "user_start_during_bot_speech")
                    self.interruption_transaction.set_data("user_interruption_timestamp", timestamp)
                    self.interruption_transaction.set_data("bot_was_speaking_since", self.bot_started_speaking)
                except Exception as e:
                    logger.error(f"Error creating interruption transaction: {e}")
    
    def user_stop(self):
        """Record when user stops speaking."""
        if not self.user_started_speaking:
            return
            
        self.user_stopped_speaking = time.time()
        logger.debug(f"User stopped speaking (Turn {self.turn_count})")
        
        # Start a transaction to track response latency
        if sentry_dsn:
            try:
                # Start transaction for response latency
                self.response_latency_transaction = sentry_sdk.start_transaction(
                    op="response_latency",
                    name=f"Response Latency Turn {self.turn_count}",
                )
                self.response_latency_transaction.set_tag("conversation_id", self.conversation_id)
                self.response_latency_transaction.set_tag("turn", self.turn_count)
                self.response_latency_transaction.set_tag("metric_type", "response_latency")
                self.response_latency_transaction.set_data("user_stopped_at", self.user_stopped_speaking)
                # This transaction will be finished when bot starts speaking
            except Exception as e:
                logger.error(f"Error creating response latency transaction: {e}")
    
    def bot_start(self):
        """Record when bot starts speaking and calculate response latency."""
        timestamp = time.time()
        self.bot_started_speaking = timestamp
        self.bot_is_speaking = True
        logger.debug(f"Bot started speaking (Turn {self.turn_count})")
        
        # Calculate and record response latency if this follows a user turn
        if self.user_stopped_speaking:
            latency = timestamp - self.user_stopped_speaking
            response_time = datetime.now()
            self.response_latencies[response_time] = latency
            logger.info(f"Response latency: {latency:.3f} seconds (Turn {self.turn_count})")
            
            # Finish the response latency transaction and record the measurement
            if sentry_dsn and self.response_latency_transaction:
                try:
                    # Set transaction data and finish it
                    self.response_latency_transaction.set_data("bot_started_at", timestamp)
                    self.response_latency_transaction.set_data("latency_seconds", latency)
                    sentry_sdk.set_measurement("response_latency", latency, "second")
                    self.response_latency_transaction.finish()
                except Exception as e:
                    logger.error(f"Error finishing response latency transaction: {e}")
                finally:
                    # Clear the reference
                    self.response_latency_transaction = None
    
    def bot_stop(self):
        """Record when bot stops speaking and finalize interruption latency if applicable."""
        if not self.bot_is_speaking:
            return
            
        timestamp = time.time()
        self.bot_stopped_speaking = timestamp
        self.bot_is_speaking = False
        logger.debug(f"Bot stopped speaking (Turn {self.turn_count})")
        
        # If this was an interruption, calculate and record interruption latency
        if self.bot_was_interrupted and self.interruption_timestamp:
            latency = timestamp - self.interruption_timestamp
            interrupt_time = datetime.now()
            self.interrupt_latencies[interrupt_time] = latency
            logger.info(f"Interruption latency: {latency:.3f} seconds (Turn {self.turn_count})")
            
            # Finish the interruption latency transaction
            if sentry_dsn and self.interruption_transaction:
                try:
                    self.interruption_transaction.set_data("bot_stopped_at", timestamp)
                    self.interruption_transaction.set_data("latency_seconds", latency)
                    sentry_sdk.set_measurement("interruption_latency", latency, "second")
                    self.interruption_transaction.finish()
                except Exception as e:
                    logger.error(f"Error finishing interruption transaction: {e}")
                finally:
                    self.interruption_transaction = None
            
        # Reset interruption state for next turn
        self.bot_was_interrupted = False
        self.interruption_timestamp = None
        self.bot_started_speaking = None
        self.bot_stopped_speaking = None
    
    def explicit_interruption(self):
        """Handle explicit interruption frame from pipeline.
        
        This is an alternative detection method when BotInterruptionFrame is received.
        """
        timestamp = time.time()
        logger.info(f"Explicit interruption frame received (Turn {self.turn_count})")
        
        # Only track if bot is actually speaking
        if self.bot_is_speaking and not self.bot_was_interrupted:
            self.bot_was_interrupted = True
            self.interruption_timestamp = timestamp
            
            # Start interruption latency tracking
            if sentry_dsn:
                try:
                    # Create a transaction for this interruption
                    self.interruption_transaction = sentry_sdk.start_transaction(
                        op="interruption_latency",
                        name=f"Explicit Interruption Turn {self.turn_count}",
                    )
                    self.interruption_transaction.set_tag("conversation_id", self.conversation_id)
                    self.interruption_transaction.set_tag("turn", self.turn_count)
                    self.interruption_transaction.set_tag("metric_type", "interruption_latency")
                    self.interruption_transaction.set_tag("interruption_type", "explicit")
                    self.interruption_transaction.set_data("detection_method", "bot_interruption_frame")
                    self.interruption_transaction.set_data("interruption_timestamp", timestamp)
                    self.interruption_transaction.set_data("bot_was_speaking_since", self.bot_started_speaking)
                except Exception as e:
                    logger.error(f"Error creating explicit interruption transaction: {e}")
    
    def export_metrics(self):
        """Return current latency metrics.
        
        Returns:
            dict: Dictionary containing response and interrupt latency metrics
        """
        metrics = {
            "response_latencies": self.response_latencies,
            "interrupt_latencies": self.interrupt_latencies,
            "avg_response_latency": sum(self.response_latencies.values()) / len(self.response_latencies) if self.response_latencies else 0,
            "avg_interrupt_latency": sum(self.interrupt_latencies.values()) / len(self.interrupt_latencies) if self.interrupt_latencies else 0,
            "total_turns": self.turn_count,
            "interruptions": len(self.interrupt_latencies)
        }
        
        # Send summary metrics to Sentry
        if sentry_dsn and (self.response_latencies or self.interrupt_latencies):
            try:
                with sentry_sdk.start_transaction(
                    op="latency_summary",
                    name=f"Latency Summary - Conversation {self.conversation_id}"
                ) as transaction:
                    transaction.set_tag("conversation_id", self.conversation_id)
                    transaction.set_tag("metric_type", "latency_summary")
                    transaction.set_tag("total_turns", self.turn_count)
                    transaction.set_tag("total_interruptions", len(self.interrupt_latencies))
                    
                    if metrics["avg_response_latency"]:
                        sentry_sdk.set_measurement("avg_response_latency", metrics["avg_response_latency"], "second")
                        transaction.set_data("avg_response_latency", f"{metrics['avg_response_latency']:.3f}s")
                        
                    if metrics["avg_interrupt_latency"]:
                        sentry_sdk.set_measurement("avg_interrupt_latency", metrics["avg_interrupt_latency"], "second")
                        transaction.set_data("avg_interrupt_latency", f"{metrics['avg_interrupt_latency']:.3f}s")
            except Exception as e:
                logger.error(f"Error exporting metrics summary: {e}")
                
        return metrics


class LatencyTrackerProcessor(FrameProcessor):
    """Frame processor that tracks latency metrics.
    
    Processes frame events to calculate response and interrupt latencies.
    """
    
    def __init__(self):
        super().__init__()
        self.tracker = LatencyTracker()
        # Start metrics export task
        self.metrics_task = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update latency metrics.
        
        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        # Initialize the processor correctly
        await super().process_frame(frame, direction)
        
        # Process specific frame types to track latency
        if isinstance(frame, UserStartedSpeakingFrame):
            self.tracker.user_start()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.tracker.user_stop()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.tracker.bot_start()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.tracker.bot_stop()
        elif isinstance(frame, BotInterruptionFrame):
            # Process explicit interruption frame from pipeline
            self.tracker.explicit_interruption()
        
        # Pass the frame through
        await self.push_frame(frame, direction)
    
    async def start_metrics_export(self):
        """Start the metrics export task."""
        # Cancel existing task if it exists
        if self.metrics_task:
            self.metrics_task.cancel()
        
        # Create periodic metrics export task
        async def export_metrics_to_sentry():
            while True:
                metrics = self.tracker.export_metrics()
                # Here you would send the metrics to Sentry
                # Using your existing Sentry integration
                logger.info(f"Metrics: {metrics}")
                await asyncio.sleep(60)  # Export every minute
        
        self.metrics_task = asyncio.create_task(export_metrics_to_sentry())
        
    async def stop_metrics_export(self):
        """Stop the metrics export task."""
        if self.metrics_task:
            self.metrics_task.cancel()
            self.metrics_task = None


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

        # Initialize latency tracker processor
        latency_processor = LatencyTrackerProcessor()

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
            metrics=SentryMetrics(),
        )

        # Initialize LLM service
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            metrics=SentryMetrics(),
        )

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

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                latency_processor,  # Add latency tracker to the pipeline
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

        # Start the metrics export
        await latency_processor.start_metrics_export()

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            print(f"Participant joined: {participant}")
            await transport.start_recording()
            await transport.capture_participant_transcription(participant["id"])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await transport.stop_recording()
            # Stop the metrics export
            await latency_processor.stop_metrics_export()
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
