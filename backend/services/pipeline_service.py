"""
Pipecat Pipeline Service
Optimized for local AI rig with Ollama and Cartesia for minimum latency
"""

import os
import asyncio
from typing import Optional, Dict, Any, Callable
from loguru import logger

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import EndFrame, TTSTextFrame
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)

# Services
from pipecat.services.ollama import OLLamaLLMService
from pipecat.services.cartesia import CartesiaTTSService, CartesiaSTTService

# Transports
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketClient,
)

# Audio processing
from pipecat.audio.vad.silero import SileroVADAnalyzer

from ..config import settings, VOICE_CONFIGS, SYSTEM_PROMPTS


class PipelineService:
    """
    Manages Pipecat voice pipelines for real-time conversations
    Optimized for low-latency with local Ollama and Cartesia
    """

    def __init__(self):
        self.active_pipelines: Dict[str, PipelineTask] = {}
        self.runners: Dict[str, PipelineRunner] = {}

    async def create_pipeline(
        self,
        session_id: str,
        websocket,
        voice_config: str = "conversational",
        system_prompt: str = "default",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> PipelineTask:
        """
        Create a new voice pipeline for a session

        Args:
            session_id: Unique session identifier
            websocket: FastAPI WebSocket connection
            voice_config: Voice configuration key from VOICE_CONFIGS
            system_prompt: System prompt key from SYSTEM_PROMPTS
            custom_config: Optional custom configuration overrides

        Returns:
            PipelineTask ready to run
        """
        logger.info(f"Creating pipeline for session {session_id}")

        # Get voice configuration
        voice_conf = VOICE_CONFIGS.get(voice_config, VOICE_CONFIGS["conversational"])
        prompt = SYSTEM_PROMPTS.get(system_prompt, SYSTEM_PROMPTS["default"])

        if custom_config:
            voice_conf = {**voice_conf, **custom_config.get("voice", {})}
            prompt = custom_config.get("system_prompt", prompt)

        # Initialize services
        try:
            # 1. STT Service (Cartesia for fast transcription)
            stt = CartesiaSTTService(
                api_key=settings.CARTESIA_API_KEY,
                model=settings.CARTESIA_STT_MODEL,
                language=settings.CARTESIA_LANGUAGE,
            )

            # 2. LLM Service (Ollama on local AI rig)
            llm = OLLamaLLMService(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                api_key="ollama",  # Dummy key for local
            )

            # 3. TTS Service (Cartesia for fast, high-quality voice)
            tts = CartesiaTTSService(
                api_key=settings.CARTESIA_API_KEY,
                voice_id=voice_conf["voice_id"],
                model=settings.CARTESIA_TTS_MODEL,
                language=settings.CARTESIA_LANGUAGE,
            )

            # 4. VAD Analyzer for better turn detection
            vad = SileroVADAnalyzer()

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

        # Create LLM context with system prompt
        messages = [{"role": "system", "content": prompt}]

        # Create context aggregators
        user_aggregator = LLMUserResponseAggregator(messages)
        assistant_aggregator = LLMAssistantResponseAggregator(messages)

        # Build the pipeline
        # Flow: Input -> STT -> User Context -> LLM -> TTS -> Output -> Assistant Context
        pipeline = Pipeline(
            [
                stt,  # Speech to text
                user_aggregator,  # Aggregate user messages
                llm,  # Generate responses with Ollama
                tts,  # Text to speech
                assistant_aggregator,  # Aggregate assistant messages
            ]
        )

        # Create transport parameters for WebSocket
        transport_params = FastAPIWebsocketParams(
            websocket=websocket,
            audio_out_sample_rate=settings.AUDIO_SAMPLE_RATE,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=vad,
            vad_audio_passthrough=True,
        )

        # Create pipeline parameters
        pipeline_params = PipelineParams(
            allow_interruptions=True,
            enable_metrics=settings.ENABLE_METRICS,
            enable_usage_metrics=True,
        )

        # Create pipeline task
        task = PipelineTask(
            pipeline,
            params=pipeline_params,
            # transport_params=transport_params,  # TODO: Check if this is the correct way
        )

        # Store pipeline reference
        self.active_pipelines[session_id] = task

        logger.info(f"Pipeline created successfully for session {session_id}")
        return task

    async def run_pipeline(self, session_id: str, task: PipelineTask) -> None:
        """
        Run a pipeline task

        Args:
            session_id: Session identifier
            task: Pipeline task to run
        """
        logger.info(f"Starting pipeline for session {session_id}")

        try:
            runner = PipelineRunner()
            self.runners[session_id] = runner
            await runner.run(task)
        except Exception as e:
            logger.error(f"Pipeline error for session {session_id}: {e}")
            raise
        finally:
            await self.cleanup_pipeline(session_id)

    async def stop_pipeline(self, session_id: str) -> None:
        """
        Stop a running pipeline

        Args:
            session_id: Session identifier
        """
        logger.info(f"Stopping pipeline for session {session_id}")

        if session_id in self.active_pipelines:
            task = self.active_pipelines[session_id]
            try:
                # Send end frame to gracefully stop
                await task.queue_frame(EndFrame())
            except Exception as e:
                logger.error(f"Error stopping pipeline {session_id}: {e}")

        if session_id in self.runners:
            runner = self.runners[session_id]
            try:
                await runner.stop()
            except Exception as e:
                logger.error(f"Error stopping runner {session_id}: {e}")

    async def cleanup_pipeline(self, session_id: str) -> None:
        """
        Clean up pipeline resources

        Args:
            session_id: Session identifier
        """
        logger.info(f"Cleaning up pipeline for session {session_id}")

        if session_id in self.active_pipelines:
            del self.active_pipelines[session_id]

        if session_id in self.runners:
            del self.runners[session_id]

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs"""
        return list(self.active_pipelines.keys())

    def get_session_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_pipelines)


# Global pipeline service instance
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get or create the global pipeline service instance"""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service


async def create_voice_pipeline(
    session_id: str,
    websocket,
    voice_config: str = "conversational",
    system_prompt: str = "default",
    custom_config: Optional[Dict[str, Any]] = None,
) -> PipelineTask:
    """
    Convenience function to create a voice pipeline

    Args:
        session_id: Unique session identifier
        websocket: FastAPI WebSocket connection
        voice_config: Voice configuration key
        system_prompt: System prompt key
        custom_config: Optional custom configuration

    Returns:
        PipelineTask ready to run
    """
    service = get_pipeline_service()
    return await service.create_pipeline(
        session_id, websocket, voice_config, system_prompt, custom_config
    )
