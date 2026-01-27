"""
Ojin TTS Pipecat Pipeline Example

A speech-to-speech example using:
- Local audio input/output
- Silero VAD for voice activity detection
- Whisper STT for speech-to-text (local)
- OpenAI LLM for conversation
- Ojin TTS for text-to-speech
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# Add parent directories to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseEndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.services.ojin.tts import OjinTTSService, OjinTTSServiceSettings
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams


class LLMFullResponseAggregator(FrameProcessor):
    """Aggregates LLM text tokens into a complete response before sending to TTS.
    
    This processor buffers all TextFrames until LLMFullResponseEndFrame is received,
    then emits a single TTSSpeakFrame with the complete text. This allows TTS to
    have full context for better prosody and intonation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            # Buffer text tokens
            self._buffer += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            # LLM finished - send complete text to TTS
            if self._buffer.strip():
                logger.debug(f"Sending full response to TTS: {self._buffer[:100]}...")
                await self.push_frame(TTSSpeakFrame(text=self._buffer))
            self._buffer = ""
            # Pass through the end frame
            await self.push_frame(frame, direction)
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    """Run the speech-to-speech pipeline."""
    
    # Validate environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ojin_api_key = os.getenv("OJIN_API_KEY")
    ojin_config_id = os.getenv("OJIN_TTS_CONFIG_ID")
    ws_url = os.getenv("OJIN_WS_URL", "wss://models.ojin.ai/realtime")
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return
    if not ojin_api_key:
        logger.error("OJIN_API_KEY not set in environment")
        return
    if not ojin_config_id:
        logger.error("OJIN_TTS_CONFIG_ID not set in environment")
        return

    # Local audio transport for microphone input and speaker output
    audio_transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,  # Match Ojin TTS sample rate
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    )

    # Speech-to-Text using local Whisper (CPU mode for compatibility)
    stt = WhisperSTTService(
        model=Model.TINY,  # Use tiny model for faster inference
        device="cpu",      # Force CPU to avoid CUDA dependency
    )

    # LLM using OpenAI
    llm = OpenAILLMService(
        api_key=openai_api_key,
        model="gpt-4o-mini",  # Fast and cheap model
    )

    # Text-to-Speech using Ojin
    tts = OjinTTSService(
        OjinTTSServiceSettings(
            ws_url=ws_url,
            api_key=ojin_api_key,
            config_id=ojin_config_id,
            sample_rate=24000,
        )
    )

    # System prompt for the assistant
    messages = [
        {
            "role": "system",
            "content": """You are a helpful and expressive voice assistant. Keep your responses concise 
                        and conversational since they will be spoken aloud. Aim for 1-3 sentences per response.

                        IMPORTANT: Add emotion tags to your responses to make them more expressive. Use these tags inline 
                        where appropriate to convey emotion:
                        - <giggle> - for light amusement
                        - <laugh> - for laughter
                        - <chuckle> - for soft laughter  
                        - <laugh_harder> - for strong laughter
                        - <excited> - for excitement
                        - <curious> - for curiosity
                        - <whisper> - for whispering
                        - <gasp> - for surprise
                        - <exhale> - for sighing or relief
                        - <angry> - for anger
                        - <sarcastic> - for sarcasm

                        Place emotion tags BEFORE the text they should affect. Examples:
                        - "Yes! <laugh> I am indeed a bot!"
                        - "<whisper> Can I tell you a secret?"
                        - "<excited> That's amazing news!"
                        - "Oh really? <sarcastic> How surprising."

                        Use emotions naturally and sparingly - not every sentence needs one.""",
        },
    ]

    # Create context aggregator for managing conversation history
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Aggregator to buffer full LLM response before TTS
    response_aggregator = LLMFullResponseAggregator()

    # Build the pipeline:
    # Audio In -> STT -> User Context -> LLM -> Aggregator -> TTS -> Audio Out -> Assistant Context
    pipeline = Pipeline(
        [
            audio_transport.input(),       # Microphone input + VAD
            stt,                           # Speech-to-Text (Whisper)
            context_aggregator.user(),     # Add user message to context
            llm,                           # LLM generates response tokens
            response_aggregator,           # Buffer full response before TTS
            tts,                           # Ojin TTS converts to audio
            audio_transport.output(),      # Speaker output
            context_aggregator.assistant(), # Add assistant response to context
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
        ),
    )

    # Run the pipeline (handle_sigint=False for Windows compatibility)
    runner = PipelineRunner(handle_sigint=False)
    
    logger.info("=" * 50)
    logger.info("Speech-to-Speech Pipeline Started!")
    logger.info("Speak into your microphone to interact.")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 50)

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Pipeline stopped.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    asyncio.run(main())
