#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld AI Text-to-Speech Service Implementation.

This module provides integration with Inworld AI's HTTP-based TTS API, enabling
real-time text-to-speech synthesis with high-quality, natural-sounding voices.

Key Features:

- HTTP streaming API support for low-latency audio generation
- Multiple voice options (Ashley, Hades, etc.)
- Automatic language detection from input text (no manual language setting required)
- Real-time audio chunk processing with proper buffering
- WAV header handling and audio format conversion
- Comprehensive error handling and metrics tracking

Technical Implementation:

- Uses aiohttp for HTTP streaming connections
- Implements JSON line-by-line parsing for streaming responses
- Handles base64-encoded audio data with proper decoding
- Manages audio continuity to prevent clicks and artifacts
- Integrates with Pipecat's frame-based pipeline architecture

Usage::

    async with aiohttp.ClientSession() as session:
        tts = InworldHttpTTSService(
            api_key=os.getenv("INWORLD_API_KEY"),
            aiohttp_session=session,
            params=InworldHttpTTSService.InputParams(
                voice_id="Ashley",
                model="inworld-tts-1",
            ),
        )
"""

import base64
import io
import json
import uuid
import warnings
from typing import AsyncGenerator, List, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService, TTSService
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts


class InworldHttpTTSService(TTSService):
    """Inworld AI HTTP-based Text-to-Speech Service.

    This service integrates Inworld AI's high-quality TTS API with Pipecat's pipeline
    architecture. It provides real-time speech synthesis with natural-sounding voices
    and low-latency streaming audio delivery.

    Key Features:

    - Real-time HTTP streaming for minimal latency
    - Multiple voice options (Ashley, Hades, etc.)
    - High-quality audio output (48kHz LINEAR16 PCM)
    - Automatic audio format handling and header stripping
    - Comprehensive error handling and recovery
    - Built-in performance metrics and monitoring

    Technical Architecture:

    - Uses aiohttp for non-blocking HTTP requests
    - Implements JSON line-by-line streaming protocol
    - Processes base64-encoded audio chunks in real-time
    - Manages audio continuity to prevent artifacts
    - Integrates with Pipecat's frame-based pipeline system

    Supported Configuration:

    - Voice Selection: Ashley, Hades, and other Inworld voices
    - Models: inworld-tts-1 and other available models
    - Audio Formats: LINEAR16 PCM at various sample rates
    - Language Detection: Automatically inferred from input text (no explicit language setting required)

    Example Usage::

        async with aiohttp.ClientSession() as session:
            # Using default settings (Ashley voice, inworld-tts-1 model)
            tts = InworldHttpTTSService(
                api_key=os.getenv("INWORLD_API_KEY"),
                aiohttp_session=session,
            )

            # Or with custom voice and model via params
            params = InworldHttpTTSService.InputParams(
                voice_id="Hades",
                model="inworld-tts-1-max",
            )
            tts = InworldHttpTTSService(
                api_key=os.getenv("INWORLD_API_KEY"),
                aiohttp_session=session,
                params=params,
            )
    """

    class InputParams(BaseModel):
        """Input parameters for Inworld HTTP TTS configuration.

        Parameters:
            voice_id: Voice selection for speech synthesis (e.g., "Ashley", "Hades").
            model: TTS model to use (e.g., "inworld-tts-1", "inworld-tts-1-max").

        Note:
            Language is automatically inferred from the input text by Inworld's TTS models,
            so no explicit language parameter is required.
        """

        voice_id: Optional[str] = "Ashley"  # defaults to the Ashley voice
        model: Optional[str] = "inworld-tts-1"  # defaults to the inworld-tts-1 model

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.inworld.ai/tts/v1/voice:stream",
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Inworld HTTP TTS service.

        Sets up the TTS service with Inworld AI's streaming API configuration.
        This constructor prepares all necessary parameters for real-time speech synthesis.

        Args:
            api_key: Inworld API key for authentication (base64-encoded from Inworld Portal).
                    Get this from: Inworld Portal > Settings > API Keys > Runtime API Key
            aiohttp_session: Shared aiohttp session for HTTP requests. Must be provided
                           for proper connection pooling and resource management.
            base_url: Base URL for Inworld HTTP API. Uses streaming endpoint by default.
                     Should normally not be changed unless using a different environment.
            sample_rate: Audio sample rate in Hz. If None, uses default from StartFrame.
                        Common values: 48000 (high quality), 24000 (good quality), 16000 (basic)
            encoding: Audio encoding format. Supported options:
                     - "LINEAR16" (default) - Uncompressed PCM, best quality
                     - Other formats as supported by Inworld API
            params: Input parameters for voice and model configuration. Use this to specify:
                   - voice_id: Voice selection ("Ashley", "Hades", etc.)
                   - model: TTS model ("inworld-tts-1", "inworld-tts-1-max", etc.)
                   If None, uses default values (Ashley voice, inworld-tts-1 model).
                   Note: Language is automatically inferred from input text.
            **kwargs: Additional arguments passed to the parent TTSService class.

        Note:
            The aiohttp_session parameter is required because Inworld's HTTP API
            benefits from connection reuse and proper async session management.
        """
        # Initialize parent TTSService with audio configuration
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Use provided params or create default configuration
        params = params or InworldHttpTTSService.InputParams()

        # Store core configuration for API requests
        self._api_key = api_key  # Authentication credentials
        self._session = aiohttp_session  # HTTP session for requests
        self._base_url = base_url  # API endpoint URL

        # Build settings dictionary that matches Inworld's API expectations
        # This will be sent as JSON payload in each TTS request
        # Note: Language is automatically inferred from text by Inworld's models
        self._settings = {
            "voiceId": params.voice_id or "Ashley",  # Voice selection from params
            "modelId": params.model or "inworld-tts-1",  # TTS model selection from params
            "audio_config": {  # Audio format configuration
                "audio_encoding": encoding,  # Format: LINEAR16, MP3, etc.
                "sample_rate_hertz": 0,  # Will be set in start() from parent service
            },
        }

        # Register voice and model with parent service for metrics and tracking
        self.set_voice(params.voice_id or "Ashley")  # Used for logging and metrics
        self.set_model_name(params.model or "inworld-tts-1")  # Used for performance tracking

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld HTTP service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["audio_config"]["sample_rate_hertz"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        """Stop the Inworld HTTP TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld HTTP TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Inworld's streaming HTTP API.

        This is the core TTS processing function that:
        1. Sends text to Inworld's streaming TTS endpoint
        2. Receives JSON-streamed audio chunks in real-time
        3. Processes and cleans audio data (removes WAV headers, validates content)
        4. Yields audio frames for immediate playback in the pipeline

        Technical Details:

        - Uses HTTP streaming with JSON line-by-line responses
        - Each JSON line contains base64-encoded audio data
        - Implements buffering to handle partial JSON lines
        - Strips WAV headers to prevent audio artifacts/clicks
        - Provides real-time audio streaming for low latency

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus control frames.

        Raises:
            ErrorFrame: If API errors occur or audio processing fails.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # ================================================================================
        # STEP 1: PREPARE API REQUEST
        # ================================================================================
        # Build the JSON payload according to Inworld's API specification
        # This matches the format shown in their documentation examples
        # Note: Language is automatically inferred from the input text by Inworld's models
        payload = {
            "text": text,  # Text to synthesize
            "voiceId": self._settings["voiceId"],  # Voice selection (Ashley, Hades, etc.)
            "modelId": self._settings["modelId"],  # TTS model (inworld-tts-1)
            "audio_config": self._settings[
                "audio_config"
            ],  # Audio format settings (LINEAR16, 48kHz)
        }

        # Set up HTTP headers for authentication and content type
        # Inworld requires Basic auth with base64-encoded API key
        headers = {
            "Authorization": f"Basic {self._api_key}",  # Base64 API key from Inworld Portal
            "Content-Type": "application/json",  # JSON request body
        }

        try:
            # ================================================================================
            # STEP 2: INITIALIZE METRICS AND STREAMING
            # ================================================================================
            # Start measuring Time To First Byte (TTFB) for performance tracking
            await self.start_ttfb_metrics()

            # Signal to the pipeline that TTS generation has started
            # This allows downstream processors to prepare for incoming audio
            yield TTSStartedFrame()

            # Flag to track if we're processing the first audio chunk
            # Used for WAV header handling and debugging
            is_first_chunk = True

            # ================================================================================
            # STEP 3: MAKE HTTP STREAMING REQUEST
            # ================================================================================
            # Use aiohttp's streaming POST to Inworld's streaming endpoint
            # The endpoint returns JSON lines with audio chunks as they're generated
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                # ================================================================================
                # STEP 4: HANDLE HTTP ERRORS
                # ================================================================================
                # Check for API errors (expired keys, invalid requests, etc.)
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Inworld API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Inworld API error: {error_text}"))
                    return

                # ================================================================================
                # STEP 5: PROCESS STREAMING JSON RESPONSE
                # ================================================================================
                # Inworld streams JSON lines where each line contains audio data
                # We need to buffer incoming data and process complete lines

                # Buffer to accumulate incoming text data
                # This handles cases where JSON lines are split across HTTP chunks
                buffer = ""

                # Read HTTP response in manageable chunks (1KB each)
                # This prevents memory issues with large responses
                async for chunk in response.content.iter_chunked(1024):
                    if not chunk:
                        continue

                    # ============================================================================
                    # STEP 6: BUFFER MANAGEMENT
                    # ============================================================================
                    # Decode binary chunk to text and add to our line buffer
                    # Each chunk may contain partial JSON lines, so we need to accumulate
                    buffer += chunk.decode("utf-8")

                    # ============================================================================
                    # STEP 7: LINE-BY-LINE JSON PROCESSING
                    # ============================================================================
                    # Process all complete lines in the buffer (lines ending with \n)
                    # Leave partial lines in buffer for next iteration
                    while "\n" in buffer:
                        # Split on first newline, keeping remainder in buffer
                        line, buffer = buffer.split("\n", 1)
                        line_str = line.strip()

                        # Skip empty lines (common in streaming responses)
                        if not line_str:
                            continue

                        try:
                            # ================================================================
                            # STEP 8: PARSE JSON AND EXTRACT AUDIO
                            # ================================================================
                            # Parse the JSON line - should contain audio data
                            chunk_data = json.loads(line_str)

                            # Check if this line contains audio content
                            # Inworld's response format: {"result": {"audioContent": "base64data"}}
                            if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                                # Decode base64 audio data to binary
                                audio_chunk = base64.b64decode(chunk_data["result"]["audioContent"])

                                # ========================================================
                                # STEP 9: AUDIO DATA VALIDATION
                                # ========================================================
                                # Skip empty audio chunks that could cause discontinuities
                                # Empty chunks can create gaps or clicks in audio playback
                                if not audio_chunk:
                                    continue

                                # Start with the raw audio data
                                audio_data = audio_chunk

                                # ========================================================
                                # STEP 10: WAV HEADER REMOVAL (CRITICAL FOR AUDIO QUALITY)
                                # ========================================================
                                # Each audio chunk may have its own WAV header (44 bytes)
                                # These headers contain metadata and will sound like clicks if played
                                # We must strip them from EVERY chunk, not just the first one
                                if (
                                    len(audio_chunk) > 44  # Ensure chunk is large enough
                                    and audio_chunk.startswith(
                                        b"RIFF"
                                    )  # Check for WAV header magic bytes
                                ):
                                    # Remove the 44-byte WAV header to get pure audio data
                                    audio_data = audio_chunk[44:]

                                    # Track that we've seen our first chunk (for debugging)
                                    if is_first_chunk:
                                        is_first_chunk = False

                                # ========================================================
                                # STEP 11: YIELD AUDIO FRAME TO PIPELINE
                                # ========================================================
                                # Only yield frames with actual audio content
                                # Empty frames can cause pipeline issues
                                if len(audio_data) > 0:
                                    # Create Pipecat audio frame with processed audio data
                                    yield TTSAudioRawFrame(
                                        audio=audio_data,  # Clean audio without headers
                                        sample_rate=self.sample_rate,  # Configured sample rate (48kHz)
                                        num_channels=1,  # Mono audio
                                    )

                        except json.JSONDecodeError:
                            # Ignore malformed JSON lines - streaming can have partial data
                            # This is normal in HTTP streaming scenarios
                            continue

            # ================================================================================
            # STEP 12: FINALIZE METRICS AND CLEANUP
            # ================================================================================
            # Start usage metrics tracking after successful completion
            await self.start_tts_usage_metrics(text)

        except Exception as e:
            # ================================================================================
            # STEP 13: ERROR HANDLING
            # ================================================================================
            # Log any unexpected errors and notify the pipeline
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            # ================================================================================
            # STEP 14: CLEANUP AND COMPLETION
            # ================================================================================
            # Always stop metrics tracking, even if errors occurred
            await self.stop_ttfb_metrics()

            # Signal to pipeline that TTS generation is complete
            # This allows downstream processors to finalize audio processing
            yield TTSStoppedFrame()
