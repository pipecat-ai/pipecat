#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld AI Text-to-Speech Service Implementation.

This module provides integration with Inworld AI's HTTP-based TTS API, enabling
both streaming and non-streaming text-to-speech synthesis with high-quality,
natural-sounding voices.

Key Features:

- HTTP streaming and non-streaming API support for flexible audio generation
- Multiple voice options (Ashley, Hades, etc.)
- Automatic language detection from input text (no manual language setting required)
- Real-time audio chunk processing with proper buffering
- WAV header handling and audio format conversion
- Comprehensive error handling and metrics tracking

Technical Implementation:

- Uses aiohttp for HTTP connections
- Implements both JSON line-by-line parsing (streaming) and complete response (non-streaming)
- Handles base64-encoded audio data with proper decoding
- Manages audio continuity to prevent clicks and artifacts
- Integrates with Pipecat's frame-based pipeline architecture

Examples::

    async with aiohttp.ClientSession() as session:
        # Streaming mode (default) - real-time audio generation
        tts = InworldTTSService(
            api_key=os.getenv("INWORLD_API_KEY"),
            aiohttp_session=session,
            voice_id="Ashley",
            model="inworld-tts-1",
            streaming=True,  # Default
            params=InworldTTSService.InputParams(
                temperature=1.1,  # Optional: control synthesis variability (range: [0, 2])
            ),
        )

        # Non-streaming mode - complete audio generation then playback
        tts = InworldTTSService(
            api_key=os.getenv("INWORLD_API_KEY"),
            aiohttp_session=session,
            voice_id="Ashley",
            model="inworld-tts-1",
            streaming=False,
            params=InworldTTSService.InputParams(
                temperature=1.1,
            ),
        )
"""

import asyncio
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Inworld WebSocket TTS, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class InworldTTSService(TTSService):
    """Inworld AI HTTP-based Text-to-Speech Service.

    This unified service integrates Inworld AI's high-quality TTS API with Pipecat's pipeline
    architecture. It supports both streaming and non-streaming modes, providing flexible
    speech synthesis with natural-sounding voices.

    Key Features:

    - **Streaming Mode**: Real-time HTTP streaming for minimal latency
    - **Non-Streaming Mode**: Complete audio synthesis then chunked playback
    - Multiple voice options (Ashley, Hades, etc.)
    - High-quality audio output (48kHz LINEAR16 PCM)
    - Automatic audio format handling and header stripping
    - Comprehensive error handling and recovery
    - Built-in performance metrics and monitoring
    - Unified interface for both modes

    Technical Architecture:

    - Uses aiohttp for non-blocking HTTP requests
    - **Streaming**: Implements JSON line-by-line streaming protocol
    - **Non-Streaming**: Single HTTP POST with complete response
    - Processes base64-encoded audio chunks in real-time or batch
    - Manages audio continuity to prevent artifacts
    - Integrates with Pipecat's frame-based pipeline system

    Supported Configuration:

    - Voice Selection: Ashley, Hades, and other Inworld voices
    - Models: inworld-tts-1 and other available models
    - Audio Formats: LINEAR16 PCM at various sample rates
    - Language Detection: Automatically inferred from input text (no explicit language setting required)
    - Mode Selection: streaming=True for real-time, streaming=False for complete synthesis

    Examples::

        async with aiohttp.ClientSession() as session:
            # Streaming mode (default) - Real-time audio generation
            tts_streaming = InworldTTSService(
                api_key=os.getenv("INWORLD_API_KEY"),
                aiohttp_session=session,
                voice_id="Ashley",
                model="inworld-tts-1",
                streaming=True,  # Default behavior
                params=InworldTTSService.InputParams(
                    temperature=1.1,  # Add variability to speech synthesis (range: [0, 2])
                ),
            )

            # Non-streaming mode - Complete audio then playback
            tts_complete = InworldTTSService(
                api_key=os.getenv("INWORLD_API_KEY"),
                aiohttp_session=session,
                voice_id="Hades",
                model="inworld-tts-1-max",
                streaming=False,
                params=InworldTTSService.InputParams(
                    temperature=1.1,
                ),
            )
    """

    class InputParams(BaseModel):
        """Optional input parameters for Inworld TTS configuration.

        Parameters:
            temperature: Voice temperature control for synthesis variability (e.g., 1.1).
                        Valid range: [0, 2]. Higher values increase variability.
            speaking_rate: Speaking speed control (range: [0.5, 1.5]). Defaults to 1.0 when
                           unset.

        Note:
            Language is automatically inferred from the input text by Inworld's TTS models,
            so no explicit language parameter is required.
        """

        temperature: Optional[float] = None  # optional temperature control (range: [0, 2])
        speaking_rate: Optional[float] = None  # optional speaking rate control (range: [0.5, 1.5])

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = "Ashley",
        model: str = "inworld-tts-1",
        streaming: bool = True,
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Inworld TTS service.

        Sets up the TTS service with Inworld AI's API configuration.
        This constructor prepares all necessary parameters for speech synthesis.

        Args:
            api_key: Inworld API key for authentication (base64-encoded from Inworld Portal).
                    Get this from: Inworld Portal > Settings > API Keys > Runtime API Key
            aiohttp_session: Shared aiohttp session for HTTP requests. Must be provided
                           for proper connection pooling and resource management.
            voice_id: Voice selection for speech synthesis. Common options include:
                     - "Ashley": Clear, professional female voice (default)
                     - "Hades": Deep, authoritative male voice
                     - And many more available in your Inworld account
            model: TTS model to use for speech synthesis:
                  - "inworld-tts-1": Standard quality model (default)
                  - "inworld-tts-1-max": Higher quality model
                  - Other models as available in your Inworld account
            streaming: Whether to use streaming mode (True) or non-streaming mode (False).
                      - True: Real-time audio chunks as they're generated (lower latency)
                      - False: Complete audio file generated first, then chunked for playback (simpler)
                      The base URL is automatically selected based on this mode:
                      - Streaming: "https://api.inworld.ai/tts/v1/voice:stream"
                      - Non-streaming: "https://api.inworld.ai/tts/v1/voice"
            sample_rate: Audio sample rate in Hz. If None, uses default from StartFrame.
                        Common values: 48000 (high quality), 24000 (good quality), 16000 (basic)
            encoding: Audio encoding format. Supported options:
                     - "LINEAR16" (default) - Uncompressed PCM, best quality
                     - Other formats as supported by Inworld API
            params: Optional input parameters for additional configuration. Use this to specify:
                   - temperature: Voice temperature control for variability (range: [0, 2], e.g., 1.1, optional)
                   - speaking_rate: Set desired speaking speed (range: [0.5, 1.5], optional)
                   Language is automatically inferred from input text.
            **kwargs: Additional arguments passed to the parent TTSService class.

        Note:
            The aiohttp_session parameter is required because Inworld's HTTP API
            benefits from connection reuse and proper async session management.
        """
        # Initialize parent TTSService with audio configuration
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Use provided params or create default configuration
        params = params or InworldTTSService.InputParams()

        # Store core configuration for API requests
        self._api_key = api_key  # Authentication credentials
        self._session = aiohttp_session  # HTTP session for requests
        self._streaming = streaming  # Streaming mode selection

        # Set base URL based on streaming mode
        if streaming:
            self._base_url = "https://api.inworld.ai/tts/v1/voice:stream"  # Streaming endpoint
        else:
            self._base_url = "https://api.inworld.ai/tts/v1/voice"  # Non-streaming endpoint

        # Build settings dictionary that matches Inworld's API expectations
        # This will be sent as JSON payload in each TTS request
        # Note: Language is automatically inferred from text by Inworld's models
        self._settings = {
            "voiceId": voice_id,  # Voice selection from direct parameter
            "modelId": model,  # TTS model selection from direct parameter
            "audioConfig": {  # Audio format configuration
                "audioEncoding": encoding,  # Format: LINEAR16, MP3, etc.
                "sampleRateHertz": 0,  # Will be set in start() from parent service
            },
        }

        # Add optional temperature parameter if provided (valid range: [0, 2])
        if params and params.temperature is not None:
            self._settings["temperature"] = params.temperature
        # Add optional speaking rate if provided (valid range: [0.5, 1.5])
        if params and params.speaking_rate is not None:
            self._settings["audioConfig"]["speakingRate"] = params.speaking_rate

        # Register voice and model with parent service for metrics and tracking
        self.set_voice(voice_id)  # Used for logging and metrics
        self.set_model_name(model)  # Used for performance tracking

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["audioConfig"]["sampleRateHertz"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        """Stop the Inworld TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Inworld's HTTP API.

        This is the core TTS processing function that adapts its behavior based on the streaming mode:

        **Streaming Mode (streaming=True)**:
        1. Sends text to Inworld's streaming TTS endpoint
        2. Receives JSON-streamed audio chunks in real-time
        3. Processes and cleans audio data (removes WAV headers, validates content)
        4. Yields audio frames for immediate playback in the pipeline

        **Non-Streaming Mode (streaming=False)**:
        1. Sends text to Inworld's non-streaming TTS endpoint
        2. Receives complete audio file as base64-encoded response
        3. Processes entire audio and chunks for playback
        4. Yields audio frames in manageable pieces

        Technical Details:

        - **Streaming**: Uses HTTP streaming with JSON line-by-line responses
        - **Non-Streaming**: Single HTTP POST with complete JSON response
        - Each audio chunk contains base64-encoded audio data
        - Implements buffering to handle partial data (streaming mode)
        - Strips WAV headers to prevent audio artifacts/clicks
        - Provides optimized audio delivery for each mode

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus control frames.

        Raises:
            ErrorFrame: If API errors occur or audio processing fails.
        """
        logger.debug(f"{self}: Generating TTS [{text}] (streaming={self._streaming})")

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
            "audioConfig": self._settings["audioConfig"],  # Audio format settings (LINEAR16, 48kHz)
        }

        # Add optional temperature parameter if configured (valid range: [0, 2])
        if "temperature" in self._settings:
            payload["temperature"] = self._settings["temperature"]

        # Set up HTTP headers for authentication and content type
        # Inworld requires Basic auth with base64-encoded API key
        headers = {
            "Authorization": f"Basic {self._api_key}",  # Base64 API key from Inworld Portal
            "Content-Type": "application/json",  # JSON request body
        }

        try:
            # ================================================================================
            # STEP 2: INITIALIZE METRICS AND PROCESSING
            # ================================================================================
            # Start measuring Time To First Byte (TTFB) for performance tracking
            await self.start_ttfb_metrics()

            # Signal to the pipeline that TTS generation has started
            # This allows downstream processors to prepare for incoming audio
            yield TTSStartedFrame()

            # ================================================================================
            # STEP 3: MAKE HTTP REQUEST (MODE-SPECIFIC)
            # ================================================================================
            # Use aiohttp to make request to Inworld's endpoint
            # Behavior differs based on streaming mode
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
                    yield ErrorFrame(error=f"Inworld API error: {error_text}")
                    return

                # ================================================================================
                # STEP 5: PROCESS RESPONSE (MODE-SPECIFIC)
                # ================================================================================
                # Choose processing method based on streaming mode
                if self._streaming:
                    # Stream processing: JSON line-by-line with real-time audio
                    async for frame in self._process_streaming_response(response):
                        yield frame
                else:
                    # Non-stream processing: Complete JSON response with batch audio
                    async for frame in self._process_non_streaming_response(response):
                        yield frame

            # ================================================================================
            # STEP 6: FINALIZE METRICS AND CLEANUP
            # ================================================================================
            # Start usage metrics tracking after successful completion
            await self.start_tts_usage_metrics(text)

        except Exception as e:
            # ================================================================================
            # STEP 7: ERROR HANDLING
            # ================================================================================
            # Log any unexpected errors and notify the pipeline
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            # ================================================================================
            # STEP 8: CLEANUP AND COMPLETION
            # ================================================================================
            # Always stop metrics tracking, even if errors occurred
            await self.stop_all_metrics()

            # Signal to pipeline that TTS generation is complete
            # This allows downstream processors to finalize audio processing
            yield TTSStoppedFrame()

    async def _process_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Frame, None]:
        """Process streaming JSON response with real-time audio chunks.

        This method handles Inworld's streaming endpoint response format:
        - JSON lines containing base64-encoded audio chunks
        - Real-time processing as data arrives
        - Line buffering to handle partial JSON data

        Args:
            response: The aiohttp response object from streaming endpoint.

        Yields:
            Frame: Audio frames as they're processed from the stream.
        """
        # ================================================================================
        # STREAMING: PROCESS JSON LINE-BY-LINE RESPONSE
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
            # BUFFER MANAGEMENT
            # ============================================================================
            # Decode binary chunk to text and add to our line buffer
            # Each chunk may contain partial JSON lines, so we need to accumulate
            buffer += chunk.decode("utf-8")

            # ============================================================================
            # LINE-BY-LINE JSON PROCESSING
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
                    # PARSE JSON AND EXTRACT AUDIO
                    # ================================================================
                    # Parse the JSON line - should contain audio data
                    chunk_data = json.loads(line_str)

                    # Check if this line contains audio content
                    # Inworld's response format: {"result": {"audioContent": "base64data"}}
                    if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                        # Process the audio chunk
                        await self.stop_ttfb_metrics()
                        async for frame in self._process_audio_chunk(
                            base64.b64decode(chunk_data["result"]["audioContent"])
                        ):
                            yield frame

                except json.JSONDecodeError:
                    # Ignore malformed JSON lines - streaming can have partial data
                    # This is normal in HTTP streaming scenarios
                    continue

    async def _process_non_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Frame, None]:
        """Process complete JSON response with full audio content.

        This method handles Inworld's non-streaming endpoint response format:
        - Single JSON response with complete base64-encoded audio
        - Full audio download then chunked playback
        - Simpler processing without line buffering

        Args:
            response: The aiohttp response object from non-streaming endpoint.

        Yields:
            Frame: Audio frames chunked from the complete audio.
        """
        # ================================================================================
        # NON-STREAMING: PARSE COMPLETE JSON RESPONSE
        # ================================================================================
        # Parse the complete JSON response containing base64 audio data
        response_data = await response.json()

        # ================================================================================
        # EXTRACT AND VALIDATE AUDIO CONTENT
        # ================================================================================
        # Extract the base64-encoded audio content from response
        if "audioContent" not in response_data:
            logger.error("No audioContent in Inworld API response")
            await self.push_error(ErrorFrame(error="No audioContent in response"))
            return

        # ================================================================================
        # DECODE AND PROCESS COMPLETE AUDIO DATA
        # ================================================================================
        # Decode the base64 audio data to binary
        audio_data = base64.b64decode(response_data["audioContent"])

        # Strip WAV header if present (Inworld may include WAV header)
        # This prevents audio clicks and ensures clean audio playback
        if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
            audio_data = audio_data[44:]

        # ================================================================================
        # CHUNK AND YIELD COMPLETE AUDIO FOR PLAYBACK
        # ================================================================================
        # Chunk the complete audio for streaming playback
        # This allows the pipeline to process audio in manageable pieces
        CHUNK_SIZE = self.chunk_size

        for i in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[i : i + CHUNK_SIZE]
            if len(chunk) > 0:
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )

    async def _process_audio_chunk(self, audio_chunk: bytes) -> AsyncGenerator[Frame, None]:
        """Process a single audio chunk (common logic for both modes).

        This method handles audio chunk processing that's common to both streaming
        and non-streaming modes:
        - WAV header removal
        - Audio validation
        - Frame creation and yielding

        Args:
            audio_chunk: Raw audio data bytes to process.

        Yields:
            Frame: Audio frame if chunk contains valid audio data.
        """
        # ========================================================
        # AUDIO DATA VALIDATION
        # ========================================================
        # Skip empty audio chunks that could cause discontinuities
        # Empty chunks can create gaps or clicks in audio playback
        if not audio_chunk:
            return

        # Start with the raw audio data
        audio_data = audio_chunk

        # ========================================================
        # WAV HEADER REMOVAL (CRITICAL FOR AUDIO QUALITY)
        # ========================================================
        # Each audio chunk may have its own WAV header (44 bytes)
        # These headers contain metadata and will sound like clicks if played
        # We must strip them from EVERY chunk, not just the first one
        if (
            len(audio_chunk) > 44  # Ensure chunk is large enough
            and audio_chunk.startswith(b"RIFF")  # Check for WAV header magic bytes
        ):
            # Remove the 44-byte WAV header to get pure audio data
            audio_data = audio_chunk[44:]

        # ========================================================
        # YIELD AUDIO FRAME TO PIPELINE
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


class InworldWebsocketTTSService(WebsocketTTSService):
    """Inworld AI WebSocket-based Text-to-Speech Service with context support.

    This service uses Inworld's WebSocket TTS API, which supports multiple
    independent audio contexts per connection. Each context can be flushed or
    closed independently, allowing concurrent synthesis streams over a single
    websocket.
    """

    class InputParams(BaseModel):
        """Optional WebSocket input parameters for Inworld TTS configuration.

        Parameters:
            temperature: Voice temperature control for synthesis variability (range: [0, 2]).
            speaking_rate: Speaking speed control (range: [0.5, 1.5]).
            apply_text_normalization: Toggle Inworld's text normalization ("ON", "OFF",
                                      or "APPLY_TEXT_NORMALIZATION_UNSPECIFIED").
            max_buffer_delay_ms: Maximum buffering delay (ms) before auto-flush.
            buffer_char_threshold: Character threshold that triggers auto-flush.
        """

        temperature: Optional[float] = None
        speaking_rate: Optional[float] = None
        apply_text_normalization: Optional[str] = None
        max_buffer_delay_ms: Optional[int] = None
        buffer_char_threshold: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "Ashley",
        model: str = "inworld-tts-1",
        url: str = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional",
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        params: Optional[InputParams] = None,
        **kwargs: Any,
    ):
        """Initialize the Inworld WebSocket TTS service."""
        super().__init__(
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or InworldWebsocketTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._settings: Dict[str, Any] = {
            "voiceId": voice_id,
            "modelId": model,
            "audioConfig": {
                "audioEncoding": encoding,
                "sampleRateHertz": 0,
            },
        }

        if params.temperature is not None:
            self._settings["temperature"] = params.temperature
        if params.speaking_rate is not None:
            self._settings["audioConfig"]["speakingRate"] = params.speaking_rate
        if params.apply_text_normalization is not None:
            self._settings["applyTextNormalization"] = params.apply_text_normalization

        self._buffer_settings = {
            "maxBufferDelayMs": params.max_buffer_delay_ms,
            "bufferCharThreshold": params.buffer_char_threshold,
        }

        self._receive_task = None
        self._audio_context_task = None
        self._contexts_queue: asyncio.Queue = asyncio.Queue()
        self._contexts: Dict[str, asyncio.Queue] = {}
        self._active_contexts = set()
        self._closing_contexts = set()  # Track contexts that have been sent close_context

        self.set_voice(voice_id)
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld WebSocket TTS service."""
        await super().start(frame)
        self._settings["audioConfig"]["sampleRateHertz"] = self.sample_rate
        self._create_audio_context_task()
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Inworld WebSocket TTS service."""
        await super().stop(frame)
        await self._stop_audio_context_task()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld WebSocket TTS service."""
        await super().cancel(frame)
        await self._stop_audio_context_task()
        await self._disconnect()

    async def flush_audio(self):
        """Flush and close any pending audio contexts.

        Uses close_context instead of flush_context to release server resources,
        since Inworld limits connections to 5 concurrent contexts max.
        """
        for context_id in list(self._active_contexts):
            await self._send_close_context(context_id)

    def _create_audio_context_task(self):
        if not self._audio_context_task:
            self._contexts_queue = asyncio.Queue()
            self._contexts = {}
            self._closing_contexts = set()
            self._audio_context_task = self.create_task(self._audio_context_task_handler())

    async def _stop_audio_context_task(self):
        if self._audio_context_task:
            await self.cancel_task(self._audio_context_task)
            self._audio_context_task = None

    async def _audio_context_task_handler(self):
        """Process audio contexts in order of creation."""
        running = True
        while running:
            context_id = await self._contexts_queue.get()

            if context_id:
                await self._handle_audio_context(context_id)
                self._active_contexts.discard(context_id)
                self._closing_contexts.discard(context_id)
                del self._contexts[context_id]

                silence = b"\x00" * self.sample_rate
                frame = TTSAudioRawFrame(
                    audio=silence, sample_rate=self.sample_rate, num_channels=1
                )
                await self.push_frame(frame)
            else:
                running = False

            self._contexts_queue.task_done()

    async def _handle_audio_context(self, context_id: str):
        AUDIO_CONTEXT_TIMEOUT = 3.0
        queue = self._contexts[context_id]
        running = True
        while running:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=AUDIO_CONTEXT_TIMEOUT)
                if frame:
                    await self.push_frame(frame)
                running = frame is not None
            except asyncio.TimeoutError:
                logger.trace(f"{self} time out on audio context {context_id}")
                break

    async def create_audio_context(self, context_id: str):
        """Register a new audio context."""
        await self._contexts_queue.put(context_id)
        self._contexts[context_id] = asyncio.Queue()
        self._active_contexts.add(context_id)
        logger.trace(f"{self} created audio context {context_id}")

    async def append_to_audio_context(self, context_id: str, frame: TTSAudioRawFrame):
        """Append audio to a context queue."""
        if self.audio_context_available(context_id):
            await self._contexts[context_id].put(frame)
        else:
            logger.warning(f"{self} unable to append audio to context {context_id}")

    async def remove_audio_context(self, context_id: str):
        """Mark the context as complete."""
        if self.audio_context_available(context_id):
            await self._contexts[context_id].put(None)
        else:
            logger.warning(f"{self} unable to remove context {context_id}")

    def audio_context_available(self, context_id: str) -> bool:
        """Check whether an audio context exists."""
        return context_id in self._contexts

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Inworld WebSocket TTS")
            headers = [("Authorization", f"Basic {self._api_key}")]
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} connection exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} connection error: {e}"))
            self._websocket = None

    async def _disconnect_websocket(self):
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"{self} disconnect warning: {e}")
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Inworld."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message")
                continue

            result = msg.get("result", {})
            ctx_id = result.get("contextId") or result.get("context_id")

            # Check for errors in status
            status = result.get("status", {})
            if status.get("code", 0) != 0:
                error_msg = status.get("message", "Unknown error")
                await self.push_error(ErrorFrame(error=f"Inworld API error: {error_msg}"))
                continue

            if "error" in msg:
                await self.push_error(ErrorFrame(error=str(msg["error"])))
                continue

            # Handle audio chunk - audio is nested in result.audioChunk.audioContent
            audio_chunk = result.get("audioChunk", {})
            audio_b64 = audio_chunk.get("audioContent")

            if audio_b64:
                await self.stop_ttfb_metrics()
                audio = base64.b64decode(audio_b64)
                if len(audio) > 44 and audio.startswith(b"RIFF"):
                    audio = audio[44:]
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)

                if ctx_id:
                    if not self.audio_context_available(ctx_id):
                        await self.create_audio_context(ctx_id)
                    await self.append_to_audio_context(ctx_id, frame)
                else:
                    await self.push_frame(frame)

            # Handle flush completed or context closed as final signals
            if "flushCompleted" in result or "contextClosed" in result:
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.remove_audio_context(ctx_id)
                await self.push_frame(TTSStoppedFrame())

    async def _send_context(self, context_id: str):
        """Send context creation/configuration message."""
        create_config = {
            "voiceId": self._settings["voiceId"],
            "modelId": self._settings["modelId"],
            "audioConfig": self._settings["audioConfig"],
        }

        if "temperature" in self._settings:
            create_config["temperature"] = self._settings["temperature"]
        if "applyTextNormalization" in self._settings:
            create_config["applyTextNormalization"] = self._settings["applyTextNormalization"]
        if self._buffer_settings["maxBufferDelayMs"] is not None:
            create_config["maxBufferDelayMs"] = self._buffer_settings["maxBufferDelayMs"]
        if self._buffer_settings["bufferCharThreshold"] is not None:
            create_config["bufferCharThreshold"] = self._buffer_settings["bufferCharThreshold"]

        msg = {"create": create_config, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_text(self, context_id: str, text: str):
        msg = {"send_text": {"text": text}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_flush(self, context_id: str):
        msg = {"flush_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_close_context(self, context_id: str):
        """Close a context to release server-side resources.

        Per Inworld docs: 'for each connection, 5 contexts is the max'.
        Contexts must be explicitly closed to avoid hitting this limit.
        Closing a context automatically flushes any remaining text first.
        """
        if context_id in self._closing_contexts:
            return  # Already sent close for this context
        self._closing_contexts.add(context_id)
        msg = {"close_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Inworld's WebSocket API."""
        logger.debug(f"{self}: Generating WebSocket TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            context_id = str(uuid.uuid4())
            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)

            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            await self._send_context(context_id)
            await self._send_text(context_id, text)
            # Use close_context instead of flush_context to release server resources.
            # Per Inworld docs: "5 contexts is the max" per connection, and
            # "Sending a close context message is equivalent to sending a flush
            # message right before, so all text in the buffer will be synthesized
            # before the context is closed."
            await self._send_close_context(context_id)
            await self.start_tts_usage_metrics(text)

            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
            yield TTSStoppedFrame()
