#
# pipecat/services/soniox.py
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    StartInterruptionFrame,
    StopTaskFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADActiveFrame,
    VADInactiveFrame,
    VoicemailFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.string import is_equivalent_basic
from pipecat.services.stt_filters import contains_interruption_word
from pipecat.utils.text import voicemail

try:
    import websockets
except ModuleNotFoundError:
    logger.error("In order to use Soniox, you need to `pip install pipecat-ai[soniox]`")
    raise

# Constants for optimized behavior (matching Deepgram patterns)
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 2
IGNORE_REPEATED_MSG_AT_START_SECONDS = 1.0  # Reduced from 4 to 1 second - more conservative for Soniox
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3

def language_to_soniox_language(language: Language) -> Optional[str]:
    """Maps Pipecat Language enum to Soniox language codes."""
    # Soniox uses standard IETF language tags (e.g., "en", "es", "fr-CA")
    # We can mostly pass the language code through directly.
    return language.value

class SonioxSTTService(STTService):
    """
    Optimized Soniox STT Service following Deepgram's proven patterns.
    
    This implementation uses the best practices from Deepgram for:
    - Fast response mode with minimal latency
    - Advanced transcript filtering and accumulation
    - Bot speaking state tracking
    - Voicemail detection
    - Comprehensive performance metrics
    - False interim detection
    """

    class InputParams(BaseModel):
        language: Language = Field(default=Language.EN_US)
        model: str = "stt-rt-v4"
        language_hints: List[str] = ["es"]
        audio_format: str = "pcm_s16le"  # PCM 16-bit little-endian for best latency
        num_channels: int = 1  # Number of audio channels (1 for mono, 2 for stereo) - MUST match Soniox API
        enable_speaker_diarization: bool = True
        enable_language_identification: bool = False
        allow_interruptions: bool = True
        detect_voicemail: bool = True
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS
        context: Optional[str] = None  # Optional context for better accuracy

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int = 16000,
        params: InputParams = None,
        # Legacy parameters for backward compatibility
        language: Language = None,
        enable_partials: bool = None,
        allow_interruptions: bool = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None
        self._connection_active = False

        # Handle legacy parameters or use params
        if params is None:
            params = self.InputParams()
            if language is not None:
                params.language = language
            if allow_interruptions is not None:
                params.allow_interruptions = allow_interruptions

        self._params = params
        self._language = params.language
        self._allow_stt_interruptions = params.allow_interruptions
        self.detect_voicemail = params.detect_voicemail
        self._on_no_punctuation_seconds = params.on_no_punctuation_seconds

        # State tracking (following Deepgram pattern)
        self._user_speaking = False
        self._bot_speaking = True  # Start as True like Deepgram
        self._bot_has_ever_spoken = False  # Track if bot has spoken at least once
        self._bot_started_speaking_time = None  # Track when bot started speaking
        self._bot_stopped_speaking_time = None  # Track when bot stopped (for echo tail grace period)
        self._vad_active = False
        self._vad_inactive_time = None

        # Performance tracking
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._last_audio_chunk_time = None
        self._audio_chunk_count = 0

        # Transcript tracking
        self._last_interim_time = None
        self._last_sent_transcript = None

        # Token accumulation (like the working example)
        self._final_tokens = []  # Accumulate final tokens until endpoint
        self._last_token_time = None
        self._last_any_token_time = None  # Track ANY token to prevent premature timeout

        # First message tracking
        self._first_message = None
        self._first_message_time = None
        self._was_first_transcript_receipt = False

        # Transcript accumulation (Deepgram compatibility)
        self._accum_transcription_frames = []  # Accumulate TranscriptionFrame objects
        self._last_time_accum_transcription = time.time()

        # Diarization tracking: detect if Soniox ever sees a speaker other than "1"
        self._diarization_detected = False

        self.start_time = time.time()
        self._last_time_transcription = time.time()

        logger.info("Soniox STT Service initialized (Deepgram-optimized):")
        logger.info(f"  Model: {params.model}, Language: {params.language.value}")
        logger.info(f"  Allow interruptions: {self._allow_stt_interruptions}")
        logger.info(f"  Detect voicemail: {self.detect_voicemail}")
        logger.info(f"  No punctuation timeout: {self._on_no_punctuation_seconds}s")

    @property
    def diarization_detected(self) -> bool:
        """Whether Soniox detected a speaker other than '1' during the call."""
        return self._diarization_detected

    def get_allow_interruptions(self) -> bool:
        """Get the current allow_interruptions setting."""
        return self._allow_stt_interruptions

    def set_allow_interruptions(self, value: bool):
        """Set the allow_interruptions setting."""
        logger.debug(f"SonioxSTTService: set_allow_interruptions({value})")
        self._allow_stt_interruptions = value

    def can_generate_metrics(self) -> bool:
        return True

    async def set_language(self, language: Language):
        """Switch language for transcription."""
        logger.info(f"Switching STT language to: [{language}]")
        self._language = language
        # Need to reconnect with new language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()
        
        # Start async handler for fast response and timeout management
        if not self._async_handler_task:
            self._async_handler_task = self.create_task(self._async_handler("async_handler"))

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        Streams audio data to Soniox websocket for real-time transcription.

        :param audio: Audio data as bytes (PCM 16-bit)
        :yield: None (transcription frames are pushed via callbacks)
        """
        if not self._websocket or not self._connection_active:
            logger.debug("⚠️  WebSocket not connected, skipping audio chunk")
            logger.debug(f"   WebSocket exists: {self._websocket is not None}")
            logger.debug(f"   Connection active: {self._connection_active}")
            yield None
            return

        # Don't start new speech detection if bot is speaking (prevents echo/noise false triggers)
        if self._current_speech_start_time is None and not self._bot_speaking:
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.info("=" * 70)
            logger.info("🎤 SPEECH DETECTION STARTED")
            logger.info("=" * 70)
        # elif self._current_speech_start_time is None and self._bot_speaking:
        #     logger.debug(f"⚠️  Ignoring audio while bot is speaking (prevents false speech detection)")

        # Only track audio chunks if we have an active speech detection cycle
        if self._current_speech_start_time is not None:
            self._audio_chunk_count += 1
            self._last_audio_chunk_time = time.time()

            # Log every 50 chunks to avoid spam
            if self._audio_chunk_count % 50 == 0:
                elapsed = time.perf_counter() - self._current_speech_start_time
                logger.debug(f"🎤 Audio streaming: {self._audio_chunk_count} chunks sent ({elapsed:.2f}s elapsed)")

        try:
            # logger.debug(f"📤 Sending audio chunk #{self._audio_chunk_count} ({len(audio)} bytes)")
            # Soniox accepts raw binary PCM audio after configuration
            await self._websocket.send(audio)
            # logger.debug(f"✓ Audio chunk sent successfully")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error("=" * 70)
            logger.error("❌ WEBSOCKET CONNECTION CLOSED WHILE SENDING AUDIO")
            logger.error("=" * 70)
            logger.error(f"Close Code: {e.code if hasattr(e, 'code') else 'N/A'}")
            logger.error(f"Close Reason: {e.reason if hasattr(e, 'reason') else 'N/A'}")
            logger.error(f"Chunks sent before disconnect: {self._audio_chunk_count}")
            logger.error("=" * 70)
            logger.warning("Attempting to reconnect...")
            await self._reconnect()
        except Exception as e:
            error_msg = str(e)
            logger.error("=" * 70)
            logger.error("❌ ERROR SENDING AUDIO TO SONIOX")
            logger.error("=" * 70)
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {error_msg}")
            logger.error(f"Audio chunk size: {len(audio)} bytes")
            logger.error(f"Chunks sent: {self._audio_chunk_count}")
            logger.exception("Full traceback:")
            logger.error("=" * 70)

            # Push ErrorFrame
            error_msg_lower = error_msg.lower()
            is_fatal = any(keyword in error_msg_lower for keyword in [
                'authentication', 'unauthorized', 'api key', 'quota'
            ])

            await self.push_error(ErrorFrame(
                error=f"Soniox audio send error: {error_msg[:200]}",
                fatal=is_fatal
            ))

        yield None

    async def _connect(self):
        """Establish websocket connection to Soniox service."""
        if self._websocket and self._connection_active:
            logger.debug("🔗 Already connected to Soniox, skipping reconnection")
            return

        try:
            logger.info("=" * 70)
            logger.info("🔗 SONIOX CONNECTION ATTEMPT")
            logger.info("=" * 70)
            
            # Validate API key before attempting connection
            if not self._api_key or self._api_key.strip() == "":
                raise ValueError("Soniox API key is empty or invalid")
            
            logger.info(f"✓ API Key validated: {self._api_key[:10]}...{self._api_key[-4:]}")
            logger.info(f"✓ Model: {self._params.model}")
            logger.info(f"✓ Language: {self._language} -> {language_to_soniox_language(self._language)}")
            logger.info(f"✓ Sample Rate: {self.sample_rate} Hz")
            logger.info(f"✓ Audio Format: {self._params.audio_format}")
            logger.info(f"✓ Audio Channels: {self._params.num_channels}")
            logger.info(f"✓ Speaker Diarization: {self._params.enable_speaker_diarization}")
            logger.info(f"✓ Language ID: {self._params.enable_language_identification}")

            # Use the correct Soniox WebSocket endpoint
            uri = "wss://stt-rt.soniox.com/transcribe-websocket"
            
            logger.info(f"📡 WebSocket URI: {uri}")
            logger.info("⏳ Attempting WebSocket connection...")

            # Connect to WebSocket (no auth headers needed, will send config message)
            self._websocket = await websockets.connect(uri)
            self._connection_active = True
            
            logger.info("✅ WebSocket connection established")
            logger.info("📤 Sending configuration message...")
            
            # Build language hints from the language parameter
            language_code = language_to_soniox_language(self._language)
            # Convert "en-US" to "en", "es-ES" to "es", etc.
            language_hint = language_code.split('-')[0] if language_code else "en"
            
            # Send configuration message as per Soniox documentation
            config = {
                "api_key": self._api_key,
                "model": self._params.model,
                "audio_format": self._params.audio_format,
                "sample_rate": self.sample_rate,
                "num_channels": self._params.num_channels,  # MUST be "num_channels" not "num_audio_channels"
                "language_hints": [language_hint],
                "enable_speaker_diarization": self._params.enable_speaker_diarization,
                "enable_language_identification": self._params.enable_language_identification,
                # CRITICAL: Enable endpoint detection like the working example
                "enable_endpoint_detection": True,
            }
            
            # Add context if provided
            if self._params.context:
                config["context"] = self._params.context
            
            logger.info(f"📤 Configuration: {json.dumps({k: v if k != 'api_key' else '***' for k, v in config.items()}, indent=2)}")
            
            # Send configuration as JSON text message
            await self._websocket.send(json.dumps(config))
            
            logger.info("⏳ Waiting for configuration confirmation from Soniox...")
            
            # Wait for confirmation response before proceeding
            try:
                response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                logger.info(f"📨 Received configuration response: {json.dumps(response_data, indent=2)}")
                
                # Check for error in confirmation
                if "error_code" in response_data:
                    raise Exception(f"Soniox config error: {response_data.get('error_message')}")
                    
            except asyncio.TimeoutError:
                raise Exception("Timeout waiting for Soniox configuration confirmation")
            
            logger.info("=" * 70)
            logger.info("✅ SUCCESSFULLY CONFIGURED SONIOX SESSION")
            logger.info("=" * 70)

            if not self._receive_task:
                logger.debug("🎧 Starting receive task handler...")
                self._receive_task = self.create_task(self._receive_task_handler())
            
            logger.debug("📊 Starting TTFB metrics...")
            await self.start_ttfb_metrics()
            logger.debug("📊 Starting processing metrics...")
            await self.start_processing_metrics()
            
            logger.info("✅ Soniox service fully initialized and ready")
            
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error("=" * 70)
            logger.error("❌ SONIOX CONNECTION FAILED - HTTP STATUS ERROR")
            logger.error("=" * 70)
            logger.error(f"Status Code: {e.status_code}")
            logger.error(f"Error Message: {e}")
            logger.error(f"Headers: {e.headers if hasattr(e, 'headers') else 'N/A'}")
            logger.error(f"API Key (masked): {self._api_key[:10]}...{self._api_key[-4:]}")
            logger.error(f"Endpoint: wss://api.soniox.com/v1/speech-to-text-rt")
            logger.error(f"Model: {self._params.model}")
            logger.error(f"Language: {language_to_soniox_language(self._language)}")
            logger.error("Possible Issues:")
            logger.error("  1. Invalid API key")
            logger.error("  2. Invalid model name")
            logger.error("  3. Unsupported language code")
            logger.error("  4. API endpoint URL incorrect")
            logger.error("  5. Account not active or insufficient credits")
            logger.error("=" * 70)
            # FATAL: Connection failed - terminate pipeline to prevent TTS waste
            await self.push_error(ErrorFrame(f"Soniox connection failed: {e}", fatal=True))
            await self._terminate_pipeline("STT connection failed - HTTP status error")

        except websockets.exceptions.InvalidURI as e:
            logger.error("=" * 70)
            logger.error("❌ SONIOX CONNECTION FAILED - INVALID URI")
            logger.error("=" * 70)
            logger.error(f"URI Error: {e}")
            logger.error(f"Attempted URI: {uri if 'uri' in locals() else 'Not constructed'}")
            logger.error("=" * 70)
            # FATAL: Connection failed - terminate pipeline to prevent TTS waste
            await self.push_error(ErrorFrame(f"Soniox invalid URI: {e}", fatal=True))
            await self._terminate_pipeline("STT connection failed - invalid URI")

        except Exception as e:
            logger.error("=" * 70)
            logger.error("❌ SONIOX CONNECTION FAILED - UNEXPECTED ERROR")
            logger.error("=" * 70)
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error(f"API Key (masked): {self._api_key[:10]}...{self._api_key[-4:] if len(self._api_key) > 14 else '****'}")
            logger.exception("Full traceback:")
            logger.error("=" * 70)
            # FATAL: Connection failed - terminate pipeline to prevent TTS waste
            await self.push_error(ErrorFrame(f"Soniox connection failed: {e}", fatal=True))
            await self._terminate_pipeline("STT connection failed - unexpected error")

    async def _disconnect(self):
        """Disconnect from Soniox service and clean up resources."""
        logger.info("=" * 70)
        logger.info("🔌 DISCONNECTING FROM SONIOX")
        logger.info("=" * 70)

        # Mark connection as inactive first to stop loops
        self._connection_active = False

        # Cancel async handler task
        if self._async_handler_task:
            logger.debug("⏹️  Cancelling async handler task...")
            try:
                await self.cancel_task(self._async_handler_task)
            except Exception as e:
                logger.debug(f"Error cancelling async handler task: {e}")
            finally:
                self._async_handler_task = None
                logger.debug("✓ Async handler task cancelled")

        # Cancel receive task
        if self._receive_task:
            logger.debug("⏹️  Cancelling receive task...")
            try:
                await self.cancel_task(self._receive_task)
            except Exception as e:
                logger.debug(f"Error cancelling receive task: {e}")
            finally:
                self._receive_task = None
                logger.debug("✓ Receive task cancelled")

        # Close websocket
        if self._websocket:
            logger.debug("🔌 Closing WebSocket connection...")
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            finally:
                self._websocket = None
                logger.info("✅ Disconnected from Soniox")

        logger.info("=" * 70)

    async def _reconnect(self):
        """Attempt to reconnect to Soniox service."""
        logger.warning("=" * 70)
        logger.warning("🔄 ATTEMPTING TO RECONNECT TO SONIOX")
        logger.warning("=" * 70)
        await self._disconnect()
        logger.info("⏳ Waiting 1 second before reconnection attempt...")
        await asyncio.sleep(1)
        await self._connect()

    async def _terminate_pipeline(self, reason: str):
        """Terminate the pipeline by pushing StopTaskFrame.

        This is called when STT connection fails to prevent the pipeline
        from continuing to consume TTS credits without functional STT.

        Args:
            reason: Human-readable reason for termination
        """
        logger.error("=" * 70)
        logger.error("🛑 TERMINATING PIPELINE DUE TO STT FAILURE")
        logger.error(f"   Reason: {reason}")
        logger.error("   This prevents unnecessary TTS (ElevenLabs) credit consumption")
        logger.error("=" * 70)

        # Push StopTaskFrame to signal pipeline termination
        # This will be caught by the pipeline runner and trigger graceful shutdown
        await self.push_frame(StopTaskFrame())

        logger.info("✅ StopTaskFrame pushed - pipeline will terminate")

    async def _receive_task_handler(self):
        """Handle incoming transcription messages from Soniox."""
        logger.info("🎧 Receive task handler started and listening for messages...")
        message_count = 0

        try:
            while self._connection_active:
                try:
                    logger.debug("⏳ Waiting for message from Soniox WebSocket...")
                    message = await self._websocket.recv()
                    message_count += 1
                    logger.debug(f"📨 Received message #{message_count} from Soniox")
                    logger.debug(f"📨 Raw message length: {len(message)} bytes")
                    logger.debug(f"📨 Raw message preview: {message[:200]}..." if len(message) > 200 else f"📨 Raw message: {message}")

                    await self._on_message(json.loads(message))

                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning("=" * 70)
                    logger.warning("⚠️  SONIOX CONNECTION CLOSED DURING RECEIVE")
                    logger.warning("=" * 70)
                    logger.warning(f"Close Code: {e.code if hasattr(e, 'code') else 'N/A'}")
                    logger.warning(f"Close Reason: {e.reason if hasattr(e, 'reason') else 'N/A'}")
                    logger.warning(f"Messages received before close: {message_count}")
                    logger.warning("=" * 70)
                    break

                except json.JSONDecodeError as e:
                    logger.error("=" * 70)
                    logger.error("❌ JSON DECODE ERROR")
                    logger.error("=" * 70)
                    logger.error(f"Error: {e}")
                    logger.error(f"Message that failed to parse: {message[:500] if 'message' in locals() else 'N/A'}")
                    logger.error("=" * 70)
                    continue

                except Exception as e:
                    error_msg = str(e)
                    logger.error("=" * 70)
                    logger.error("❌ ERROR IN SONIOX RECEIVE TASK")
                    logger.error("=" * 70)
                    logger.error(f"Error Type: {type(e).__name__}")
                    logger.error(f"Error Message: {error_msg}")
                    logger.error(f"Messages received before error: {message_count}")
                    logger.exception("Full traceback:")
                    logger.error("=" * 70)

                    # Push ErrorFrame before breaking
                    error_msg_lower = error_msg.lower()
                    is_fatal = any(keyword in error_msg_lower for keyword in [
                        'authentication', 'unauthorized', 'api key', 'quota', 'closed'
                    ])

                    await self.push_error(ErrorFrame(
                        error=f"Soniox receive error: {error_msg[:200]}",
                        fatal=is_fatal
                    ))
                    break

        except asyncio.CancelledError:
            logger.info(f"🛑 Receive task cancelled (received {message_count} messages)")
            # Ensure connection is marked inactive
            self._connection_active = False
            # Close websocket if still open
            if self._websocket:
                try:
                    await self._websocket.close()
                    logger.debug("✅ WebSocket closed during task cancellation")
                except Exception as e:
                    logger.debug(f"Error closing websocket during cancellation: {e}")
            raise
        finally:
            logger.info(f"🎧 Receive task handler stopped. Total messages received: {message_count}")

    def _is_fatal_error(self, error_code: int, error_message: str) -> bool:
        """Determine if an error is fatal based on code and message."""
        fatal_codes = [401, 402, 403, 429]
        fatal_keywords = [
            'authentication', 'unauthorized', 'api key', 'invalid key',
            'quota', 'insufficient', 'subscription', 'billing', 'payment'
        ]

        return (
            error_code in fatal_codes or
            any(keyword in error_message.lower() for keyword in fatal_keywords)
        )

    async def _handle_error_response(self, data: Dict) -> bool:
        """Handle error responses from Soniox API.

        Returns:
            True if an error was handled, False otherwise
        """
        if "error_code" not in data:
            return False

        error_code = data.get('error_code')
        error_message = data.get('error_message', '')
        is_fatal = self._is_fatal_error(error_code, error_message)

        if is_fatal:
            logger.error(f"❌ Soniox FATAL error {error_code}: {error_message}")
            await self.push_error(ErrorFrame(
                error=f"Soniox error: {error_message}",
                fatal=True
            ))
        else:
            logger.warning(f"⚠️ Soniox non-fatal error {error_code}: {error_message}")

        return True

    async def _handle_session_finished(self, data: Dict) -> bool:
        """Handle session finished response.

        Returns:
            True if session finished, False otherwise
        """
        if not data.get("finished"):
            return False

        logger.info("✅ Soniox session finished")
        await self._send_accumulated_tokens()
        return True

    async def _check_token_gap_endpoint(self):
        """Check if token gap indicates an endpoint and send if needed."""
        if not self._final_tokens:
            return

        elapsed_since_last_token = time.time() - (self._last_token_time or 0)
        if elapsed_since_last_token > 0.5:  # 500ms gap indicates endpoint
            logger.info("🔚 Detected endpoint (token gap) - sending accumulated tokens")
            await self._send_accumulated_tokens()

    def _separate_tokens(self, tokens: List[Dict]) -> tuple[List[Dict], bool]:
        """Separate tokens into final and non-final, filtering out empty/end tokens.

        Returns:
            Tuple of (non_final_tokens, has_final)
        """
        non_final_tokens = []
        has_final = False

        for token in tokens:
            text = token.get("text", "")
            if not text or text == "<end>":
                continue

            # Track diarization: detect if any token has a speaker other than "1"
            speaker = token.get("speaker")
            if not self._diarization_detected and speaker and speaker != "1":
                self._diarization_detected = True
                logger.info(f"🔊 Diarization detected: speaker '{speaker}' found in token '{text}'")

            # Log token confidence during bot speech for echo analysis
            confidence = token.get("confidence", 0)
            if self._bot_speaking or self._in_post_bot_grace_period():
                logger.info(f"🔍 Token while bot active: '{text}' confidence={confidence:.2f} speaker='{speaker}' final={token.get('is_final')}")

            # Track ANY token to prevent premature timeout
            self._last_any_token_time = time.time()

            if token.get("is_final"):
                self._final_tokens.append(token)
                has_final = True
                self._last_token_time = time.time()
            else:
                non_final_tokens.append(token)

        return non_final_tokens, has_final

    def _build_transcript_texts(self, non_final_tokens: List[Dict]) -> tuple[str, str, str]:
        """Build final, non-final, and combined transcript texts.

        Returns:
            Tuple of (final_text, non_final_text, combined_text)
        """
        final_text = "".join(t["text"] for t in self._final_tokens)
        non_final_text = "".join(t["text"] for t in non_final_tokens)
        combined_text = final_text + non_final_text

        return final_text, non_final_text, combined_text

    async def _process_tokens(self, tokens: List[Dict]):
        """Process tokens and determine whether to send final or interim transcript."""
        non_final_tokens, has_final = self._separate_tokens(tokens)
        final_text, non_final_text, combined_text = self._build_transcript_texts(non_final_tokens)

        logger.debug(f"📝 Final tokens accumulated: {len(self._final_tokens)}")
        logger.debug(f"📝 Non-final tokens: {len(non_final_tokens)}")
        logger.debug(f"📝 Combined text: '{combined_text}'")

        # Check if this is an endpoint (all final tokens, no non-finals)
        is_endpoint = has_final and len(non_final_tokens) == 0 and len(self._final_tokens) > 0

        if is_endpoint:
            logger.info("🔚 Detected endpoint (all final, no non-finals) - sending accumulated tokens")
            await self._send_accumulated_tokens()
        elif combined_text.strip():
            # CRITICAL: Check for voicemail FIRST, before any filtering
            # Voicemail messages often come when VAD is inactive (pre-recorded)
            # so they need priority over VAD-based filtering
            if await self._detect_and_handle_voicemail(combined_text):
                logger.info("🔊 Voicemail detected in interim tokens - returning early")
                return

            # Calculate average confidence for interims
            all_tokens = self._final_tokens + non_final_tokens
            confidence = sum(t.get("confidence", 1.0) for t in all_tokens) / len(all_tokens) if all_tokens else 1.0

            # Get time_start from first token if available
            time_start = all_tokens[0].get("start", 0) if all_tokens else 0

            should_ignore = await self._should_ignore_transcription(
                combined_text,
                is_final=False,  # This is an interim (not endpoint)
                confidence=confidence,
                time_start=time_start
            )
            if not should_ignore:
                await self._on_interim_transcript_message(combined_text, self._language)

    async def _on_message(self, data: Dict):
        """Process incoming transcription message.

        Following the working example pattern:
        - Accumulate final tokens until endpoint is detected
        - Send non-final tokens as interim immediately
        - Only send TranscriptionFrame when we detect endpoint (gap in tokens or explicit signal)
        """
        try:
            logger.debug("=" * 70)
            logger.debug("📨 PROCESSING SONIOX MESSAGE")
            logger.debug("=" * 70)
            logger.debug(f"Message keys: {list(data.keys())}")
            logger.debug(f"Full message data: {json.dumps(data, indent=2)}")

            # Handle error responses
            if await self._handle_error_response(data):
                return

            # Handle session finished
            if await self._handle_session_finished(data):
                return

            await self.stop_ttfb_metrics()

            # Parse and validate tokens
            tokens = data.get("tokens", [])
            logger.debug(f"📝 Tokens count: {len(tokens)}")

            if not tokens:
                await self._check_token_gap_endpoint()
                logger.debug("⚠️  No tokens in message, skipping")
                logger.debug("=" * 70)
                return

            # Process tokens
            await self._process_tokens(tokens)
            logger.debug("=" * 70)

        except Exception as e:
            error_msg = str(e)
            logger.error("=" * 70)
            logger.error("❌ ERROR PROCESSING SONIOX MESSAGE")
            logger.error("=" * 70)
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {error_msg}")
            logger.error(f"Message data: {data if 'data' in locals() else 'N/A'}")
            logger.exception("Full traceback:")
            logger.error("=" * 70)

            # Push ErrorFrame for message processing errors
            await self.push_error(ErrorFrame(
                error=f"Soniox message processing error: {error_msg[:200]}",
                fatal=False
            ))

    def _is_accum_transcription(self, text: str) -> bool:
        """Check if text should be accumulated (doesn't end with phrase-ending punctuation).

        Deepgram compatibility: Matches Deepgram's logic for accumulation.
        """
        END_OF_PHRASE_CHARACTERS = ['.', '?']

        text = text.strip()
        if not text:
            return True

        return text[-1] not in END_OF_PHRASE_CHARACTERS

    def _append_accum_transcription(self, frame: TranscriptionFrame):
        """Append a TranscriptionFrame to the accumulation buffer.

        Deepgram compatibility: Matches Deepgram's accumulation pattern.
        """
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)
        logger.debug(f"📝 Accumulated transcript: '{frame.text}' (total frames: {len(self._accum_transcription_frames)})")

    async def _send_accum_transcriptions(self):
        """Send all accumulated transcription frames followed by UserStoppedSpeakingFrame.

        Deepgram compatibility: This is the critical method that sends frames in the correct order:
        1. All accumulated TranscriptionFrames
        2. UserStoppedSpeakingFrame (to signal end of user speech)

        Note: We do NOT call _handle_user_speaking() here because it was already called
        in _on_final_transcript_message() before accumulating. Calling it again would
        trigger a second StartInterruptionFrame which causes pipeline task cancellations
        and race conditions that can lose the transcription frames.
        """
        if not len(self._accum_transcription_frames):
            logger.debug("No accumulated transcriptions to send")
            return

        logger.info("=" * 70)
        logger.info("📤 SENDING ACCUMULATED TRANSCRIPTIONS")
        logger.info(f"   Frame count: {len(self._accum_transcription_frames)}")
        logger.info("=" * 70)

        # Note: _handle_user_speaking() was already called in _on_final_transcript_message()
        # which triggered the StartInterruptionFrame. We only call it here as a fallback
        # if user is not marked as speaking (edge case from async timeout handler).
        if not self._user_speaking:
            logger.debug("⚠️  User not marked as speaking, triggering user speaking state")
            await self._handle_user_speaking()

        # Send all accumulated transcription frames
        for frame in self._accum_transcription_frames:
            logger.debug(f"📤 Sending accumulated frame: '{frame.text}'")
            await self.push_frame(frame)

        self._accum_transcription_frames = []

        # Send UserStoppedSpeakingFrame AFTER transcripts (critical for aggregator)
        await self._handle_user_silence()
        await self.stop_processing_metrics()

        logger.info("✅ All accumulated transcriptions sent")
        logger.info("=" * 70)

    async def _send_accumulated_tokens(self):
        """Send accumulated final tokens as a complete TranscriptionFrame.

        Soniox-specific: Converts Soniox tokens to transcript text and sends via
        _on_final_transcript_message(), which will then accumulate into frames and
        eventually send via _send_accum_transcriptions().

        This is called when we detect an endpoint (no more tokens coming).
        """
        if not self._final_tokens:
            logger.debug("No accumulated tokens to send")
            return

        # Build final transcript from accumulated tokens, filtering out <end> tokens
        transcript = "".join(t["text"] for t in self._final_tokens if t["text"] != "<end>")

        if not transcript.strip():
            logger.debug("Empty accumulated transcript, skipping")
            self._final_tokens = []
            return

        # Calculate confidence and time_start for filtering
        confidence = sum(t.get("confidence", 1.0) for t in self._final_tokens) / len(self._final_tokens) if self._final_tokens else 1.0
        time_start = self._final_tokens[0].get("start", 0) if self._final_tokens else 0

        # CRITICAL: Check if we should ignore this transcription
        should_ignore = await self._should_ignore_transcription(
            transcript,
            is_final=True,
            confidence=confidence,
            time_start=time_start
        )
        if should_ignore:
            logger.info("=" * 70)
            logger.info("🚫 IGNORING ACCUMULATED FINAL TRANSCRIPT")
            logger.info("=" * 70)
            logger.info(f"📝 '{transcript}'")
            logger.info(f"🔤 Tokens: {len(self._final_tokens)}, Chars: {len(transcript)}")
            logger.info(f"🎯 Confidence: {confidence:.2f}")
            logger.info("=" * 70)
            # Clear accumulated tokens for next utterance
            self._final_tokens = []
            self._last_token_time = None
            self._last_any_token_time = None
            return

        logger.info("=" * 70)
        logger.info("✅ SENDING ACCUMULATED FINAL TRANSCRIPT")
        logger.info("=" * 70)
        logger.info(f"📝 '{transcript}'")
        logger.info(f"🔤 Tokens: {len(self._final_tokens)}, Chars: {len(transcript)}")
        logger.info(f"🎯 Confidence: {confidence:.2f}")
        logger.info("=" * 70)

        # Record performance
        self._record_stt_performance(transcript, self._final_tokens)

        # Send the final transcription (will accumulate and send via _send_accum_transcriptions)
        await self._on_final_transcript_message(transcript, self._language)

        # Clear accumulated tokens for next utterance
        self._final_tokens = []
        self._last_token_time = None
        self._last_any_token_time = None
        self._last_final_transcript_time = time.time()
        self._last_time_transcription = time.time()

    async def _on_final_transcript_message(self, transcript: str, language: Language, speech_final: bool = True):
        """Handle final transcript - accumulate and send when appropriate.

        Deepgram compatibility: This now matches Deepgram's pattern exactly:
        1. Check for voicemail (early return if detected)
        2. Trigger user speaking state
        3. Create TranscriptionFrame and accumulate it
        4. Handle first message tracking
        5. Check if we should send accumulated frames based on punctuation or speech_final
        6. _send_accum_transcriptions() will send frames + UserStoppedSpeakingFrame in correct order
        """
        logger.debug("🔵 Processing final transcript...")

        # Check for voicemail detection first (Deepgram does this early)
        if await self._detect_and_handle_voicemail(transcript):
            logger.info("🔊 Voicemail detected and handled - returning early")
            return

        # Check for repeated first message BEFORE proceeding
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug(f"⏭️  Ignoring repeated first message: '{transcript}'")
            return

        # Trigger user speaking if not already active
        await self._handle_user_speaking()

        # Create transcription frame
        frame = TranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        logger.debug(f"📦 Created TranscriptionFrame: '{transcript}'")

        # Handle first message tracking
        self._handle_first_message(frame.text)

        # Accumulate the frame (Deepgram pattern)
        self._append_accum_transcription(frame)
        self._was_first_transcript_receipt = True

        # ALWAYS clear speech detection timer when processing final transcript
        self._current_speech_start_time = None

        # Fast response mode is always active - check if should send now
        logger.debug("🚀 Fast response mode - checking if should send now")
        await self._fast_response_send_accum_transcriptions()

    async def _on_interim_transcript_message(self, transcript: str, language: Language):
        """Handle interim transcript.

        Deepgram compatibility: Simplified to match Deepgram's pattern.
        Shows real-time progress and triggers user_speaking state.

        Note: Voicemail detection is handled in _process_tokens before this is called.
        """
        logger.debug(f"🟡 Interim: '{transcript[:50]}...' ({len(transcript)} chars)")

        # Update interim time for false interim detection
        self._last_interim_time = time.time()

        # Trigger user speaking (Deepgram always does this for interims)
        await self._handle_user_speaking()

        # Send interim frame
        frame = InterimTranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)
        logger.trace(f"📤 Sent InterimTranscriptionFrame: '{transcript[:30]}...'")

    def _record_stt_performance(self, transcript, tokens):
        """Record STT performance metrics."""
        if self._current_speech_start_time:
            elapsed = time.perf_counter() - self._current_speech_start_time
            self._stt_response_times.append(elapsed)
            # Calculate average confidence from tokens
            confidence = sum(t.get("confidence", 0) for t in tokens) / len(tokens) if tokens else 0
            logger.info(f"📊 ⚡ Soniox: ⏱️ STT Response Time: {elapsed:.3f}s")
            logger.info(f"   📝 Final Transcript: '{transcript}'")
            logger.info(f"   🎯 Avg. Confidence: {confidence:.2f}")
            logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
            self._current_speech_start_time = None

    async def _handle_user_speaking(self):
        """Handle user started speaking event.

        Push StartInterruptionFrame and UserStartedSpeakingFrame only when user
        is NOT already marked as speaking. This prevents multiple interruption
        cycles that can cause pipeline task cancellation race conditions.
        """
        if self._user_speaking:
            logger.debug("👤 User already marked as speaking, skipping interruption frames")
            return

        # CRITICAL: Set flag IMMEDIATELY to prevent race conditions
        # Multiple concurrent calls can happen within milliseconds, and if we
        # set this flag after the await, both calls will pass the check above
        self._user_speaking = True

        logger.info("=" * 70)
        logger.info("👤 USER STARTED SPEAKING")
        logger.info("=" * 70)

        # Push interruption frames AFTER setting the flag
        await self.push_frame(StartInterruptionFrame())
        await self.push_frame(UserStartedSpeakingFrame())

        logger.info("✓ User speaking state activated")
        logger.info("=" * 70)

    async def _handle_user_silence(self):
        """Handle user stopped speaking event.

        Deepgram compatibility: Push UserStoppedSpeakingFrame without explicit direction
        (uses default DOWNSTREAM), and reset speech timer.
        """
        if not self._user_speaking:
            logger.debug("👤 User already marked as not speaking, skipping")
            return

        # CRITICAL: Set flags IMMEDIATELY to prevent race conditions
        self._user_speaking = False
        self._current_speech_start_time = None

        logger.info("=" * 70)
        logger.info("👤 USER STOPPED SPEAKING")
        logger.info("=" * 70)

        # Push frame AFTER setting flags
        await self.push_frame(UserStoppedSpeakingFrame())

        logger.info("✓ User silence state activated")
        logger.info("=" * 70)
            
    async def _handle_bot_speaking(self):
        """Handle bot started speaking event."""
        self._bot_speaking = True
        self._bot_has_ever_spoken = True
        self._bot_started_speaking_time = time.time()
        self._bot_stopped_speaking_time = None  # Reset - bot is speaking now
        logger.debug(f"🤖 {self}: Bot started speaking at {self._bot_started_speaking_time}")

    async def _handle_bot_silence(self):
        """Handle bot stopped speaking event."""
        self._bot_speaking = False
        self._bot_started_speaking_time = None
        self._bot_stopped_speaking_time = time.time()  # Track for echo tail grace period
        logger.debug(f"🤖 {self}: Bot stopped speaking")

    # Confidence thresholds for echo rejection
    CONFIDENCE_INTERIM_MIN = 0.7       # Minimum confidence for interims (general)
    CONFIDENCE_BOT_SPEAKING_MIN = 0.80  # Minimum confidence while bot is speaking (echo rejection)
    CONFIDENCE_POST_BOT_MIN = 0.80      # Minimum confidence in grace period after bot stops
    POST_BOT_GRACE_SECONDS = 1.0        # Grace period after bot stops (catches echo tail)

    def _in_post_bot_grace_period(self) -> bool:
        """Check if we're within the grace period after the bot stopped speaking.
        Echo tail can finalize right after bot stops — this catches it.
        """
        if not self._bot_stopped_speaking_time:
            return False
        if self._bot_speaking:
            return False
        return (time.time() - self._bot_stopped_speaking_time) < self.POST_BOT_GRACE_SECONDS

    async def _should_ignore_transcription(self, transcript: str, is_final: bool, confidence: float = 1.0, time_start: float = 0) -> bool:
        """Check if transcription should be ignored based on various conditions.

        Uses confidence-based filtering to reject phone echo while bot is speaking.
        Echo audio is degraded (phone network round-trip) so Soniox gives it lower confidence.

        Args:
            transcript: The transcript text
            is_final: Whether this is a final transcript
            confidence: Confidence score (0.0 to 1.0)
            time_start: Start time of the transcript

        Returns:
            True if transcript should be ignored, False otherwise
        """
        word_count = self._transcript_words_count(transcript)
        char_count = len(transcript.strip())
        is_during_bot = self._bot_speaking
        is_post_bot = self._in_post_bot_grace_period()

        # Check 1: Low confidence interims (general baseline)
        if not is_final and confidence < self.CONFIDENCE_INTERIM_MIN:
            logger.debug(f"🔇 Ignoring interim, low confidence: {confidence:.2f} < {self.CONFIDENCE_INTERIM_MIN}: '{transcript}'")
            return True

        # Check 2: Bot speaking — require HIGH confidence to interrupt (echo rejection)
        # Phone echo goes through: bot speaker → phone mic → phone network → Soniox
        # This degrades audio quality, so echo gets lower confidence than real user speech
        if is_during_bot:
            if confidence < self.CONFIDENCE_BOT_SPEAKING_MIN:
                logger.info(f"🔇 Echo rejected (bot speaking): confidence {confidence:.2f} < {self.CONFIDENCE_BOT_SPEAKING_MIN}, '{transcript}'")
                return True
            # Also require substantial text to interrupt (short echo fragments)
            # BUT allow through if it contains a known interruption word
            if word_count <= 3 or char_count < 20:
                if contains_interruption_word(transcript, self._language):
                    logger.info(f"✅ Short but contains interruption word, allowing: '{transcript}'")
                else:
                    logger.info(f"🔇 Echo rejected (bot speaking, too short): words={word_count}, chars={char_count}, '{transcript}'")
                    return True

        # Check 3: Post-bot grace period — echo tail that finalizes right after bot stops
        if is_post_bot:
            time_since = time.time() - self._bot_stopped_speaking_time
            if confidence < self.CONFIDENCE_POST_BOT_MIN:
                logger.info(f"🔇 Echo tail rejected ({time_since:.1f}s after bot): confidence {confidence:.2f} < {self.CONFIDENCE_POST_BOT_MIN}, '{transcript}'")
                return True
            if word_count <= 2 or char_count < 15:
                if contains_interruption_word(transcript, self._language):
                    logger.info(f"✅ Short post-bot but contains interruption word, allowing: '{transcript}'")
                else:
                    logger.info(f"🔇 Echo tail rejected ({time_since:.1f}s after bot, too short): words={word_count}, chars={char_count}, '{transcript}'")
                    return True

        # Check 4: Fast greetings at start
        if self._should_ignore_fast_greeting(transcript, time_start):
            logger.debug(f"🔇 Ignoring fast greeting at start: '{transcript}'")
            return True

        # Check 5: Repeated first message
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug(f"🔇 Ignoring repeated first message: '{transcript}'")
            return True

        # Check 6: VAD inactive + interim (don't process interims when VAD is inactive)
        if not self._vad_active and not is_final:
            logger.debug(f"🔇 Ignoring interim - VAD inactive: '{transcript}'")
            return True

        # Check 7: Bot speaking + no interruptions allowed
        if self._bot_speaking and not self._allow_stt_interruptions:
            logger.debug(f"🔇 Ignoring transcript: bot speaking and interruptions disabled: '{transcript}'")
            return True

        # Log accepted transcripts with confidence for tuning
        if is_during_bot or is_post_bot:
            logger.info(f"✅ Accepted transcript (confidence={confidence:.2f}, words={word_count}, bot_speaking={is_during_bot}, post_bot={is_post_bot}): '{transcript}'")

        return False

    def _time_since_init(self):
        """Get time since service initialization."""
        return time.time() - self.start_time

    def _transcript_words_count(self, transcript: str):
        """Count words in transcript."""
        return len(transcript.split())

    def _should_ignore_fast_greeting(self, transcript: str, time_start: float) -> bool:
        """Ignore very fast greetings that might be false positives."""
        if time_start > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        # Common greeting words that might be false positives
        greeting_words = ["hello", "hi", "hey", "hola", "yes", "no"]
        transcript_lower = transcript.lower().strip()
        
        # If it's a single word greeting in the first few seconds, might be false
        words = transcript_lower.split()
        if len(words) == 1 and words[0] in greeting_words and time_start < 1.0:
            return True
        
        return False



    def _handle_first_message(self, text):
        """Track first message for duplicate detection."""
        if self._first_message:
            return
        
        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):
        """Check if this is a repeated first message to ignore.
        
        Only ignores EXACT duplicates within 1 second - Soniox is less prone
        to repetition than Deepgram, so we're very conservative here.
        """
        if not self._first_message:
            return False
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        # EXACT match only (not fuzzy match) to avoid false positives
        return text.strip() == self._first_message.strip()

    async def _detect_and_handle_voicemail(self, transcript: str) -> bool:
        """Detect and handle voicemail messages.

        Deepgram compatibility: Returns bool and pushes BOTH TranscriptionFrame and VoicemailFrame.

        Returns:
            True if voicemail was detected and handled, False otherwise
        """
        if not self.detect_voicemail:
            return False

        time_since_init = self._time_since_init()

        # Only detect voicemail in the first N seconds AND after first transcript receipt
        if time_since_init > VOICEMAIL_DETECTION_SECONDS and self._was_first_transcript_receipt:
            return False

        # Check if transcript matches voicemail patterns
        if not voicemail.is_text_voicemail(transcript):
            return False

        logger.info(f"🔊 Voicemail detected: '{transcript}'")

        # Push BOTH frames (Deepgram compatibility)
        await self.push_frame(
            TranscriptionFrame(transcript, "", time_now_iso8601(), self._language)
        )
        await self.push_frame(
            VoicemailFrame(text=transcript)
        )

        logger.debug("📤 Voicemail frames pushed")
        return True



    async def _async_handle_accum_transcription(self, current_time):
        """Handle accumulated transcriptions with timeout.

        Deepgram compatibility: Fast response mode is always active.
        """
        if not self._last_interim_time:
            self._last_interim_time = 0.0

        # Fast response mode is always active
        await self._fast_response_send_accum_transcriptions()

    async def _handle_false_interim(self, current_time):
        """Handle false interim detection.

        Deepgram compatibility: Re-enabled to match Deepgram's behavior.
        Detects when we get an interim but no follow-up, indicating false detection.
        """
        if not self._user_speaking:
            return
        if not self._last_interim_time:
            return
        if self._vad_active:
            return
        # Don't trigger false interim if we have accumulated final tokens waiting to be sent
        if self._final_tokens:
            return

        last_interim_delay = current_time - self._last_interim_time

        if last_interim_delay > FALSE_INTERIM_SECONDS:
            return

        logger.debug("🚨 False interim detected - triggering user silence")
        await self._handle_user_silence()

    async def _async_handler(self, task_name):
        """Async handler for timeout management and false interim detection."""
        try:
            while self._connection_active:
                await asyncio.sleep(0.1)

                current_time = time.time()

                # Check if we should send accumulated transcription due to timeout
                # Use _last_any_token_time to check if user is still speaking (non-finals arriving)
                # Only timeout if NO tokens (final or non-final) have arrived recently
                if self._final_tokens and self._last_any_token_time:
                    elapsed = current_time - self._last_any_token_time
                    # If no new tokens at all for 800ms, consider it an endpoint
                    if elapsed > 0.8:
                        logger.debug(f"🔚 Endpoint detected by timeout ({elapsed:.2f}s since last ANY token)")
                        await self._send_accumulated_tokens()

                await self._async_handle_accum_transcription(current_time)
                await self._handle_false_interim(current_time)

            logger.debug("🛑 Async handler exiting - connection no longer active")

        except asyncio.CancelledError:
            logger.debug("🛑 Async handler cancelled")
            # Ensure connection is marked inactive
            self._connection_active = False
            raise
        finally:
            logger.debug("🛑 Async handler task completed")

    async def _fast_response_send_accum_transcriptions(self):
        """Send accumulated transcriptions with intelligent timing logic.

        Deepgram compatibility: Fast response mode is always active for optimal performance.
        """
        if len(self._accum_transcription_frames) == 0:
            return

        current_time = time.time()

        # Bypass VAD check if we have a complete sentence (ends with punctuation)
        if self._vad_active:
            if len(self._accum_transcription_frames) > 0:
                last_text = self._accum_transcription_frames[-1].text.strip()
                has_ending_punctuation = last_text and last_text[-1] in '.?!'
                if not has_ending_punctuation:
                    # Do not send if VAD is active and it's been less than 10 seconds since first message
                    if self._first_message_time and (current_time - self._first_message_time) > 10.0:
                        return

        last_message_time = max(self._last_interim_time or 0, self._last_time_accum_transcription)

        is_short_sentence = len(self._accum_transcription_frames) <= 2
        is_sentence_end = not self._is_accum_transcription(self._accum_transcription_frames[-1].text)
        time_since_last_message = current_time - last_message_time

        if is_short_sentence:
            if is_sentence_end:
                logger.debug("🚀 Fast response: Sending accum transcriptions (short sentence, end of phrase)")
                await self._send_accum_transcriptions()

            if not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("🚀 Fast response: Sending accum transcriptions (short sentence, timeout)")
                await self._send_accum_transcriptions()
        else:
            if is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("🚀 Fast response: Sending accum transcriptions (long sentence, end of phrase)")
                await self._send_accum_transcriptions()

            if not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds * 2:
                logger.debug("🚀 Fast response: Sending accum transcriptions (long sentence, timeout)")
                await self._send_accum_transcriptions()

    def get_stt_stats(self) -> Dict:
        """Get comprehensive STT performance statistics."""
        if not self._stt_response_times:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "latest": 0.0,
                "all_times": []
            }
        
        return {
            "count": len(self._stt_response_times),
            "average": round(sum(self._stt_response_times) / len(self._stt_response_times), 3),
            "min": round(min(self._stt_response_times), 3),
            "max": round(max(self._stt_response_times), 3),
            "latest": round(self._stt_response_times[-1], 3) if self._stt_response_times else 0.0,
            "all_times": [round(t, 3) for t in self._stt_response_times]
        }

    def is_connected(self) -> bool:
        """Check if websocket is connected and active."""
        return self._connection_active and self._websocket is not None

    def get_stt_response_times(self) -> List[float]:
        """Get list of STT response times."""
        return self._stt_response_times
    
    def get_average_stt_response_time(self) -> float:
        """Get average STT response time."""
        if not self._stt_response_times:
            return 0.0
        return sum(self._stt_response_times) / len(self._stt_response_times)

    def clear_stt_response_times(self):
        """Clear STT response times."""
        self._stt_response_times = []

    def log_stt_performance(self):
        """Log comprehensive STT performance statistics."""
        stats = self.get_stt_stats()
        logger.info("=" * 50)
        logger.info("📊 Soniox STT Performance Summary")
        logger.info("=" * 50)
        logger.info(f"Total transcriptions: {stats['count']}")
        logger.info(f"Average response time: {stats['average']:.3f}s")
        logger.info(f"Min response time: {stats['min']:.3f}s")
        logger.info(f"Max response time: {stats['max']:.3f}s")
        logger.info(f"Latest response time: {stats['latest']:.3f}s")
        logger.info("=" * 50)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for bot speaking state and interruption handling."""
        # Log all frames for debugging
        # frame_name = type(frame).__name__
        # logger.debug(f"🎯 Processing frame: {frame_name} (direction: {direction.name})")
        
        await super().process_frame(frame, direction)
        
        # Handle bot speaking state for interruption detection
        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("🤖 Received BotStartedSpeakingFrame")
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("🤖 Received BotStoppedSpeakingFrame")
            await self._handle_bot_silence()
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True
            self._vad_inactive_time = None  # Reset when VAD becomes active
            logger.info(f"🎤 {self}: VAD ACTIVE (voice detected)")
        elif isinstance(frame, VADInactiveFrame):
            self._vad_active = False
            self._vad_inactive_time = time.time()
            logger.info(f"🎤 {self}: VAD INACTIVE (silence detected at {self._vad_inactive_time})")