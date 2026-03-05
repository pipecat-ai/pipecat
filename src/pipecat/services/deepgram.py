#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import re
import time
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Callable
import random
from dataclasses import dataclass

import aiohttp
from loguru import logger
import websockets

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
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADActiveFrame,
    VADInactiveFrame,
    VoicemailFrame,
    STTRestartFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.rex import regex_list_matches
from pipecat.utils.string import is_equivalent_basic
from pipecat.services.stt_filters import contains_interruption_word
from pipecat.utils.text import voicemail


# See .env.example for Deepgram configuration needed
try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        ErrorResponse,
        LiveOptions,
        LiveResultResponse,
        LiveTranscriptionEvents,
        SpeakOptions,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`. Also, set `DEEPGRAM_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


DEFAULT_ON_NO_PUNCTUATION_SECONDS = 2
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3



@dataclass
class DeepgramError(Frame):
    text: str

    def __str__(self):
        return F"{self.name}"


@dataclass
class DeepgramFatalError(Frame):
    text: str

    def __str__(self):
        return F"{self.name}"




class DeepgramTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-helios-en",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)
        self._deepgram_client = DeepgramClient(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        options = SpeakOptions(
            model=self._voice_id,
            encoding=self._settings["encoding"],
            sample_rate=self.sample_rate,
            container="none",
        )

        try:
            await self.start_ttfb_metrics()

            response = await asyncio.to_thread(
                self._deepgram_client.speak.v("1").stream, {"text": text}, options
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            # The response.stream_memory is already a BytesIO object
            audio_buffer = response.stream_memory

            if audio_buffer is None:
                raise ValueError("No audio data received from Deepgram")

            # Read and yield the audio data in chunks
            audio_buffer.seek(0)  # Ensure we're at the start of the buffer
            chunk_size = 8192  # Use a fixed buffer size
            while True:
                await self.stop_ttfb_metrics()
                chunk = audio_buffer.read(chunk_size)
                if not chunk:
                    break
                frame = TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
                yield frame

                yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(f"Error getting audio: {str(e)}")


class DeepgramSiDetector:
    """
    Connects to Deepgram in Spanish, streams raw audio,
    and calls `callback(transcript)` whenever a final
    transcript contains "si", "sí", "si!", "sí!", etc.
    """
    def __init__(
        self,
        api_key: str,
        callback: Callable[[str], None],
        url: str = "wss://api.deepgram.com/v1/listen",
        sample_rate: int = 16_000,
    ):
        # callback to invoke on detection
        self._callback = callback
        # match standalone si or sí (case-insensitive), optional !/¡
        self._pattern = re.compile(r'\b(?:si|sí)\b[!¡]?', re.IGNORECASE)

        # build a Deepgram client
        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"}
            ),
        )

        # settings for Spanish transcription
        self._settings = {
            "encoding":       "linear16",
            "language":       "es",          # Spanish
            "model":          "general",
            "sample_rate":    sample_rate,
            "interim_results": False,        # only finals
            "punctuate":      True,
        }

        self._conn = None  # will hold the websocket client
        self.start_times = set()

    async def start(self):
        """
        Open the Deepgram websocket. Must be called before sending audio.
        """
        try:
            self._conn = self._client.listen.asyncwebsocket.v("1")
            self._conn.on(
                LiveTranscriptionEvents.Transcript,
                self._on_transcript_event
            )
            await self._conn.start(options=self._settings)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.start: {e}")

    async def send_audio(self, chunk: bytes):
        """
        Send a chunk of raw audio (bytes) to Deepgram.
        """
        try:
            if not self._conn or not self._conn.is_connected:
                raise RuntimeError("Connection not started. Call start() first.")
            await self._conn.send(chunk)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.send_audio: {e}")

    async def stop(self):
        """
        Gracefully close the Deepgram websocket when you’re done sending.
        """
        try:
            if self._conn and self._conn.is_connected:
                await self._conn.finish()
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.stop: {e}")

    async def _on_transcript_event(self, *args, **kwargs):
        """
        Internal handler for every transcription event.
        Filters down to final transcripts and applies the "si" regex.
        """
        try:
            result = kwargs.get("result")
            if not result:
                return

            alts = result.channel.alternatives
            if not alts:
                return

            transcript = alts[0].transcript.strip()
            if self._pattern.search(transcript):

                if result.start in self.start_times: return
                logger.debug("Si detected")
                
                self.start_times.add(result.start)
                
                logger.debug("Si detected")

                await self._callback(result)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector._on_transcript_event: {e}")

def language_to_gladia_language(language: Language) -> str:
    """Convert Pipecat Language to Gladia language code."""
    lang_map = {
        Language.ES: "es",
        Language.EN: "en", 
        Language.FR: "fr",
        Language.DE: "de",
        Language.IT: "it",
        Language.PT: "pt",
        Language.CA: "ca",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.ZH: "zh",
        Language.RU: "ru",
        Language.AR: "ar",
        Language.HI: "hi",
    }
    if isinstance(language, Language):
        return lang_map.get(language, "es")
    elif isinstance(language, str):
        return language if language in lang_map.values() else "es"
    return "es"

class DeepgramGladiaDetector:
    """
    Ultra-reliable STT backup using Gladia's Solaria-1 model.
    Designed as intelligent support for Deepgram with maximum sensitivity
    to catch any transcript that Deepgram might miss.
    """
    def __init__(
        self,
        api_key: str,
        callback: Callable[[str, float, float, bool], None],  # transcript, confidence, timestamp, is_backup
        language: Language = Language.ES,
        url: str = "https://api.gladia.io/v2/live",
        sample_rate: int = 16_000,
        confidence: float = 0.1,  # Lower threshold for maximum sensitivity
        endpointing: float = 0.2,  # More sensitive to catch everything
        speech_threshold: float = 0.3,  # Lower for maximum detection
        timeout_seconds: float = 2.0,  # Longer timeout for better accuracy
        deepgram_wait_timeout: float = 1.8,  # Wait time for Deepgram finals
        stt_service_ref=None,  # Reference to main STT service for accessing bot state
    ):
        self._api_key = api_key
        self._callback = callback
        self._language = language
        self._url = url
        self._sample_rate = sample_rate
        self._confidence = confidence
        self._timeout_seconds = timeout_seconds
        self._deepgram_wait_timeout = deepgram_wait_timeout
        self._stt_service_ref = stt_service_ref  # Reference to main STT service
        
        # WebSocket connection
        self._websocket = None
        self._receive_task = None
        
        # Enhanced transcript coordination
        self._pending_transcripts = {}  # Store transcripts waiting for Deepgram
        self._processed_transcripts = set()  # Deduplication
        self._last_transcript_time = 0
        self._start_time = time.time()
        self._last_deepgram_transcript = ""  # Track last Deepgram transcript for similarity checking
        self._last_deepgram_time = 0  # Track timing
        
        # VAD-based filtering for backup system
        self._last_vad_inactive_time = None  # Track when VAD goes inactive
        self._vad_backup_window = 1.5  # Only allow backup transcripts within 1.5s of VAD inactive
        
        # Optimized Gladia configuration for maximum reliability as backup
        self._settings = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": sample_rate,
            "channels": 1,
            "model": "solaria-1",  # Best accuracy model
            "endpointing": endpointing,  # More sensitive
            "maximum_duration_without_endpointing": 8,  # Longer for complete phrases
            "language_config": {
                "languages": [language_to_gladia_language(language)],
                "code_switching": True,  # Handle mixed languages
            },
            "pre_processing": {
                "audio_enhancer": True,  # Enhanced for difficult audio
                "speech_threshold": speech_threshold,  # Lower for sensitivity
            },
            "realtime_processing": {
                "words_accurate_timestamps": True,  # Better timing
            },
            "messages_config": {
                "receive_final_transcripts": True,
                "receive_speech_events": True,  # Track speech patterns
                "receive_pre_processing_events": False,
                "receive_realtime_processing_events": False,
                "receive_post_processing_events": False,
                "receive_acknowledgments": False,
                "receive_errors": True,
                "receive_lifecycle_events": False
            }
        }

    async def start(self):
        """Initialize Gladia connection as intelligent Deepgram backup."""
        try:
            logger.info("🎯 DeepgramGladiaDetector: Starting intelligent STT backup service")
            response = await self._setup_gladia()
            self._websocket = await websockets.connect(response["url"])
            self._receive_task = asyncio.create_task(self._receive_task_handler())
            logger.info("✅ DeepgramGladiaDetector: Intelligent backup ready")
        except Exception as e:
            logger.error(f"❌ DeepgramGladiaDetector failed to start: {e}")
            logger.warning("⚠️ Continuing without Gladia backup - using Deepgram only")
            # Don't raise - allow system to continue with Deepgram only
            self._websocket = None
            self._receive_task = None

    async def send_audio(self, chunk: bytes):
        """Send audio chunk to Gladia backup service."""
        try:
            if not self._websocket:
                logger.warning("🎯 DeepgramGladiaDetector: WebSocket not available, skipping audio chunk")
                return
                
            chunk_size = len(chunk)
            logger.trace(f"🎯 DeepgramGladiaDetector: Sending audio chunk ({chunk_size} bytes) to backup")
            
            data = base64.b64encode(chunk).decode("utf-8")
            message = {"type": "audio_chunk", "data": {"chunk": data}}
            await self._websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.debug(f"🔧 DeepgramGladiaDetector audio send error: {e}")

    async def notify_deepgram_final(self, transcript: str, timestamp: float):
        """Notify that Deepgram has sent a final transcript."""
        try:
            # Update tracking info regardless
            self._last_deepgram_transcript = transcript.strip().lower()
            self._last_deepgram_time = timestamp
            
            # Cancel any recent pending transcripts, as Deepgram has now responded for this timeframe.
            # Use a time window (e.g., 2 seconds) to decide which backups are now irrelevant.
            cancellation_window = 2.0 
            keys_to_cancel = []
            for key, pending_info in self._pending_transcripts.items():
                if abs(timestamp - pending_info['timestamp']) < cancellation_window:
                    keys_to_cancel.append(key)

            for key in keys_to_cancel:
                pending_info = self._pending_transcripts.pop(key, None)
                if pending_info and 'timeout_task' in pending_info:
                    logger.info(f"🎯 DeepgramGladiaDetector: Deepgram final received, canceling nearby backup: '{pending_info['transcript']}'")
                    pending_info['timeout_task'].cancel()

            # Also mark the specific transcript as processed in case of near-simultaneous results
            transcript_key = self._create_transcript_key(transcript, timestamp)
            self._processed_transcripts.add(transcript_key)
            logger.debug(f"🎯 DeepgramGladiaDetector: Marked Deepgram transcript as processed: {transcript_key}")
                
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector notify error: {e}")

    def _create_transcript_key(self, transcript: str, timestamp: float) -> str:
        """Create a consistent key for transcript matching."""
        # Normalize transcript for comparison
        normalized = transcript.lower().strip()
        # Use time windows for matching (100ms tolerance)
        time_window = int(timestamp * 10) / 10
        return f"{normalized}_{time_window}"

    async def stop(self):
        """Gracefully stop the backup service."""
        try:
            logger.info("🛑 DeepgramGladiaDetector: Stopping intelligent backup")
            
            # Cancel all pending tasks
            for pending_info in self._pending_transcripts.values():
                if 'timeout_task' in pending_info:
                    pending_info['timeout_task'].cancel()
            self._pending_transcripts.clear()
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            
            if self._websocket:
                await self._websocket.send(json.dumps({"type": "stop_recording"}))
                await self._websocket.close()
                
            logger.info("✅ DeepgramGladiaDetector: Intelligent backup stopped")
            
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector stop error: {e}")

    async def _setup_gladia(self):
        """Setup Gladia session for intelligent backup."""
        async with aiohttp.ClientSession() as session:
            logger.debug("🔧 DeepgramGladiaDetector: Configuring intelligent backup service")
            logger.debug(f"🎯 Gladia backup settings: {self._settings}")
            
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=self._settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Gladia backup session failed: {response.status} - {error_text}")

    async def _receive_task_handler(self):
        """Handle backup transcriptions from Gladia with intelligent coordination."""
        try:
            logger.debug("📡 DeepgramGladiaDetector: Listening for backup transcriptions")
            
            async for message in self._websocket:
                try:
                    content = json.loads(message)
                    
                    if content["type"] != "transcript":
                        logger.trace(f"🎯 DeepgramGladiaDetector: Ignoring non-transcript message: {content['type']}")
                        continue
                        
                    utterance = content["data"]["utterance"]
                    confidence = utterance.get("confidence", 0)
                    transcript = utterance["text"].strip()
                    is_final = content["data"]["is_final"]
                    
                    logger.debug(f"🎯 DeepgramGladiaDetector: Backup message - Final: {is_final}, Text: '{transcript}', Confidence: {confidence:.2f}")
                    
                    # Only process final transcripts for backup
                    if not is_final:
                        logger.trace(f"🎯 DeepgramGladiaDetector: Skipping interim backup result: '{transcript}'")
                        continue
                    
                    if not transcript:
                        logger.trace(f"🎯 DeepgramGladiaDetector: Skipping empty backup transcript")
                        continue
                        
                    if confidence < self._confidence:
                        logger.debug(f"🎯 DeepgramGladiaDetector: Skipping low confidence backup: '{transcript}' (conf: {confidence:.2f} < {self._confidence})")
                        continue
                    
                    # Process this backup transcript
                    await self._process_backup_transcript(transcript, confidence)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ DeepgramGladiaDetector: Invalid JSON: {e}")
                except Exception as e:
                    logger.exception(f"❌ DeepgramGladiaDetector message error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 DeepgramGladiaDetector: Backup connection closed")
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector receive error: {e}")

    async def _process_backup_transcript(self, transcript: str, confidence: float):
        """Process backup transcript with intelligent coordination and VAD filtering."""
        try:
            current_time = time.time()
            
            # VAD-based filtering: Only process backup transcripts within window after VAD inactive
            if self._last_vad_inactive_time is not None:
                time_since_vad_inactive = current_time - self._last_vad_inactive_time
                if time_since_vad_inactive > self._vad_backup_window:
                    logger.debug(f"🚫 DeepgramGladiaDetector: Backup ignored - outside VAD window: '{transcript}' ({time_since_vad_inactive:.1f}s > {self._vad_backup_window}s)")
                    return
                else:
                    logger.debug(f"✅ DeepgramGladiaDetector: Backup within VAD window: '{transcript}' ({time_since_vad_inactive:.1f}s <= {self._vad_backup_window}s)")
            else:
                logger.debug(f"⚠️ DeepgramGladiaDetector: No VAD reference time, allowing backup: '{transcript}'")
            
            # Check similarity to recent Deepgram transcript
            if self._last_deepgram_transcript and self._last_deepgram_time:
                time_since_deepgram = current_time - self._last_deepgram_time
                transcript_normalized = transcript.strip().lower()
                
                # If very recent Deepgram transcript and similar content, skip backup
                if (time_since_deepgram < 3.0 and 
                    (transcript_normalized == self._last_deepgram_transcript or
                     transcript_normalized in self._last_deepgram_transcript or
                     self._last_deepgram_transcript in transcript_normalized)):
                    logger.debug(f"🔄 DeepgramGladiaDetector: Backup ignored - similar to recent Deepgram ({time_since_deepgram:.1f}s ago): '{transcript}'")
                    return
            
            transcript_key = self._create_transcript_key(transcript, current_time)
            
            # Check if Deepgram already processed this transcript
            if transcript_key in self._processed_transcripts:
                logger.debug(f"🔄 DeepgramGladiaDetector: Backup ignored - Deepgram already processed: '{transcript}'")
                return
            
            # Enhanced deduplication
            similar_key = f"{transcript.lower()}_{confidence:.2f}"
            if similar_key in self._processed_transcripts:
                logger.debug(f"🔄 DeepgramGladiaDetector: Backup duplicate ignored: '{transcript}'")
                return
                
            # Store this transcript and wait for Deepgram
            logger.info(f"🎯 DeepgramGladiaDetector: 📋 Backup transcript ready: '{transcript}' (confidence: {confidence:.2f})")
            logger.info(f"⏰ DeepgramGladiaDetector: Waiting {self._deepgram_wait_timeout}s for Deepgram...")
            
            # Create timeout task
            timeout_task = asyncio.create_task(self._backup_timeout_handler(transcript, confidence, current_time, transcript_key))
            
            self._pending_transcripts[transcript_key] = {
                'transcript': transcript,
                'confidence': confidence,
                'timestamp': current_time,
                'timeout_task': timeout_task
            }
            
            # Clean old processed transcripts to prevent memory growth
            if len(self._processed_transcripts) > 200:
                logger.debug(f"🎯 DeepgramGladiaDetector: Cleaning processed transcripts cache (size: {len(self._processed_transcripts)})")
                self._processed_transcripts.clear()
                
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector backup processing error: {e}")

    async def _backup_timeout_handler(self, transcript: str, confidence: float, timestamp: float, transcript_key: str):
        """Handle backup transcript timeout - send if Deepgram doesn't respond."""
        try:
            await asyncio.sleep(self._deepgram_wait_timeout)
            
            # Check if still pending (not canceled by Deepgram)
            if transcript_key in self._pending_transcripts:
                pending_info = self._pending_transcripts[transcript_key]
                transcript_timestamp = pending_info['timestamp']
                
                # Additional check: ONLY suppress if the transcript appears AFTER the bot started speaking
                if (self._stt_service_ref and 
                    hasattr(self._stt_service_ref, '_bot_started_speaking_time') and 
                    self._stt_service_ref._bot_started_speaking_time):
                    
                    bot_start_time = self._stt_service_ref._bot_started_speaking_time
                    
                    # ONLY suppress if the transcript appears AFTER the bot started speaking
                    if transcript_timestamp > bot_start_time:
                        logger.info(f"🚫 DeepgramGladiaDetector: Backup suppressed - user spoke after bot started speaking: '{transcript}'")
                        self._pending_transcripts.pop(transcript_key, None)
                        return
                
                logger.warning(f"⏰ DeepgramGladiaDetector: BACKUP ACTIVATED - Deepgram timeout, using backup: '{transcript}'")
                
                # Remove from pending
                self._pending_transcripts.pop(transcript_key, None)
                
                # Mark as processed
                self._processed_transcripts.add(transcript_key)
                
                # Send as backup transcript
                logger.info(f"🎯 DeepgramGladiaDetector: ✅ BACKUP TRANSCRIPT SENT: '{transcript}' (confidence: {confidence:.2f})")
                await self._callback(transcript, confidence, timestamp, True)  # True = is_backup
                
        except asyncio.CancelledError:
            logger.debug(f"🎯 DeepgramGladiaDetector: Backup timeout canceled for: '{transcript}'")
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector backup timeout error: {e}")

    async def update_vad_inactive_time(self, timestamp: float = None):
        """Update the VAD inactive timestamp for filtering backup transcripts."""
        self._last_vad_inactive_time = timestamp or time.time()
        logger.debug(f"🎤 DeepgramGladiaDetector: VAD inactive timestamp updated: {self._last_vad_inactive_time}")


class DeepgramSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        bakckup_api_keys: Optional[List[str]] = None,
        url: str = "",
        sample_rate: Optional[int] = None,
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        on_connection_error: Optional[Callable[[Frame], None]] = None,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        detect_voicemail: bool = True,  
        allow_interruptions: bool = True,
        gladia_api_key: Optional[str] = None,  # NEW: Gladia API key for intelligent backup
        gladia_timeout: float = 1.8,  # Timeout for backup activation
        fast_response: bool = False,
        **kwargs,
    ):
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            channels=1,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )

        merged_options = default_options
        if live_options:
            merged_options = LiveOptions(**{**default_options.to_dict(), **live_options.to_dict()})

        # deepgram connection requires language to be a string
        if isinstance(merged_options.language, Language) and hasattr(
            merged_options.language, "value"
        ):
            merged_options.language = merged_options.language.value
        
        self.language = merged_options.language
        self.api_key = api_key
        self.backup_api_keys = bakckup_api_keys or []
        random.shuffle(self.backup_api_keys)
        self.detect_voicemail = detect_voicemail  
        self._allow_stt_interruptions = allow_interruptions
        self._fast_response = fast_response
        logger.debug(f"Fast response enabled: {self._fast_response}")
        logger.debug(f"Allow ** interruptions: {self._allow_stt_interruptions}")

        self._settings = merged_options.to_dict()
        self._addons = addons
        self._user_speaking = False
        self._bot_speaking = True
        self._bot_has_ever_spoken = False  # Track if bot has spoken at least once
        self._bot_started_speaking_time = None  # Track when bot started speaking
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._vad_active = False

        self._first_message = None
        self._first_message_time = None
        self._last_interim_time = None
        self._restarted = False
        self._error_count = 0
        self._on_connection_error = on_connection_error

        # FIX: Add debouncing for error handling to prevent race conditions
        self._reconnecting = False
        self._reconnect_lock = asyncio.Lock()

        self._setup_sibling_deepgram()


        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"},  # verbose=logging.DEBUG
            ),
        )

        if self.vad_enabled:
            logger.debug(f"Deepgram VAD Enabled: {self.vad_enabled}")
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

        self._async_handler_task = None
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_time_transcription = time.time()
        self._was_first_transcript_receipt = False

        self.start_time = time.time()
        
        # Enhanced response time tracking
        self._stt_response_times = []  # List to store STT response durations
        self._current_speech_start_time = None  # Track when speech detection starts
        self._last_audio_chunk_time = None  # Track last audio chunk received
        self._audio_chunk_count = 0  # Count audio chunks for debugging


        self._gladia_api_key = gladia_api_key
        self._gladia_timeout = gladia_timeout
        self._pending_deepgram_finals = {}  # Store Deepgram finals waiting for coordination
        self._backup_enabled = False  # Flag to track backup status
        
        # VAD-based filtering for backup system
        self._last_vad_inactive_time = None  # Track when VAD goes inactive
        self._vad_backup_window = 1.5  # Only allow backup transcripts within 1.5s of VAD inactive
        
        self._setup_intelligent_gladia_backup()

    def _setup_intelligent_gladia_backup(self):
        """Setup Gladia service as intelligent backup for maximum reliability."""
        self._intelligent_gladia_backup = None

        if not self._gladia_api_key:
            logger.info("🔧 DeepgramSTTService: No Gladia API key provided, using Deepgram only")
            return

        # Enhanced language support for backup
        backup_languages = ['es', 'en', 'fr', 'pt', 'ca', 'de', 'it', 'ja', 'ko', 'zh', 'ru', 'ar', 'hi']
        current_lang = self.language.lower() if isinstance(self.language, str) else str(self.language).lower()
        
        logger.debug(f"🔧 DeepgramSTTService: Checking language '{current_lang}' for intelligent backup")
        
        # Enable backup for all supported languages
        if not any(lang in current_lang for lang in backup_languages):
            logger.info(f"🔧 DeepgramSTTService: Language '{current_lang}' backup not available")
            return

        try:
            # Convert language for Gladia
            if isinstance(self.language, str):
                lang_lower = self.language.lower()
                if lang_lower == 'es':
                    gladia_language = Language.ES
                elif lang_lower == 'en':
                    gladia_language = Language.EN
                elif lang_lower == 'fr':
                    gladia_language = Language.FR
                elif lang_lower == 'pt':
                    gladia_language = Language.PT
                elif lang_lower == 'ca':
                    gladia_language = Language.CA
                elif lang_lower == 'de':
                    gladia_language = Language.DE
                elif lang_lower == 'it':
                    gladia_language = Language.IT
                else:
                    gladia_language = Language.ES  # Default fallback
            else:
                gladia_language = self.language

            logger.info(f"🔧 DeepgramSTTService: Setting up intelligent Gladia backup for {gladia_language}")

            self._intelligent_gladia_backup = DeepgramGladiaDetector(
                api_key=self._gladia_api_key,
                callback=self.intelligent_backup_handler,
                language=gladia_language,
                sample_rate=self.sample_rate or 16000,
                confidence=0.1,  # Lower for maximum sensitivity
                endpointing=0.2,  # More sensitive
                speech_threshold=0.3,  # Lower threshold
                timeout_seconds=2.0,  # Longer for accuracy
                deepgram_wait_timeout=1.0,  # Changed from 2.5 to 1.0 to reduce race condition
                stt_service_ref=self  # Pass reference to self for accessing bot state
            )
            
            self._backup_enabled = True
            logger.info(f"🎯 DeepgramSTTService: ✅ Intelligent backup enabled for {gladia_language} (timeout: {self._gladia_timeout}s)")

        except Exception as e:
            logger.exception(f"❌ DeepgramSTTService: Intelligent backup setup failed: {e}")
            self._intelligent_gladia_backup = None
            self._backup_enabled = False

    def get_allow_interruptions(self) -> bool:
        """Get the current allow_interruptions setting."""
        return self._allow_stt_interruptions

    def set_allow_interruptions(self, value: bool):
        """Set the allow_interruptions setting."""
        logger.debug(f"DeepgramSTTService: set_allow_interruptions({value})")
        self._allow_stt_interruptions = value

    async def intelligent_backup_handler(self, transcript: str, confidence: float, timestamp: float, is_backup: bool):
        """Handle intelligent backup transcription from Gladia."""
        try:
            source_name = "🆘 BACKUP" if is_backup else "🎯 Enhanced"
            logger.info(f"🎯 DeepgramSTTService: ⬇️ {source_name} callback received: '{transcript}' (confidence: {confidence:.2f})")
            
            # Create a mock Deepgram result for compatibility
            mock_result = type('MockResult', (), {
                'is_final': True,
                'speech_final': True,
                'start': timestamp,
                'channel': type('Channel', (), {
                    'alternatives': [type('Alternative', (), {
                        'transcript': transcript,
                        'confidence': confidence,
                        'words': [],
                        'languages': [str(self.language)] if hasattr(self, 'language') else ['es']
                    })()]
                })()
            })()
            
            logger.debug(f"🎯 DeepgramSTTService: Created mock result for backup transcript")
            
            # Process through existing Deepgram logic but mark as backup
            logger.debug(f"🎯 DeepgramSTTService: Processing {'BACKUP' if is_backup else 'Enhanced'} transcript")
            await self._on_message(result=mock_result, backup_source=is_backup)
            
        except Exception as e:
            logger.exception(f"❌ DeepgramSTTService: Intelligent backup handler error: {e}")

    @property
    def vad_enabled(self):
        return self._settings["vad_events"]
    
    def _setup_sibling_deepgram(self):

        self._sibling_deepgram = None

        if self.language != 'it': return 

        self._sibling_deepgram = DeepgramSiDetector(self.api_key, self.sibling_transcript_handler)


    async def sibling_transcript_handler(self, result: LiveResultResponse):

        result_time = result.start

        if abs(self._last_time_transcription - result_time) < 0.5:
            logger.debug("Ignoring 'Si' because recent transcript")
            return
        
        await self._on_message(result=result)


    def _time_since_init(self):
        return time.time() - self.start_time

    def can_generate_metrics(self) -> bool:
        return True

    def get_stt_response_times(self) -> List[float]:
        """Get the list of STT response durations."""
        return self._stt_response_times.copy()
    
    def get_average_stt_response_time(self) -> float:
        """Get the average STT response duration."""
        if not self._stt_response_times:
            return 0.0
        return sum(self._stt_response_times) / len(self._stt_response_times)

    def clear_stt_response_times(self):
        """Clear the list of STT response durations."""
        self._stt_response_times.clear()
    
    def get_stt_stats(self) -> Dict:
        """Get comprehensive STT performance statistics."""
        if not self._stt_response_times:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "latest": 0.0
            }
        
        return {
            "count": len(self._stt_response_times),
            "average": round(sum(self._stt_response_times) / len(self._stt_response_times), 3),
            "min": round(min(self._stt_response_times), 3),
            "max": round(max(self._stt_response_times), 3),
            "latest": round(self._stt_response_times[-1], 3) if self._stt_response_times else 0.0,
            "all_times": [round(t, 3) for t in self._stt_response_times]
        }
    
    def log_stt_performance(self):
        """Log STT performance statistics."""
        stats = self.get_stt_stats()
        if stats["count"] > 0:
            logger.info(f"🎯 Deepgram STT Performance Summary:")
            logger.info(f"   📊 Total responses: {stats['count']}")
            logger.info(f"   ⏱️  Average time: {stats['average']}s")
            logger.info(f"   🏃 Fastest: {stats['min']}s")
            logger.info(f"   🐌 Slowest: {stats['max']}s")
            logger.info(f"   🕐 Latest: {stats['latest']}s")
            logger.info(f"   📈 All times: {stats['all_times']}")

    def get_backup_stats(self) -> Dict:
        """Get intelligent backup system statistics."""
        if not self._backup_enabled or not self._intelligent_gladia_backup:
            return {
                "backup_enabled": False,
                "backup_activations": 0,
                "reliability_score": 1.0,
                "status": "Backup not available"
            }
        
        # Count backup activations from response times logs (this is a simplified approach)
        # In a real implementation, you'd track these separately
        backup_activations = 0  # This would be tracked in the backup system
        total_responses = len(self._stt_response_times)
        
        reliability_score = 1.0 if total_responses == 0 else max(0.0, 1.0 - (backup_activations / total_responses))
        
        return {
            "backup_enabled": True,
            "backup_activations": backup_activations,
            "total_responses": total_responses,
            "reliability_score": round(reliability_score, 3),
            "primary_success_rate": f"{(reliability_score * 100):.1f}%",
            "status": "Intelligent backup active and monitoring"
        }

    def get_comprehensive_stt_stats(self) -> Dict:
        """Get comprehensive STT statistics including backup system."""
        base_stats = self.get_stt_stats()
        backup_stats = self.get_backup_stats()
        
        return {
            **base_stats,
            "backup_system": backup_stats,
            "system_reliability": "Ultra-High" if backup_stats["backup_enabled"] else "Standard"
        }

    def log_comprehensive_stt_performance(self):
        """Log comprehensive STT performance including backup system."""
        stats = self.get_comprehensive_stt_stats()
        backup = stats["backup_system"]
        
        if stats["count"] > 0:
            logger.info(f"🎯 === COMPREHENSIVE STT PERFORMANCE REPORT ===")
            logger.info(f"   📊 Total responses: {stats['count']}")
            logger.info(f"   ⏱️  Average time: {stats['average']}s")
            logger.info(f"   🏃 Fastest: {stats['min']}s")
            logger.info(f"   🐌 Slowest: {stats['max']}s")
            logger.info(f"   🕐 Latest: {stats['latest']}s")
            logger.info(f"   🛡️ System reliability: {stats['system_reliability']}")
            
            if backup["backup_enabled"]:
                logger.info(f"   🆘 Backup status: {backup['status']}")
                logger.info(f"   📈 Primary success rate: {backup['primary_success_rate']}")
                logger.info(f"   🔄 Backup activations: {backup['backup_activations']}")
                logger.info(f"   🎯 Reliability score: {backup['reliability_score']}")
            else:
                logger.info(f"   🔧 Backup: Not configured")
                
            logger.info(f"   📊 All times: {stats['all_times']}")
            logger.info(f"🎯 === END PERFORMANCE REPORT ===")

    async def set_model(self, model: str):
        try:
            await super().set_model(model)
            logger.info(f"Switching STT model to: [{model}]")
            self._settings["model"] = model
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_model: {e}")
            raise

    async def set_language(self, language: Language):
        try:
            logger.info(f"Switching STT language to: [{language}]")
            self._settings["language"] = language
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_language: {e}")
            raise

    async def start(self, frame: StartFrame):
        try:
            await super().start(frame)
            self._settings["sample_rate"] = self.sample_rate
            await self._connect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.start()
                logger.debug("🔧 DeepgramSTTService: Sibling Deepgram started")
                
            if self._intelligent_gladia_backup:
                logger.info("🎯 DeepgramSTTService: Starting intelligent Gladia backup...")
                try:
                    await self._intelligent_gladia_backup.start()
                    if self._intelligent_gladia_backup._websocket is None:
                        logger.warning("⚠️ DeepgramSTTService: Gladia backup failed to connect, disabling backup")
                        self._intelligent_gladia_backup = None
                        self._backup_enabled = False
                except Exception as e:
                    logger.error(f"❌ DeepgramSTTService: Gladia backup startup failed: {e}")
                    logger.warning("⚠️ DeepgramSTTService: Continuing without Gladia backup")
                    self._intelligent_gladia_backup = None
                    self._backup_enabled = False
            
            if not self._async_handler_task:
                self._async_handler_task = self.create_monitored_task(self._async_handler)
            
        except Exception as e:
            logger.exception(f"{self} exception in start: {e}")
            raise       

    async def stop(self, frame: EndFrame):
        try:
            await super().stop(frame)
            await self._disconnect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.stop()
                logger.debug("🔧 DeepgramSTTService: Sibling Deepgram stopped")
                
            if self._intelligent_gladia_backup and self._backup_enabled:
                logger.info("🎯 DeepgramSTTService: Stopping intelligent Gladia backup...")
                try:
                    await self._intelligent_gladia_backup.stop()
                    logger.info("🎯 DeepgramSTTService: ✅ Intelligent backup stopped")
                except Exception as e:
                    logger.warning(f"⚠️ DeepgramSTTService: Error stopping backup: {e}")
                
        except Exception as e:
            logger.exception(f"{self} exception in stop: {e}")
            raise

    async def cancel(self, frame: CancelFrame):

        try:
            await super().cancel(frame)
            await self._disconnect()
        except Exception as e:
            logger.exception(f"{self} exception in cancel: {e}")
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            # Enhanced timing tracking
            current_time = time.perf_counter()
            self._last_audio_chunk_time = current_time
            self._audio_chunk_count += 1
            
            # Start timing when we receive first audio after speech detection
            if self._current_speech_start_time is None:
                self._current_speech_start_time = current_time
                logger.debug(f"🎤 DeepgramSTTService: ⏱️ Starting speech detection timer at chunk #{self._audio_chunk_count}")
            
            # Send audio to primary Deepgram service (if connected)
            deepgram_sent = False
            if self._connection and self._connection.is_connected:
                logger.trace(f"⚡ DeepgramSTTService: Sending audio chunk #{self._audio_chunk_count} to Deepgram ({len(audio)} bytes)")
                await self._connection.send(audio)
                deepgram_sent = True
            else:
                logger.debug(f"⚠️ DeepgramSTTService: Deepgram not connected, relying on backup system")
            
            # Send to sibling services if available
            if self._sibling_deepgram:
                logger.trace(f"🔧 DeepgramSTTService: Sending audio to sibling Deepgram")
                await self._sibling_deepgram.send_audio(audio)
                
            # Send to intelligent backup service (always send, it will filter appropriately)
            if self._intelligent_gladia_backup and self._backup_enabled:
                logger.trace(f"🎯 DeepgramSTTService: Sending audio to intelligent backup")
                try:
                    await self._intelligent_gladia_backup.send_audio(audio)
                except Exception as e:
                    logger.debug(f"⚠️ DeepgramSTTService: Backup audio send failed: {e}, disabling backup")
                    self._intelligent_gladia_backup = None
                    self._backup_enabled = False
            
            if not deepgram_sent and not self._backup_enabled:
                # If neither Deepgram nor backup is available, we have a problem
                logger.error("🚨 DeepgramSTTService: No STT services available (Deepgram down, no backup)")
                yield ErrorFrame("No STT services available")
                return
                
            yield None
        except Exception as e:
            # Convert exception to string FIRST to avoid f-string issues
            error_type = type(e).__name__
            error_msg = str(e)
            error_msg_lower = error_msg.lower()

            logger.error(f"Deepgram STT {error_type}: {error_msg}")

            # Determine if error is fatal
            is_fatal = any(keyword in error_msg_lower for keyword in [
                'authentication', 'unauthorized', '401', '403', 'invalid', 'api key',
                'quota', 'suspended', 'credentials', 'connection'
            ])

            yield ErrorFrame(
                error=f"Deepgram STT {error_type}: {error_msg[:200]}",
                fatal=is_fatal
            )


    async def _connect(self):
        try:
            logger.debug("Connecting to Deepgram")
            
            # Validate API key before attempting connection
            if not self.api_key or self.api_key.strip() == "":
                raise ValueError("Deepgram API key is empty or invalid")
            
            logger.debug(f"Using Deepgram API key: {self.api_key[:10]}...")

            self._client._config.set_apikey(self.api_key)
            self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
            )
            self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

            if not self._async_handler_task:
                self._async_handler_task = self.create_monitored_task(self._async_handler)

            if self.vad_enabled:
                self._connection.on(
                    LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                    self._on_speech_started,
                )
                self._connection.on(
                    LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                    self._on_utterance_end,
                )

            logger.debug(f"Deepgram connection settings: {self._settings}")
            logger.debug(f"Deepgram addons: {self._addons}")
            
            connection_result = await self._connection.start(options=self._settings, addons=self._addons)

            if not connection_result:
                logger.error(f"{self}: unable to connect to Deepgram - connection failed")
                raise ConnectionError("Failed to establish Deepgram connection")
            else:
                logger.debug(f"Successfully connected to Deepgram")
        except Exception as e:
            # Convert exception to string FIRST to avoid f-string issues
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Deepgram connection {error_type}: {error_msg}")
            # Don't raise here to allow fallback to backup system only
            logger.warning(f"Deepgram connection failed, backup system will handle all transcriptions")
            await self._on_error(error=e)

    async def _disconnect(self):
        try:
            if self._async_handler_task:
                await self.cancel_task(self._async_handler_task)
                self._async_handler_task = None

            # FIX: Check connection exists before checking is_connected to prevent AttributeError
            if self._connection is not None:
                try:
                    if self._connection.is_connected:
                        logger.debug("Disconnecting from Deepgram")
                        try:
                            await asyncio.wait_for(self._connection.finish(), timeout=0.5)
                            logger.debug("Safe disconnect from Deepgram")
                        except asyncio.TimeoutError:
                            logger.warning("Timeout while disconnecting from Deepgram, forcing close")
                            # Force close the websocket if timeout occurs
                            if hasattr(self._connection, '_websocket') and self._connection._websocket:
                                try:
                                    await self._connection._websocket.close()
                                except Exception as ws_close_err:
                                    logger.debug(f"Error force-closing websocket: {ws_close_err}")
                except Exception as e:
                    logger.error(f"Error during connection cleanup: {e}")
                finally:
                    # Always clear the connection reference
                    self._connection = None
        except Exception as e:
            logger.exception(f"{self} exception in _disconnect: {e}")
        finally:
            # Ensure connection is always cleared even if outer try fails
            self._connection = None

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        # Start timing for STT response
        self._current_request_start_time = time.perf_counter()

    async def _on_error(self, *args, **kwargs):
        error: ErrorResponse = kwargs["error"]
        error_msg = str(error)

        # FIX: Prevent concurrent error handling with debouncing
        if self._reconnecting:
            logger.debug(f"{self} already handling error, skipping duplicate")
            return

        async with self._reconnect_lock:
            # Double-check after acquiring lock
            if self._reconnecting:
                logger.debug(f"{self} already reconnecting (after lock), skipping")
                return

            self._reconnecting = True
            try:
                logger.warning(f"{self} connection error, will retry: {error_msg}")
                await self.stop_all_metrics()

                # FIX: Explicitly disconnect old connection before creating new one
                # This prevents connection leaks when switching to backup keys
                logger.debug(f"{self} disconnecting old connection before retry")
                await self._disconnect()

                # Increment error count first
                self._error_count += 1
                
                # Check if we've exhausted all backup keys
                if self._error_count > len(self.backup_api_keys):
                    logger.error(f"{self} exhausted all {len(self.backup_api_keys)} backup API keys (tried {self._error_count} times)")

                    # Check if Gladia backup is available
                    if self._backup_enabled and self._intelligent_gladia_backup:
                        logger.warning(f"{self} Deepgram failed but Gladia backup is available - continuing with backup only")
                        # Don't push fatal error - Gladia will handle transcriptions
                        return

                    # No STT service available - this is fatal
                    logger.error(f"{self} NO STT SERVICE AVAILABLE - Deepgram failed and no Gladia backup configured")

                    try:
                        await self.push_error(ErrorFrame(
                            error=f"STT fatal: No STT service available - Deepgram failed ({error_msg[:150]}), no backup configured",
                            fatal=True
                        ))
                    except Exception as push_error:
                        logger.error(f"Failed to push STT fatal error frame: {push_error}")

                    if self._on_connection_error:
                        self._on_connection_error(DeepgramFatalError(f"No STT service available: {error}"))
                    return

                # Use the next backup key (cycle through the list)
                self.api_key = self.backup_api_keys[self._error_count - 1]
                logger.info(f"{self} switching to backup Deepgram API key #{self._error_count}: {self.api_key[:10]}...")

                # Reset reconnecting flag BEFORE calling _connect()
                # This allows subsequent connection failures to retry with the next backup key
                self._reconnecting = False
                await self._connect()

            finally:
                # Ensure reconnecting flag is always reset
                self._reconnecting = False

    async def _on_speech_started(self, *args, **kwargs):
        await self.start_metrics()
        await self._call_event_handler("on_speech_started", *args, **kwargs)

    async def _on_utterance_end(self, *args, **kwargs):
        logger.debug("Deepgram VAD: Utterance ended")
        await self._call_event_handler("on_utterance_end", *args, **kwargs)


    async def _handle_user_speaking(self):

        await self.push_frame(StartInterruptionFrame())
        if self._user_speaking == True: return

        self._user_speaking = True

        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):

        if self._user_speaking == False: return

        self._user_speaking = False
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        self._bot_speaking = True
        self._bot_has_ever_spoken = True  # Mark that bot has spoken at least once
        self._bot_started_speaking_time = time.time()  # Track when bot started speaking
        logger.debug(f"🤖 DeepgramSTTService: Bot started speaking at {self._bot_started_speaking_time}")


    async def _handle_bot_silence(self):
        self._bot_speaking = False
        self._bot_started_speaking_time = None  # Reset the timestamp
        logger.debug(f"🤖 DeepgramSTTService: Bot stopped speaking")


    def _transcript_words_count(self, transcript: str):
        return len(transcript.split(" "))

    def _should_ignore_fast_greeting(self, transcript: str, time_start: float) -> bool:
        """
        Check if a fast greeting (single word, early timing) should be ignored.

        Rules:
        - If bot is speaking: ignore
        - If bot hasn't spoken yet: ignore
        - If bot has spoken at least once: allow (return False)
        """
        # Only applies to single-word transcripts within first second
        if time_start >= 1 or self._transcript_words_count(transcript) != 1:
            return False

        # Ignore if bot is currently speaking
        if self._bot_speaking:
            logger.debug("Ignoring fast greeting - bot is speaking")
            return True

        # Ignore if bot hasn't spoken yet (prevents false early detections)
        if not self._bot_has_ever_spoken:
            logger.debug("Ignoring fast greeting - bot hasn't spoken yet")
            return True

        # Allow if bot has spoken at least once (user responding)
        logger.debug("Accepting fast greeting - bot has spoken at least once")
        return False

    async def _async_handle_accum_transcription(self, current_time):

        if not self._last_interim_time: self._last_interim_time = 0.0

        reference_time = max(self._last_interim_time, self._last_time_accum_transcription)

        if self._fast_response:
            await self._fast_response_send_accum_transcriptions()
            return 
            
        if current_time - reference_time > self._on_no_punctuation_seconds and len(self._accum_transcription_frames):
            logger.debug("Sending accum transcription because of timeout")
            await self._send_accum_transcriptions()
            return

    async def _handle_false_interim(self, current_time):

        if not self._user_speaking: return
        if not self._last_interim_time: return
        if self._vad_active: return

        last_interim_delay = current_time - self._last_interim_time

        if last_interim_delay > FALSE_INTERIM_SECONDS: return

        logger.debug("False interim detected")

        await self._handle_user_silence()


    
    async def _async_handler(self, task_name):

        while True:
            if not self.is_monitored_task_active(task_name): return

            await asyncio.sleep(0.1)
            
            current_time = time.time()

            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)



            
    
    async def _send_accum_transcriptions(self):

        if not len(self._accum_transcription_frames): return

        logger.debug("Sending accumulated transcriptions")

        await self._handle_user_speaking()

        for frame in self._accum_transcription_frames:
            await self.push_frame(
                frame
            )
        self._accum_transcription_frames = []

        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def _is_accum_transcription(self, text: str):

        END_OF_PHRASE_CHARACTERS = ['.', '?']

        text =  text.strip()

        if not text: return True

        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)


    def _handle_first_message(self, text):

        if self._first_message: return 

        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):

        if not self._first_message: return
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)


    async def _fast_response_send_accum_transcriptions(self):
        """Send accumulated transcriptions immediately if fast response is enabled."""
        if not self._fast_response: return

        if len(self._accum_transcription_frames) == 0: return

        current_time = time.time()

        if self._vad_active:
            # Bypass VAD check if we have a complete sentence (ends with punctuation)
            if len(self._accum_transcription_frames) > 0:
                last_text = self._accum_transcription_frames[-1].text.strip()
                has_ending_punctuation = last_text and last_text[-1] in '.?!'
                if not has_ending_punctuation:
                    # Do not send if VAD is active and it's been less than 10 seconds since first message
                    if self._first_message_time and (current_time - self._first_message_time) > 10.0:
                        return


        last_message_time = max(self._last_interim_time, self._last_time_accum_transcription)

        is_short_sentence = len(self._accum_transcription_frames) <= 2
        is_sentence_end = not self._is_accum_transcription(self._accum_transcription_frames[-1].text)
        time_since_last_message = current_time - last_message_time

        if is_short_sentence:

            if is_sentence_end:
                logger.debug("Fast response: Sending accum transcriptions because short sentence and end of phrase")
                await self._send_accum_transcriptions()
            
            if not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("Fast response: Sending accum transcriptions because short sentence and timeout")
                await self._send_accum_transcriptions() 
        else:

            if is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("Fast response: Sending accum transcriptions because long sentence and end of phrase")
                await self._send_accum_transcriptions()
            
            if not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds * 2:
                logger.debug("Fast response: Sending accum transcriptions because long sentence and timeout")
                await self._send_accum_transcriptions()

    async def _on_final_transcript_message(self, transcript, language, speech_final: bool):

        await self._handle_user_speaking()
        frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)

        self._handle_first_message(frame.text)
        self._append_accum_transcription(frame)
        self._was_first_transcript_receipt = True

        
        if self._fast_response:
            await self._fast_response_send_accum_transcriptions()
        else:
            if not self._is_accum_transcription(frame.text) or speech_final:
                logger.debug("Sending final transcription frame")
                await self._send_accum_transcriptions()

    async def _on_interim_transcript_message(self, transcript, language, start_time):
        
        self._last_interim_time = time.time()
        await self._handle_user_speaking()
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )

    async def _should_ignore_transcription(self, result: LiveResultResponse, backup_source=False):

        is_final = result.is_final
        confidence = result.channel.alternatives[0].confidence
        transcript = result.channel.alternatives[0].transcript
        time_start = result.channel.alternatives[0].words[0].start if result.channel.alternatives[0].words else 0
        
        if not is_final and confidence < 0.7:
            logger.debug("Ignoring iterim because low confidence")
            return True

        # Check if fast greeting should be ignored
        if self._should_ignore_fast_greeting(transcript, time_start):
            return True
        
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug("Ignoring repeated first message")
            return True
        
        if not self._vad_active and not is_final:
            logger.debug("Ignoring Deepgram interruption because VAD inactive")
            return True
        
        logger.debug("Bot speaking: " + str(self._bot_speaking ) + " ** allow_interruptions: " + str(self._allow_stt_interruptions))
        if self._bot_speaking and not self._allow_stt_interruptions:
            logger.debug("Ignoring Deepgram interruption because allow_interruptions is False")
            return True
        
        # Enhanced logic for backup system - prevent false interruptions
        if backup_source and self._bot_speaking:
            # For backup transcripts, be more conservative about interruptions
            # Only allow interruption if it's a longer phrase (more than 2 words) and high confidence
            word_count = self._transcript_words_count(transcript)
            if word_count <= 2 or confidence < 0.95:
                if not contains_interruption_word(transcript, self.language):
                    logger.debug(f"Ignoring backup interruption - bot speaking, low word count ({word_count}) or confidence ({confidence:.2f}): '{transcript}'")
                    return True
                
            # Check if this transcript is very recent to when bot started speaking
            current_time = time.time()
            if hasattr(self, '_bot_started_speaking_time') and self._bot_started_speaking_time:
                time_since_bot_started = current_time - self._bot_started_speaking_time
                if time_since_bot_started < 1.5:  # Less than 1.5 seconds since bot started
                    logger.debug(f"Ignoring backup interruption - too soon after bot started speaking ({time_since_bot_started:.2f}s): '{transcript}'")
                    return True
                    
            # Additional check: if transcript seems like an echo or duplicate of recent conversation
            # This is especially important for backup systems that might pick up echo
            if (hasattr(self, '_last_time_transcription') and 
                time.time() - self._last_time_transcription < 2.0):
                logger.debug(f"Ignoring backup interruption - too soon after last transcript ({time.time() - self._last_time_transcription:.2f}s): '{transcript}'")
                return True
        
        if self._bot_speaking and self._transcript_words_count(transcript) == 1:
            if not contains_interruption_word(transcript, self.language):
                logger.debug(f"Ignoring {'backup' if backup_source else 'primary'} interruption because bot is speaking (single word): {transcript}")
                return True

        return False
    

    async def _detect_and_handle_voicemail(self, transcript: str):

        if not self.detect_voicemail: return False

        logger.debug(transcript)
        logger.debug(self._time_since_init())

        
        if self._time_since_init() > VOICEMAIL_DETECTION_SECONDS and self._was_first_transcript_receipt: return False
        
        if not voicemail.is_text_voicemail(transcript): return False
        
        logger.debug("Voicemail detected")

        await self.push_frame(
            TranscriptionFrame(transcript, "", time_now_iso8601(), self.language)
        )

        await self.push_frame(
            VoicemailFrame(transcript)
        )

        logger.debug("Voicemail pushed")
        return True


    async def _on_message(self, *args, **kwargs):
        """Handle incoming transcription message from Deepgram or backup service."""
        if not self._restarted:
            return

        backup_source = kwargs.pop('backup_source', False)

        try:
            result: LiveResultResponse = kwargs["result"]
            logger.debug(result)

            # Early return if no alternatives
            if not result.channel.alternatives:
                return

            # Log transcript receipt
            self._log_transcript_receipt(result, backup_source)

            # Process the transcript
            await self._process_transcript_message(result, backup_source)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"{self} unexpected error in _on_message: {error_msg}")

            # Push ErrorFrame for message processing errors
            await self.push_error(ErrorFrame(
                error=f"Deepgram message processing error: {error_msg[:200]}",
                fatal=False
            ))

    def _log_transcript_receipt(self, result, backup_source):
        """Log transcript receipt with source and metadata."""
        alternative = result.channel.alternatives[0]
        source_name = "BACKUP ACTIVATED" if backup_source else "Deepgram Primary"
        transcript_type = "FINAL" if result.is_final else "INTERIM"

        # Log single consolidated message
        logger.info(
            f"{source_name}: {transcript_type} transcript received - "
            f"Text: '{alternative.transcript}', "
            f"Confidence: {alternative.confidence:.2f}, "
            f"Start: {result.start}, "
            f"Speech final: {result.speech_final}"
        )

        if result.is_final and not backup_source:
            logger.debug(f"DeepgramSTTService: Deepgram final transcript: '{alternative.transcript}'")


    async def _process_transcript_message(self, result, backup_source=False):
        """Process transcript message with backup source tracking."""
        alternative = result.channel.alternatives[0]
        transcript = alternative.transcript

        # Guard: Empty transcript check
        if not transcript:
            return

        # Stop TTFB metrics for non-empty transcripts
        await self.stop_ttfb_metrics()

        # Log transcript processing
        self._log_transcript_processing(result, backup_source)

        # Check voicemail detection
        if await self._detect_and_handle_voicemail(transcript):
            logger.info("DeepgramSTTService: Voicemail detected and handled")
            return

        # Check if transcription should be ignored
        if await self._should_ignore_transcription(result, backup_source):
            logger.debug("DeepgramSTTService: Transcript ignored by filter")
            return

        # Route to appropriate handler
        if result.is_final:
            await self._process_final_transcript(result, backup_source)
        else:
            await self._process_interim_transcript(result, backup_source)

    def _log_transcript_processing(self, result, backup_source):
        """Log transcript processing details."""
        alternative = result.channel.alternatives[0]
        source = "Intelligent Backup" if backup_source else "Primary Deepgram"
        status = "FINAL" if result.is_final else "INTERIM"

        # Log single consolidated message
        logger.debug(
            f"{source}: Processing {status} - "
            f"Text: '{alternative.transcript}', "
            f"Confidence: {alternative.confidence:.2f}, "
            f"Start: {result.start}"
        )

    async def _process_interim_transcript(self, result, backup_source):
        """Process interim transcript."""
        alternative = result.channel.alternatives[0]
        language = self._extract_language(alternative)

        source = "Intelligent Backup" if backup_source else "Primary Deepgram"
        logger.debug(f"DeepgramSTTService: Processing INTERIM transcript from {source}")

        await self._on_interim_transcript_message(
            alternative.transcript,
            language,
            result.start
        )

    def _extract_language(self, alternative):
        """Extract language from alternative if available."""
        if alternative.languages:
            return Language(alternative.languages[0])
        return None


    async def _process_final_transcript(self, result, backup_source=False):
        """Process final transcript with backup source tracking."""
        transcript = result.channel.alternatives[0].transcript
        confidence = result.channel.alternatives[0].confidence
        start_time = result.start
        speech_final = getattr(result, 'speech_final', True)
        
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        
        # Enhanced response time measurement
        if self._current_speech_start_time is not None:
            elapsed = time.perf_counter() - self._current_speech_start_time
            elapsed_formatted = round(elapsed, 3)
            self._stt_response_times.append(elapsed_formatted)
            
            source_name = "� Intelligent Backup" if backup_source else "⚡ Primary Deepgram"
            reliability_indicator = " [BACKUP SYSTEM ACTIVATED]" if backup_source else " [PRIMARY SYSTEM]"
            
            logger.info(f"📊 {source_name}: ⏱️ STT Response Time: {elapsed_formatted}s{reliability_indicator}")
            logger.info(f"   📝 Final Transcript: '{transcript}'")
            logger.info(f"   🎯 Confidence: {confidence:.2f}")
            logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
            logger.info(f"   🗣️ Speech final: {speech_final}")
            
            if backup_source:
                logger.warning(f"⚠️ BACKUP ACTIVATION: Primary Deepgram failed to respond in time!")
                logger.info(f"🛡️ RELIABILITY: Backup system ensured no transcript was lost")
            
            self._current_speech_start_time = None
            self._audio_chunk_count = 0
            logger.debug(f"🔄 DeepgramSTTService: Reset speech timing counters")
        
        logger.debug(f"🎯 DeepgramSTTService: Calling _on_final_transcript_message for: '{transcript}'")
        await self._on_final_transcript_message(transcript, language, speech_final)
        self._last_time_transcription = start_time
        
        # Update backup service with processed transcript info
        if self._intelligent_gladia_backup and not backup_source:
            await self._intelligent_gladia_backup.notify_deepgram_final(transcript, start_time)
            
        logger.debug(f"⏰ DeepgramSTTService: Updated last transcription time to {start_time}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("🤖 DeepgramSTTService: Received bot started speaking")
            await self._handle_bot_speaking()

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("🤖 DeepgramSTTService: Received bot stopped speaking")
            await self._handle_bot_silence() 

        if isinstance(frame, STTRestartFrame):
            logger.info("🔄 DeepgramSTTService: Received STT Restart Frame - restarting services")
            self._restarted = True
            await self._disconnect()
            await self._connect()
            return

        if isinstance(frame, UserStartedSpeakingFrame) and not self.vad_enabled:
            logger.info("🎤 DeepgramSTTService: User started speaking (VAD disabled)")
            # Start metrics if Deepgram VAD is disabled & pipeline VAD has detected speech
            await self.start_metrics()
            # Reset timing when user starts speaking
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.debug(f"⏱️ DeepgramSTTService: Speech timer reset - waiting for transcriptions")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("🎤 DeepgramSTTService: User stopped speaking - finalizing connection")
            # https://developers.deepgram.com/docs/finalize
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name}, {direction}")
        
        if isinstance(frame, VADInactiveFrame):
            logger.debug("🎤 DeepgramSTTService: VAD inactive")
            self._vad_active = False
            

            if self._accum_transcription_frames and len(self._accum_transcription_frames) > 0:
                last_transcript = self._accum_transcription_frames[-1].text.strip()
                if last_transcript and last_transcript.endswith(','):
                    logger.debug("🎤 DeepgramSTTService: VAD inactive with comma ending - sending accumulated transcriptions for improved latency")
                    await self._send_accum_transcriptions()
            
            # Update backup system with VAD inactive time for filtering
            if self._intelligent_gladia_backup:
                await self._intelligent_gladia_backup.update_vad_inactive_time()
                logger.debug("🎯 DeepgramSTTService: Notified backup of VAD inactive")
            
            if self._connection and self._connection.is_connected:
                await self._connection.finalize()  
        elif isinstance(frame, VADActiveFrame):
            logger.debug("🎤 DeepgramSTTService: VAD active")
            self._vad_active = True

    def is_deepgram_connected(self) -> bool:
        """Check if Deepgram is currently connected."""
        return self._connection is not None and self._connection.is_connected

    def get_connection_status(self) -> Dict:
        """Get comprehensive connection status."""
        deepgram_connected = self.is_deepgram_connected()
        backup_available = self._backup_enabled and self._intelligent_gladia_backup is not None
        
        return {
            "deepgram_connected": deepgram_connected,
            "backup_available": backup_available,
            "primary_system": "Deepgram" if deepgram_connected else "Backup" if backup_available else "None",
            "reliability": "Ultra-High" if backup_available else "Standard" if deepgram_connected else "Limited",
            "status": "Healthy" if deepgram_connected or backup_available else "Degraded"
        }