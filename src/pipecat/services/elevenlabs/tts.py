#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, model_validator
import hashlib

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
from pipecat.services.tts_service import InterruptibleWordTTSService, TTSService
from pipecat.transcriptions.language import Language

# See .env.example for ElevenLabs configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use ElevenLabs, you need to `pip install pipecat-ai[elevenlabs]`.")
    raise Exception(f"Missing module: {e}")

ElevenLabsOutputFormat = Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"]

# Models that support language codes
# The following models are excluded as they don't support language codes:
# - eleven_flash_v2
# - eleven_turbo_v2
# - eleven_multilingual_v2
ELEVENLABS_MULTILINGUAL_MODELS = {
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
}


def language_to_elevenlabs_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BG: "bg",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SV: "sv",
        Language.TA: "ta",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


def output_format_from_sample_rate(sample_rate: int) -> str:
    match sample_rate:
        case 8000:
            return "pcm_8000"
        case 16000:
            return "pcm_16000"
        case 22050:
            return "pcm_22050"
        case 24000:
            return "pcm_24000"
        case 44100:
            return "pcm_44100"
    logger.warning(
        f"ElevenLabsTTSService: No output format available for {sample_rate} sample rate"
    )
    return "pcm_24000"


def build_elevenlabs_voice_settings(
    settings: Dict[str, Any],
) -> Optional[Dict[str, Union[float, bool]]]:
    """Build voice settings dictionary for ElevenLabs based on provided settings.

    Args:
        settings: Dictionary containing voice settings parameters

    Returns:
        Dictionary of voice settings or None if required parameters are missing
    """
    voice_settings = {}
    if settings["stability"] is not None and settings["similarity_boost"] is not None:
        voice_settings["stability"] = settings["stability"]
        voice_settings["similarity_boost"] = settings["similarity_boost"]
        if settings["style"] is not None:
            voice_settings["style"] = settings["style"]
        if settings["use_speaker_boost"] is not None:
            voice_settings["use_speaker_boost"] = settings["use_speaker_boost"]
        if settings["speed"] is not None:
            voice_settings["speed"] = settings["speed"]
    else:
        if settings["style"] is not None:
            logger.warning(
                "'style' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
            )
        if settings["use_speaker_boost"] is not None:
            logger.warning(
                "'use_speaker_boost' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
            )
        if settings["speed"] is not None:
            logger.warning(
                "'speed' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
            )

    return voice_settings or None


def calculate_word_times(
    alignment_info: Mapping[str, Any], cumulative_time: float
) -> List[Tuple[str, float]]:
    zipped_times = list(zip(alignment_info["chars"], alignment_info["charStartTimesMs"]))

    words = "".join(alignment_info["chars"]).split(" ")

    # Calculate start time for each word. We do this by finding a space character
    # and using the previous word time, also taking into account there might not
    # be a space at the end.
    times = []
    for i, (a, b) in enumerate(zipped_times):
        if a == " " or i == len(zipped_times) - 1:
            t = cumulative_time + (zipped_times[i - 1][1] / 1000.0)
            times.append(t)

    word_times = list(zip(words, times))

    return word_times


class SharedTTSCache:
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_order: List[str] = []
        self._cache_size = max_size
        self._cache_lock = asyncio.Lock()

    async def add(self, key: str, messages: List[Dict[str, Any]]):
        """Adiciona mensagens ao cache com política LRU."""
        async with self._cache_lock:
            if key in self._cache:
                self._cache_order.remove(key)
            
            self._cache[key] = messages
            self._cache_order.append(key)
            
            if len(self._cache_order) > self._cache_size:
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]

    async def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Obtém mensagens do cache e atualiza seu status LRU."""
        async with self._cache_lock:
            if key in self._cache:
                self._cache_order.remove(key)
                self._cache_order.append(key)
                return self._cache[key]
            return None

    async def clear(self):
        """Limpa todo o cache."""
        async with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()

    async def remove(self, key: str):
        """Remove uma entrada específica do cache."""
        async with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                self._cache_order.remove(key)

    def get_stats(self):
        """Retorna estatísticas sobre o cache."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "memory_usage_estimate": sum(
                len(str(msg)) for msgs in self._cache.values() for msg in msgs
            ),
        }

# Criar uma única instância compartilhada do cache
SHARED_TTS_CACHE = SharedTTSCache(max_size=1000)  # Aumentei para 1000 entradas

class ElevenLabsTTSService(InterruptibleWordTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = None
        optimize_streaming_latency: Optional[str] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None
        auto_mode: Optional[bool] = True

        @model_validator(mode="after")
        def validate_voice_settings(self):
            stability = self.stability
            similarity_boost = self.similarity_boost
            if (stability is None) != (similarity_boost is None):
                raise ValueError(
                    "Both 'stability' and 'similarity_boost' must be provided when using voice settings"
                )
            return self

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        url: str = "wss://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. ElevenLabs gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        #
        # Finally, ElevenLabs doesn't provide information on when the bot stops
        # speaking for a while, so we want the parent class to send TTSStopFrame
        # after a short period not receiving any audio.
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
            "optimize_streaming_latency": params.optimize_streaming_latency,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
            "speed": params.speed,
            "auto_mode": str(params.auto_mode).lower(),
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

        self._receive_task = None
        self._keepalive_task = None

        # Usar o cache compartilhado ao invés do cache por instância
        self._cache = SHARED_TTS_CACHE

        self._receive_lock = asyncio.Lock()  # Adicionar um lock para recebimento de mensagens

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_elevenlabs_language(language)

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        await self._disconnect()
        await self._connect()

    async def _update_settings(self, settings: Mapping[str, Any]):
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            await self._disconnect()
            await self._connect()
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._websocket:
            msg = {"text": " ", "flush": True}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("LLMFullResponseEndFrame", 0), ("Reset", 0)])

    async def _connect(self):
        await self._connect_websocket()

        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket:
                return

            logger.debug("Connecting to ElevenLabs")

            voice_id = self._voice_id
            model = self.model_name
            output_format = self._output_format
            url = f"{self._url}/v1/text-to-speech/{voice_id}/stream-input?model_id={model}&output_format={output_format}&auto_mode={self._settings['auto_mode']}"

            if self._settings["optimize_streaming_latency"]:
                url += f"&optimize_streaming_latency={self._settings['optimize_streaming_latency']}"

            # Language can only be used with the ELEVENLABS_MULTILINGUAL_MODELS
            language = self._settings["language"]
            if model in ELEVENLABS_MULTILINGUAL_MODELS and language is not None:
                url += f"&language_code={language}"
                logger.debug(f"Using language code: {language}")
            elif language is not None:
                logger.warning(
                    f"Language code [{language}] not applied. Language codes can only be used with multilingual models: {', '.join(sorted(ELEVENLABS_MULTILINGUAL_MODELS))}"
                )

            # Set max websocket message size to 16MB for large audio responses
            self._websocket = await websockets.connect(url, max_size=16 * 1024 * 1024)

            # According to ElevenLabs, we should always start with a single space.
            msg: Dict[str, Any] = {
                "text": " ",
                "xi_api_key": self._api_key,
            }
            if self._voice_settings:
                msg["voice_settings"] = self._voice_settings
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from ElevenLabs")
                await self._websocket.send(json.dumps({"text": ""}))
                await self._websocket.close()
                self._websocket = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async with self._receive_lock:  # Usar o lock para garantir acesso exclusivo
            async for message in self._get_websocket():
                msg = json.loads(message)
                if msg.get("audio"):
                    await self.stop_ttfb_metrics()
                    self.start_word_timestamps()

                    audio = base64.b64decode(msg["audio"])
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    await self.push_frame(frame)
                if msg.get("alignment"):
                    word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                    await self.add_word_timestamps(word_times)
                    self._cumulative_time = word_times[-1][1]

    async def _keepalive_task_handler(self):
        while True:
            await asyncio.sleep(10)
            try:
                await self._send_text("")
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    async def _send_text(self, text: str):
        if self._websocket:
            msg = {"text": text + " "}
            await self._websocket.send(json.dumps(msg))

    def _generate_cache_key(self, text: str) -> str:
        """Gera uma chave única baseada no texto e configurações atuais."""
        key_components = [
            text,
            self._voice_id,
            self.model_name,
            self._output_format,
            str(self._voice_settings),
            str(self._settings.get("language", "")),
        ]
        key_string = "_".join([str(comp) for comp in key_components])
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        # Adicionar este log para debug
        logger.debug(f"Cache key components: {key_components}")
        logger.debug(f"Generated cache key: {cache_key}")
        
        return cache_key

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Gera fala a partir do texto usando cache quando possível."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            cache_key = self._generate_cache_key(text)
            cached_messages = await self._cache.get(cache_key)  # Usar get() do cache compartilhado

            if cached_messages:
                logger.debug(f"{self}: Cache hit for text '{text}'")
                
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0

                # Processa mensagens do cache
                for msg in cached_messages:
                    if msg.get("audio"):
                        await self.stop_ttfb_metrics()
                        audio = base64.b64decode(msg["audio"])
                        yield TTSAudioRawFrame(audio, self.sample_rate, 1)
                    
                    if msg.get("alignment"):
                        word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                        await self.add_word_timestamps(word_times)
                        self._cumulative_time = word_times[-1][1]

            else:
                logger.debug(f"{self}: Cache miss for text '{text}'")
                messages_to_cache = []

                try:
                    if not self._started:
                        await self.start_ttfb_metrics()
                        yield TTSStartedFrame()
                        self._started = True
                        self._cumulative_time = 0

                    await self._send_text(text)
                    await self.start_tts_usage_metrics(text)

                    # Usar o lock para receber mensagens
                    async with self._receive_lock:
                        async for message in self._get_websocket():
                            msg = json.loads(message)
                            messages_to_cache.append(msg)

                            if msg.get("audio"):
                                await self.stop_ttfb_metrics()
                                audio = base64.b64decode(msg["audio"])
                                yield TTSAudioRawFrame(audio, self.sample_rate, 1)
                            
                            if msg.get("alignment"):
                                word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                                await self.add_word_timestamps(word_times)
                                self._cumulative_time = word_times[-1][1]

                    # Armazenar no cache compartilhado
                    await self._cache.add(cache_key, messages_to_cache)

                except Exception as e:
                    logger.error(f"{self} error sending message: {e}")
                    yield TTSStoppedFrame()
                    await self._disconnect()
                    await self._connect()
                    return

            yield None

        except Exception as e:
            logger.error(f"{self} exception: {e}")

    def get_cache_stats(self):
        """Retorna estatísticas sobre o cache compartilhado."""
        return self._cache.get_stats()


class ElevenLabsHttpTTSService(TTSService):
    """ElevenLabs Text-to-Speech service using HTTP streaming with caching.

    Args:
        api_key: ElevenLabs API key
        voice_id: ID of the voice to use
        aiohttp_session: aiohttp ClientSession
        model: Model ID (default: "eleven_flash_v2_5" for low latency)
        base_url: API base URL
        sample_rate: Output sample rate
        params: Additional parameters for voice configuration
        cache_size: Maximum number of cached responses to store (default: 100)
    """

    class InputParams(BaseModel):
        language: Optional[Language] = None
        optimize_streaming_latency: Optional[int] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "eleven_flash_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        cache_size: int = 100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._params = params
        self._session = aiohttp_session

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
            "optimize_streaming_latency": params.optimize_streaming_latency,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
            "speed": params.speed,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()
        
        # Cache for storing TTS results
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size
        self._cache_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)

    def _generate_cache_key(self, text: str) -> str:
        """Generate a unique cache key based on text and current settings.
        
        This ensures that caching considers not just the text but also the voice,
        model, and other parameters that affect the audio output.
        """
        key_components = [
            text,
            self._voice_id,
            self._model_name,
            self._output_format,
            str(self._voice_settings),
            self._settings.get("language", ""),
        ]
        key_string = "_".join([str(comp) for comp in key_components])
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _add_to_cache(self, key: str, audio_chunks: List[bytes]):
        """Add audio chunks to the cache with LRU eviction policy."""
        async with self._cache_lock:
            # If this key already exists in the cache, remove it to update its position
            if key in self._cache:
                self._cache_order.remove(key)
            
            # Add to cache
            self._cache[key] = audio_chunks
            self._cache_order.append(key)
            
            # Enforce cache size limit (LRU eviction)
            if len(self._cache_order) > self._cache_size:
                oldest_key = self._cache_order.pop(0)
                del self._cache[oldest_key]

    async def _get_from_cache(self, key: str) -> Optional[List[bytes]]:
        """Get audio chunks from cache and update its LRU status."""
        async with self._cache_lock:
            if key in self._cache:
                # Update LRU order
                self._cache_order.remove(key)
                self._cache_order.append(key)
                return self._cache[key]
            return None

    async def _fetch_and_cache_audio(self, text: str, cache_key: str) -> AsyncGenerator[bytes, None]:
        """Fetch audio from ElevenLabs API and cache the result."""
        url = f"{self._base_url}/v1/text-to-speech/{self._voice_id}/stream"

        payload: Dict[str, Union[str, Dict[str, Union[float, bool]]]] = {
            "text": text,
            "model_id": self._model_name,
        }

        if self._voice_settings:
            payload["voice_settings"] = self._voice_settings

        language = self._settings["language"]
        if self._model_name in ELEVENLABS_MULTILINGUAL_MODELS and language:
            payload["language_code"] = language
            logger.debug(f"Using language code: {language}")
        elif language:
            logger.warning(
                f"Language code [{language}] not applied. Language codes can only be used with multilingual models: {', '.join(sorted(ELEVENLABS_MULTILINGUAL_MODELS))}"
            )

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        # Build query parameters
        params = {
            "output_format": self._output_format,
        }
        if self._settings["optimize_streaming_latency"] is not None:
            params["optimize_streaming_latency"] = self._settings["optimize_streaming_latency"]

        logger.debug(f"ElevenLabs request - payload: {payload}, params: {params}")

        # Collect chunks for caching while also yielding them
        chunks_for_cache = []
        CHUNK_SIZE = 1024

        async with self._session.post(
            url, json=payload, headers=headers, params=params
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"{self} error: {error_text}")
                raise Exception(f"ElevenLabs API error: {error_text}")

            # Process the streaming response
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                if len(chunk) > 0:
                    chunks_for_cache.append(chunk)
                    yield chunk

        # Store in cache after successful completion
        await self._add_to_cache(cache_key, chunks_for_cache)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ElevenLabs streaming API with caching.

        Args:
            text: The text to convert to speech

        Yields:
            Frames containing audio data and status information
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Generate cache key based on text and current settings
        cache_key = self._generate_cache_key(text)

        try:
            await self.start_ttfb_metrics()
            
            # Check cache first
            cached_chunks = await self._get_from_cache(cache_key)
            
            if cached_chunks:
                logger.debug(f"{self}: Cache hit for text '{text}'")
                
                # Metrics for cached responses
                await self.start_tts_usage_metrics(text)
                
                yield TTSStartedFrame()
                
                # Process cached chunks
                for chunk in cached_chunks:
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
            else:
                logger.debug(f"{self}: Cache miss for text '{text}', fetching from API")
                
                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()
                
                # Fetch from API and cache simultaneously
                try:
                    async for chunk in self._fetch_and_cache_audio(text, cache_key):
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                except Exception as e:
                    logger.error(f"Error fetching from API: {e}")
                    yield ErrorFrame(error=str(e))
                    return
        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    async def clear_cache(self):
        """Clear the entire cache."""
        async with self._cache_lock:
            self._cache = {}
            self._cache_order = []
            
    async def remove_from_cache(self, text: str):
        """Remove a specific text entry from the cache."""
        cache_key = self._generate_cache_key(text)
        async with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._cache_order.remove(cache_key)
                
    def get_cache_stats(self):
        """Return statistics about the cache."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "memory_usage_estimate": sum(sum(len(chunk) for chunk in chunks) for chunks in self._cache.values()),
        }