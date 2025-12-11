#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

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
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService, WordTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

InworldTimestampType = Literal["WORD", "CHARACTER", "TIMESTAMP_TYPE_UNSPECIFIED"]


def calculate_word_times_from_inworld(
    timestamp_info: Dict[str, Any],
    cumulative_time: float,
    timestamp_type: str = "WORD",
) -> Tuple[List[Tuple[str, float]], float]:
    word_times: List[Tuple[str, float]] = []
    max_end_time = cumulative_time

    if timestamp_type == "WORD":
        alignment = timestamp_info.get("wordAlignment", {})
        words = alignment.get("words", [])
        start_times = alignment.get("wordStartTimeSeconds", [])
        end_times = alignment.get("wordEndTimeSeconds", [])

        if words and start_times and len(words) == len(start_times):
            for i, word in enumerate(words):
                word_start = cumulative_time + start_times[i]
                word_times.append((word, word_start))
                if end_times and i < len(end_times):
                    max_end_time = max(max_end_time, cumulative_time + end_times[i])

    elif timestamp_type == "CHARACTER":
        alignment = timestamp_info.get("characterAlignment", {})
        characters = alignment.get("characters", [])
        start_times = alignment.get("characterStartTimeSeconds", [])
        end_times = alignment.get("characterEndTimeSeconds", [])

        if characters and start_times and len(characters) == len(start_times):
            current_word = ""
            word_start_time = None

            for i, char in enumerate(characters):
                if char == " ":
                    if current_word and word_start_time is not None:
                        word_times.append((current_word, word_start_time))
                    current_word = ""
                    word_start_time = None
                else:
                    if word_start_time is None:
                        word_start_time = cumulative_time + start_times[i]
                    current_word += char

                if end_times and i < len(end_times):
                    max_end_time = max(max_end_time, cumulative_time + end_times[i])

            if current_word and word_start_time is not None:
                word_times.append((current_word, word_start_time))

    return (word_times, max_end_time)


class InworldTTSService(WordTTSService):
    """Inworld AI HTTP-based TTS service.

    Supports both streaming and non-streaming modes via the `streaming` parameter.
    Outputs LINEAR16 audio at configurable sample rates with word/character timestamps.
    Language is automatically detected from input text.
    """

    class InputParams(BaseModel):
        temperature: Optional[float] = None
        speaking_rate: Optional[float] = None
        timestamp_type: InworldTimestampType = "WORD"

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
        params: InputParams = None,
        **kwargs,
    ):
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or InworldTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session
        self._streaming = streaming
        self._timestamp_type = params.timestamp_type

        if streaming:
            self._base_url = "https://api.inworld.ai/tts/v1/voice:stream"
        else:
            self._base_url = "https://api.inworld.ai/tts/v1/voice"

        self._settings = {
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

        self._cumulative_time = 0.0
        self._started = False

        self.set_voice(voice_id)
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["audioConfig"]["sampleRateHertz"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._started = False
            self._cumulative_time = 0.0
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}] (streaming={self._streaming})")

        payload = {
            "text": text,
            "voiceId": self._settings["voiceId"],
            "modelId": self._settings["modelId"],
            "audioConfig": self._settings["audioConfig"],
        }

        if "temperature" in self._settings:
            payload["temperature"] = self._settings["temperature"]

        if self._timestamp_type != "TIMESTAMP_TYPE_UNSPECIFIED":
            payload["timestampType"] = self._timestamp_type

        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            await self.start_ttfb_metrics()

            if not self._started:
                self.start_word_timestamps()
                yield TTSStartedFrame()
                self._started = True
                self._cumulative_time = 0.0

            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Inworld API error: {error_text}")
                    yield ErrorFrame(error=f"Inworld API error: {error_text}")
                    return

                if self._streaming:
                    async for frame in self._process_streaming_response(response):
                        yield frame
                else:
                    async for frame in self._process_non_streaming_response(response):
                        yield frame

            await self.start_tts_usage_metrics(text)

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            await self.stop_all_metrics()

    async def _process_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Frame, None]:
        buffer = ""

        async for chunk in response.content.iter_chunked(1024):
            buffer += chunk.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line_str = line.strip()

                if not line_str:
                    continue

                try:
                    chunk_data = json.loads(line_str)

                    if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                        await self.stop_ttfb_metrics()
                        async for frame in self._process_audio_chunk(
                            base64.b64decode(chunk_data["result"]["audioContent"])
                        ):
                            yield frame

                    if "result" in chunk_data and "timestampInfo" in chunk_data["result"]:
                        timestamp_info = chunk_data["result"]["timestampInfo"]
                        word_times, new_cumulative = calculate_word_times_from_inworld(
                            timestamp_info, self._cumulative_time, self._timestamp_type
                        )
                        if word_times:
                            await self.add_word_timestamps(word_times)
                            self._cumulative_time = new_cumulative

                except json.JSONDecodeError:
                    continue

    async def _process_non_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Frame, None]:
        response_data = await response.json()

        if "audioContent" not in response_data:
            logger.error("No audioContent in Inworld API response")
            await self.push_error(ErrorFrame(error="No audioContent in response"))
            return

        if "timestampInfo" in response_data:
            timestamp_info = response_data["timestampInfo"]
            word_times, new_cumulative = calculate_word_times_from_inworld(
                timestamp_info, self._cumulative_time, self._timestamp_type
            )
            if word_times:
                await self.add_word_timestamps(word_times)
                self._cumulative_time = new_cumulative

        audio_data = base64.b64decode(response_data["audioContent"])

        if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
            audio_data = audio_data[44:]

        chunk_size = self.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            if chunk:
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )

    async def _process_audio_chunk(self, audio_chunk: bytes) -> AsyncGenerator[Frame, None]:
        if not audio_chunk:
            return

        audio_data = audio_chunk

        if len(audio_chunk) > 44 and audio_chunk.startswith(b"RIFF"):
            audio_data = audio_chunk[44:]

        if audio_data:
            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )


class InworldWebsocketTTSService(AudioContextWordTTSService):
    """Inworld AI WebSocket-based TTS service.

    Uses bidirectional WebSocket for lower latency streaming. Supports multiple
    independent audio contexts per connection (max 5). Outputs LINEAR16 audio
    with word/character timestamps.
    """

    class InputParams(BaseModel):
        temperature: Optional[float] = None
        speaking_rate: Optional[float] = None
        apply_text_normalization: Optional[str] = None
        timestamp_type: InworldTimestampType = "WORD"
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
        params: InputParams = None,
        **kwargs: Any,
    ):
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or InworldWebsocketTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._timestamp_type = params.timestamp_type
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
        self._context_id = None
        self._started = False
        self._cumulative_time = 0.0

        self.set_voice(voice_id)
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["audioConfig"]["sampleRateHertz"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._context_id:
            await self._send_close_context(self._context_id)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._started = False
            self._cumulative_time = 0.0
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)

        if self._context_id and self._websocket:
            logger.trace(f"Closing context {self._context_id} due to interruption")
            try:
                await self._send_close_context(self._context_id)
            except Exception as e:
                logger.error(f"{self} exception: {e}")
                await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
            self._context_id = None
            self._started = False
            self._cumulative_time = 0.0

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
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Inworld WebSocket TTS")
                if self._context_id:
                    try:
                        await self._send_close_context(self._context_id)
                    except Exception:
                        pass
                await self._websocket.close()
                logger.debug("Disconnected from Inworld WebSocket TTS")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            self._started = False
            self._context_id = None
            self._cumulative_time = 0.0
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message")
                continue

            result = msg.get("result", {})
            ctx_id = result.get("contextId") or result.get("context_id")

            status = result.get("status", {})
            if status.get("code", 0) != 0:
                error_msg = status.get("message", "Unknown error")
                await self.push_error(ErrorFrame(error=f"Inworld API error: {error_msg}"))
                continue

            if "error" in msg:
                await self.push_error(ErrorFrame(error=str(msg["error"])))
                continue

            audio_chunk = result.get("audioChunk", {})
            audio_b64 = audio_chunk.get("audioContent")

            if audio_b64:
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()
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

            timestamp_info = result.get("timestampInfo")
            if timestamp_info:
                word_times, new_cumulative = calculate_word_times_from_inworld(
                    timestamp_info, self._cumulative_time, self._timestamp_type
                )
                if word_times:
                    await self.add_word_timestamps(word_times)
                    self._cumulative_time = new_cumulative

            if "flushCompleted" in result or "contextClosed" in result:
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.remove_audio_context(ctx_id)
                await self.push_frame(TTSStoppedFrame())

    async def _send_context(self, context_id: str):
        create_config: Dict[str, Any] = {
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

        if self._timestamp_type != "TIMESTAMP_TYPE_UNSPECIFIED":
            create_config["timestampType"] = self._timestamp_type

        msg = {"create": create_config, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_text(self, context_id: str, text: str):
        msg = {"send_text": {"text": text}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_flush(self, context_id: str):
        msg = {"flush_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_close_context(self, context_id: str):
        msg = {"close_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating WebSocket TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            context_id = str(uuid.uuid4())
            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)

            self._context_id = context_id

            if not self._started:
                await self.start_ttfb_metrics()
                self.start_word_timestamps()
                yield TTSStartedFrame()
                self._started = True
                self._cumulative_time = 0.0

            await self._send_context(context_id)
            await self._send_text(context_id, text)
            await self._send_close_context(context_id)
            await self.start_tts_usage_metrics(text)

            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
            yield TTSStoppedFrame()
