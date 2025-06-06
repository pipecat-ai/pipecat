#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import AsyncGenerator, Optional

from loguru import logger

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
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    raise Exception(f"Missing module: {e}")


class ResembleTTSService(AudioContextWordTTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice_uuid: str,
        url: str = "wss://websocket.cluster.resemble.ai/stream",
        sample_rate: Optional[int] = 32000,
        output_format: str = "wav",
        precision: str = "PCM_16",
        binary_response: bool = False,
        no_audio_header: bool = True, # avoid audio header included in the output
        text_aggregator: Optional[BaseTextAggregator] = None,
        **kwargs,
    ):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # We can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator or SkipTagsAggregator([("<spell>", "</spell>")]),
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._voice_uuid = voice_uuid
        self._output_format = output_format
        self._precision = precision
        self._binary_response = binary_response
        self._no_audio_header = no_audio_header
        
        self._websocket = None
        self._receive_task = None
        self._request_id = 0
        self._current_audio_buffer = bytearray()
        self._is_receiving_audio = False

    def can_generate_metrics(self) -> bool:
        return True

    def _build_msg(self, text: str = "", request_id: Optional[int] = None) -> str:
        if request_id is None:
            request_id = self._request_id
            self._request_id += 1
            
        msg = {
            "voice_uuid": self._voice_uuid,
            "data": text,
            "binary_response": self._binary_response,
            "request_id": request_id,
            "output_format": self._output_format,
            "sample_rate": self.sample_rate,
            "precision": self._precision,
            "no_audio_header": self._no_audio_header,
        }
        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

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
            if self._websocket and self._websocket.open:
                return
            logger.debug("Connecting to Resemble")

            self._websocket = await websockets.connect(self._url, extra_headers={"Authorization": f"Bearer {self._api_key}"})
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Resemble")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._current_audio_buffer = bytearray()
        self._is_receiving_audio = False

    async def flush_audio(self):
        if not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")

        self._current_audio_buffer = bytearray()
        self._is_receiving_audio = False

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                if self._binary_response:
                    await self._handle_binary_message(message)
                else:
                    await self._handle_json_message(message)
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _handle_binary_message(self, message: bytes):
        if isinstance(message, bytes):
            self._current_audio_buffer.extend(message)
            
            if not self._is_receiving_audio:
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()
                self._is_receiving_audio = True
            
            if len(self._current_audio_buffer) > 8192:  # 8KB chunks
                await self._send_audio_chunk()
        else:
            try:
                msg = json.loads(message)
                if msg.get("type") == "audio_end":
                    if self._current_audio_buffer:
                        await self._send_audio_chunk()
                    await self._handle_audio_end(msg)
                elif msg.get("type") == "error":
                    await self._handle_error(msg)
            except json.JSONDecodeError:
                # If it's not JSON, treat as binary audio data
                if isinstance(message, str):
                    message = message.encode()
                self._current_audio_buffer.extend(message)

    async def _send_audio_chunk(self):
        if self._current_audio_buffer:
            frame = TTSAudioRawFrame(
                audio=bytes(self._current_audio_buffer),
                sample_rate=self.sample_rate,
                num_channels=1,
            )
            await self.push_frame(frame)
            self._current_audio_buffer = bytearray()

    async def _handle_json_message(self, message: str):
        msg = json.loads(message)
        request_id = msg.get("request_id")
        
        if not request_id and not self._request_id and request_id != self._request_id:
            return
        
        if msg["type"] == "audio_end":
            await self._handle_audio_end(msg)
            
        elif msg["type"] == "audio":
            await self.stop_ttfb_metrics()
            self.start_word_timestamps()
            
            audio_data = base64.b64decode(msg["audio_content"])
            
            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=msg.get("sample_rate", self.sample_rate),
                num_channels=1,
            )
            await self.push_frame(frame)
            
        elif msg["type"] == "error":
            await self._handle_error(msg)

    async def _handle_audio_end(self, msg: dict):
        await self.stop_ttfb_metrics()
        await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)])
        await self.push_frame(TTSStoppedFrame())
        self._is_receiving_audio = False

    async def _handle_error(self, msg: dict):
        error_message = msg.get('message', 'Unknown error')
        logger.error(f"{self} error: {error_message}")
        await self.push_frame(TTSStoppedFrame())
        await self.stop_all_metrics()
        await self.push_error(ErrorFrame(f"{self} error: {error_message}"))
        self._is_receiving_audio = False
        self._current_audio_buffer = bytearray()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()
            
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()
            
            msg = self._build_msg(text=text)
            
            try:
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _receive_task_handler(self, error_callback):
        try:
            await self._receive_messages()
        except Exception as e:
            await error_callback(e)