#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
from aiohttp import WSMsgType
from aiohttp.web import WebSocketResponse
import wave
from aiohttp import web
from typing import Awaitable, Callable
from pydantic import BaseModel

import base64
import json
import time
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
    Frame,
    StartInterruptionFrame,
    AudioRawFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from urllib.parse import parse_qs
from websockets.http import Headers
import os
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from loguru import logger

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class WebsocketServerParams(TransportParams):
    add_wav_header: bool = False
    audio_frame_size: int = 6400  # 200ms
    serializer: FrameSerializer = ProtobufFrameSerializer()


class WebsocketServerCallbacks(BaseModel):
    on_client_connected: Callable[[websockets.WebSocketServerProtocol, str, str], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol, str], Awaitable[None]]


class WebsocketServerInputTransport(BaseInputTransport):

    def __init__(
            self,
            host: str,
            port: int,
            params: WebsocketServerParams,
            callbacks: WebsocketServerCallbacks,
            **kwargs
        ):
        super().__init__(params, **kwargs)

        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: WebSocketResponse | None = None

        self._stop_server_event = asyncio.Event()
        self._runner = None
        self._site = None
        self.last_log_time = time.time()  # Add this line
        self._shutdown_task = None  # Add this line to track shutdown task



    async def start(self, frame: StartFrame):
        self._server_task = self.get_event_loop().create_task(self._server_task_handler())
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._stop_server_event.set()
        await self._server_task

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._stop_server_event.set()
        await self._server_task


    async def _server_task_handler(self):
        app = web.Application()
        app.router.add_get('/ws', self.http_handler)
        
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        
        await self._stop_server_event.wait()
        
        # Cleanup server resources
        await self._site.stop()
        await self._runner.cleanup()

    def auth_user(self, headers: Headers, headers_as_qs = None) -> str | None:
        token = None

        # Try Authorization header first
        auth_header = headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split('Bearer ')[-1]

        # If no token found, try Sec-WebSocket-Protocol
        if headers_as_qs and token is None:
            try:
                token = headers_as_qs.get('Authorization', '').split('Bearer ')[-1]
            except (base64.binascii.Error, json.JSONDecodeError) as e:
                logger.warning(f"Failed to decode query params: {e}")
        
        if not token:
            logger.warning("No token found in Authorization header or Sec-WebSocket-Protocol")
            return None

        try:
            # Replace 'your-secret-key' with your actual secret key
            decoded = jwt.decode(token, os.getenv('JWT_SECRET_KEY'), algorithms=['HS256'])
            user_id = decoded.get('sub')
            logger.info(f"Authenticated user ID: {user_id}")
            return str(user_id)
        except ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except InvalidTokenError:
            logger.warning("Invalid token")
            return None

    def get_ai_soul_id(self, headers: Headers, headers_as_qs = None) -> str | None:
        ai_soul_id_header = headers.get('X-Ai-Soul-Id')

        if headers_as_qs and ai_soul_id_header is None:
            ai_soul_id_header = headers_as_qs.get('X-Ai-Soul-Id')

        return ai_soul_id_header

    async def check_machine_id_and_redirect(self, headers: Headers, headers_as_qs = None):
        # Parse machine ID from X-Machine-ID header
        machine_id = headers.get('X-Machine-ID')

        if headers_as_qs and machine_id is None:
            machine_id = headers_as_qs.get('X-Machine-ID')

        logger.info(f"Received machine id: {machine_id}")

        current_machine_id = os.getenv('FLY_MACHINE_ID', 'default')

        if current_machine_id == "default": 
            #We just pass here
            return None

        if not machine_id:
            logger.warning("No machine ID provided")
            return ('400 Bad Request', [], b'No machine ID provided')

        if machine_id != current_machine_id:
            logger.info(f"Redirecting to machine id {machine_id}")
            return ('307 Temporary Redirect', [('Fly-Replay', f'instance={machine_id}')], b'Redirecting...')

        return None

    async def http_handler(self, request):
        headers = Headers(request.headers)
        # This is for browsers as we can't use Headers with WebSocket browser API
        # This value should be send as base64 encode (e.g /ws?headers=...)
        headers_as_qs = str(request.rel_url.query.get('headers', ''))  # or whatever your parameter name is
        if headers_as_qs:
            headers_as_qs_dict = json.loads(base64.b64decode(headers_as_qs).decode('utf-8'))
        else:
            headers_as_qs_dict = None
        print("Handling http with" )
        user = self.auth_user(headers, headers_as_qs_dict)
        ai_soul_id = self.get_ai_soul_id(headers, headers_as_qs_dict)
        if not user:
            logger.warning("Authentication failed")
            return web.Response(status=401, text="Authentication failed")
        logger.info(f"User authenticated: {user}")
        # Check if the request should be redirected
        redirect_response = await self.check_machine_id_and_redirect(headers, headers_as_qs_dict)
        if redirect_response is not None:
            status, headers, body = redirect_response
            logger.info(f"Redirecting with status {status}")
            # if not self._websocket and not self._shutdown_task:
            #     self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return web.Response(status=int(status.split()[0]), headers=dict(headers), body=body)
        
        if self._shutdown_task:
            self._shutdown_task.cancel()
            self._shutdown_task = None
            
        if self._websocket is not None:
            logger.warning("Only one client allowed, rejecting new connection")
            return web.Response(status=400, text="Only one client allowed")

        # Proceed with WebSocket upgrade if no redirect
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self.websocket_handler(ws, user=user, ai_soul_id=ai_soul_id)
        return ws


    async def websocket_handler(self, websocket: WebSocketResponse, user: str, ai_soul_id: str):
        if self._websocket is not None:
            logger.warning("Only one client allowed, rejecting new connection")
            await websocket.close()
            return

        self._websocket = websocket
        logger.info(f"WebSocket handler initiated: {self._websocket}")

        # Notify
        await self._callbacks.on_client_connected(websocket, user, ai_soul_id)

        # Handle incoming messages
        async for message in websocket:
            # Check heartbeat timeout first
            # if time.time() - last_pong_time > 90:  # 90 seconds without ping
            #     logger.warning("Client heartbeat timeout, closing connection")
            #     await websocket.close()
            #     break
            
            if message.type == WSMsgType.BINARY:
                    frame = self._params.serializer.deserialize(message.data)
            elif message.type == WSMsgType.TEXT:
                continue

            if not frame:
                continue

            
            if isinstance(frame, AudioRawFrame):
                time_since_last_log = time.time() - self.last_log_time
                if time_since_last_log > 10:
                    logger.info("Pushing audio frame")
                    self.last_log_time = time.time()
                    
                await self.push_audio_frame(
                    InputAudioRawFrame(
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    )
                )
            else:
                await self.push_frame(frame)

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket, user)

        await self._websocket.close()        
        self._websocket = None

        logger.info(f"Client disconnected")


class WebsocketServerOutputTransport(BaseOutputTransport):
    def __init__(self, params: WebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)

        self._params = params

        self._websocket: WebSocketResponse | None = None

        self._websocket_audio_buffer = bytes()

        self._send_interval = (self._audio_chunk_size / self._params.audio_out_sample_rate) / 2
        self._next_send_time = 0

    async def set_client_connection(self, websocket: WebSocketResponse | None):
        logger.info(F"Setting client connection for {self._websocket} to {websocket}")
        if self._websocket:
            await self._websocket.close()
            logger.warning("Only one client allowed, using new connection")
        self._websocket = websocket

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
                
            # await self._write_frame(frame)
            #Here we send our trasport message to the client
            logger.info("=======================WE ARE INTERRUPTING===============================>")
            interrupt_transport = TransportMessageFrame(json.dumps({"type": "interrupt"}))
            await self.send_message(interrupt_transport)
            self._next_send_time = 0

    async def write_raw_audio_frames(self, frames: bytes):
        
        if self._websocket == None:
            return
        
        self._websocket_audio_buffer += frames
        while len(self._websocket_audio_buffer) >= self._params.audio_frame_size:
            frame = OutputAudioRawFrame(
                audio=self._websocket_audio_buffer[: self._params.audio_frame_size],
                sample_rate=self._params.audio_out_sample_rate,
                num_channels=self._params.audio_out_channels,
            )

            if self._params.add_wav_header:
                content = io.BytesIO()
                ww = wave.open(content, "wb")
                ww.setsampwidth(2)
                ww.setnchannels(frame.num_channels)
                ww.setframerate(frame.sample_rate)
                ww.writeframes(frame.audio)
                ww.close()
                content.seek(0)
                wav_frame = OutputAudioRawFrame(
                    audio=content.read(), sample_rate=frame.sample_rate, num_channels=frame.num_channels
                )
                frame = wav_frame

            proto = self._params.serializer.serialize(frame)
            if proto:
                await self._websocket.send_bytes(proto)

            self._websocket_audio_buffer = self._websocket_audio_buffer[
                self._params.audio_frame_size :
            ]            

    async def _write_frame(self, frame: Frame):
        payload = self._params.serializer.serialize(frame)
        if payload and (self._websocket != None):
            await self._websocket.send_bytes(payload)

    async def _write_audio_sleep(self):
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval
            
    async def send_message(self, message):
        if self._websocket is not None and not self._websocket.closed:
            try:
                logger.info(f"Sending message: {message}")
                proto = self._params.serializer.serialize(message)
                if proto:
                    await self._websocket.send_bytes(proto)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")


class WebsocketServerTransport(BaseTransport):

    def __init__(
            self,
            host: str = "localhost",
            port: int = 7860,
            params: WebsocketServerParams = WebsocketServerParams(),
            input_name: str | None = None,
            output_name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected
        )
        self._input: WebsocketServerInputTransport | None = None
        self._output: WebsocketServerOutputTransport | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self) -> WebsocketServerInputTransport:
        if not self._input:
            self._input = WebsocketServerInputTransport(
                self._host, self._port, self._params, self._callbacks, name=self._input_name)
        return self._input

    def output(self) -> WebsocketServerOutputTransport:
        if not self._output:
            self._output = WebsocketServerOutputTransport(self._params, name=self._output_name)
        return self._output

    async def _on_client_connected(self, websocket, user, ai_soul_id):
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket, user, ai_soul_id)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket, user):
        if self._output:
            await self._output.set_client_connection(None)
            await self._call_event_handler("on_client_disconnected", websocket, user)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")