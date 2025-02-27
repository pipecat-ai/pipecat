#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.webrtc.webrtc_connection import PipecatWebRTCConnection


class PipecatWebRTCCallbacks(BaseModel):
    on_client_connected: Callable[[PipecatWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[PipecatWebRTCConnection], Awaitable[None]]
    on_client_closed: Callable[[PipecatWebRTCConnection], Awaitable[None]]


class PipecatWebRTCClient:
    def __init__(
        self, webrtc_connection: PipecatWebRTCConnection, callbacks: PipecatWebRTCCallbacks
    ):
        self._webrtcConnection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

    # TODO implement to receive the audio

    async def send(self, data: str | bytes):
        if self._can_send():
            # TODO implement to send the audio
            pass

    async def connect(self):
        # I guess there is nothing to do here, we should already be connected when we receive the PipecatWebRTCConnection
        pass

    async def disconnect(self):
        if self.is_connected and not self.is_closing:
            self._closing = True
            await self._webrtcConnection.close()
            await self.trigger_client_disconnected()

    async def trigger_client_disconnected(self):
        await self._callbacks.on_client_disconnected(self._webrtcConnection)

    async def trigger_client_connected(self):
        await self._callbacks.on_client_connected(self._webrtcConnection)

    async def trigger_client_timout(self):
        await self._callbacks.on_session_timeout(self._webrtcConnection)

    def _can_send(self):
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        return self._webrtcConnection.is_connected()

    @property
    def is_closing(self) -> bool:
        return self._closing


class PipecatWebRTCInputTransport(BaseInputTransport):
    def __init__(
        self,
        client: PipecatWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.trigger_client_connected()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_messages())

    async def _stop_tasks(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_messages(self):
        # TODO: implement to read the audio and push a new InputRawAudioFrame
        pass


class PipecatWebRTCOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        client: PipecatWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        # TODO: implement it, we should send through the datachannel
        pass

    async def write_raw_audio_frames(self, frames: bytes):
        # TODO: implement it
        pass


class PipecatWebRTCTransport(BaseTransport):
    def __init__(
        self,
        webrtc_connection: PipecatWebRTCConnection,
        params: TransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = PipecatWebRTCCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_client_closed=self._on_client_closed,
        )

        self._client = PipecatWebRTCClient(webrtc_connection, self._callbacks)

        self._input = PipecatWebRTCInputTransport(self._client, self._params, name=self._input_name)
        self._output = PipecatWebRTCOutputTransport(
            self._client, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_client_closed")

    def input(self) -> PipecatWebRTCInputTransport:
        if not self._input:
            self._input = PipecatWebRTCInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> PipecatWebRTCOutputTransport:
        if not self._output:
            self._output = PipecatWebRTCOutputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def _on_client_connected(self, webrtc_connection):
        await self._call_event_handler("on_client_connected", webrtc_connection)

    async def _on_client_disconnected(self, webrtc_connection):
        await self._call_event_handler("on_client_disconnected", webrtc_connection)

    async def _on_client_closed(self, webrtc_connection):
        await self._call_event_handler("on_client_closed", webrtc_connection)
