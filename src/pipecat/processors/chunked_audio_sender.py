import asyncio
import io
import wave
from typing import Optional

from loguru import logger

from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.network.websocket_server import WebsocketServerOutputTransport, WebsocketServerParams, \
    WebsocketServerTransport

VONAGE_SAMPLE_RATE = 16000


class VonageWebsocketServerTransport(WebsocketServerTransport):

    def __init__(self,
                 params: WebsocketServerParams,
                 host: str = "localhost",
                 port: int = 8765,
                 input_name: Optional[str] = None,
                 output_name: Optional[str] = None, ):
        super().__init__(params, host, port, input_name, output_name)
        self._params = params

    def output(self) -> WebsocketServerOutputTransport:
        """Get the output transport for sending data to clients.

        Returns:
            The WebSocket server output transport instance.
        """
        if not self._output:
            self._output = VonageWebsocketServerOutputTransport(
                self,
                self._params
            )
        return self._output


class VonageWebsocketServerOutputTransport(WebsocketServerOutputTransport):

    def __init__(self, transport: BaseTransport, params: WebsocketServerParams, **kwargs):
        super().__init__(transport, params, **kwargs)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the WebSocket client with timing control.

        Args:
            frame: The output audio frame to write.
        """
        if not self._websocket:
            # Simulate audio playback with a sleep.
            await self._write_audio_sleep()
            return

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

    async def _write_frame(self, frame: Frame):
        """Serialize and send a frame to the WebSocket client."""
        if not self._params.serializer:
            return

        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._websocket:
                for chunk in payload:
                    await self._websocket.send(chunk)
                    await asyncio.sleep(self._params.serializer.sleep_interval)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")
