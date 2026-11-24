#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Haru LLM service: full-duplex speech-to-speech from a Haru vLLM-Omni server."""

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import numpy as np
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.haru.base import (
    HARU_FRAME_SAMPLES,
    HARU_SAMPLE_RATE,
    HaruBaseLLMService,
    HaruLLMSettings,
)

_REALTIME_PATH = "/v1/realtime"


def realtime_url(base_url: str) -> str:
    """Resolve a Haru server's address to its duplex realtime WebSocket endpoint.

    Args:
        base_url: The server's address, with an ``http(s)`` or ``ws(s)``
            scheme. May already include the ``/v1/realtime`` path and a query.

    Returns:
        The ``ws(s)://.../v1/realtime?duplex=1`` URL to connect to.
    """
    parts = urlsplit(base_url)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/")
    if not path.endswith(_REALTIME_PATH):
        path += _REALTIME_PATH
    query = parts.query
    if "duplex" not in parse_qs(query):
        query = f"{query}&duplex=1" if query else "duplex=1"
    return urlunsplit((scheme, parts.netloc, path, query, ""))


class HaruLLMService(HaruBaseLLMService):
    """Haru full-duplex speech-to-speech, served by a Haru vLLM-Omni server.

    Connects to the realtime duplex endpoint of a server running the
    ``haru-speech-vllm`` plugin (``vllm serve higurashi-ai/Haru-4B --omni``),
    wherever it is hosted: a GPU machine of your own, a Hugging Face
    Inference Endpoint or another cloud. See
    :class:`~pipecat.services.haru.base.HaruBaseLLMService` for how the model
    behaves in a pipeline, and
    :class:`~pipecat.services.haru.local.HaruLocalLLMService` to run the
    model in this process instead.

    Event handlers available:

    - on_session_started: Called once the model has run its prompt and is
      listening.

    Example::

        llm = HaruLLMService(
            base_url=os.environ["HARU_BASE_URL"],
            api_key=os.getenv("HARU_API_KEY"),
            voice_reference="voice.wav",
            settings=HaruLLMService.Settings(system_instruction="..."),
        )
    """

    Settings = HaruLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        voice_reference: str | os.PathLike | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Haru LLM service.

        Args:
            base_url: Address of the Haru vLLM-Omni server, for example
                ``http://localhost:8000`` or the URL of a hosted endpoint.
            api_key: Bearer token for an endpoint that requires one, such as
                a Hugging Face access token for a protected Inference
                Endpoint. A server started with plain ``vllm serve`` needs
                none.
            voice_reference: Path of an audio clip of the voice the model
                should speak with: one speaker, 3 seconds to 3 minutes of
                clean speech. Haru ships without voices.
            settings: Service settings. All of them are fixed once the
                conversation has started.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(voice_reference=voice_reference, settings=settings, **kwargs)

        self._url = realtime_url(base_url)
        self._api_key = api_key

        self._websocket = None
        self._receive_task = None
        self._disconnecting = False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    #
    # lifecycle
    #

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect to the server.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def cleanup(self):
        """Release resources at teardown."""
        await super().cleanup()
        await self._disconnect()

    #
    # session
    #

    async def _open_session(self, instructions: str):
        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "instructions": instructions,
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "sample_rate_hz": HARU_SAMPLE_RATE,
            "audio": {
                "input": {"sample_rate_hz": HARU_SAMPLE_RATE},
                "output": {"sample_rate_hz": HARU_SAMPLE_RATE},
            },
            # The model decides when to speak; the server detects no turns.
            "turn_detection": None,
        }
        options = self._session_options()
        if "temperature" in options:
            session["temperature"] = options.pop("temperature")
        session["extra_body"] = {"auto_response": True, **options}
        if self._voice_reference is not None:
            session["ref_audio"] = await asyncio.to_thread(_audio_data_uri, self._voice_reference)
        await self._send_event({"type": "session.update", "session": session})

    async def _close_session(self):
        if self._websocket and self._session_requested:
            await self._send_event({"type": "session.close"})
        await super()._close_session()
        await self._disconnect()

    async def _send_user_frame(self, audio: np.ndarray):
        await self._send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio.astype("<f4", copy=False).tobytes()).decode(
                    "ascii"
                ),
                "format": "pcm_f32le",
                "sample_rate_hz": HARU_SAMPLE_RATE,
            }
        )

    #
    # websocket communication
    #

    async def _connect(self):
        try:
            if self._websocket:
                return
            # `session.created` echoes the voice reference, up to 3 minutes of
            # audio, so incoming messages can't be capped at the default 1 MiB.
            self._websocket = await websocket_connect(
                uri=self._url, additional_headers=self._headers(), max_size=None
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            self._websocket = None
            await self.push_error(
                error_msg=f"Error connecting to {self._url}: {e}",
                exception=e,
                force_treat_as_permanent=True,
            )

    async def _disconnect(self):
        # `_disconnecting` gates every outgoing event, so it is cleared in a
        # finally: a teardown that fails part-way still leaves it reset.
        try:
            self._disconnecting = True
            self._session_started = False
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            self._disconnecting = False

    async def _send_event(self, event: dict[str, Any]):
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(event))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(
                error_msg=f"Error sending {event.get('type')}: {e}",
                exception=e,
                force_treat_as_permanent=True,
            )

    async def _receive_task_handler(self):
        assert self._websocket is not None
        try:
            async for message in self._websocket:
                try:
                    event = json.loads(message)
                except (TypeError, ValueError) as e:
                    logger.warning(f"{self}: ignoring unparseable server event: {e}")
                    continue
                await self._handle_server_event(event)
        except ConnectionClosed as e:
            if not self._disconnecting:
                await self.push_error(
                    error_msg=f"Connection closed: {e}",
                    exception=e,
                    force_treat_as_permanent=True,
                )

    async def _handle_server_event(self, event: dict[str, Any]):
        kind = event.get("type")
        if kind == "response.output_audio.delta":
            if not self._session_started:
                # The model's first frame: its prompt has run and it is listening.
                await self._session_ready()
            await self._push_model_audio(base64.b64decode(event["delta"]))
        elif kind == "response.output_audio_transcript.delta":
            await self._push_model_text(event.get("delta", ""))
        elif kind == "session.created":
            # The first frame appended makes the model run its prompt; the
            # audio of that frame is not used, so it is silence.
            await self._send_user_frame(np.zeros(HARU_FRAME_SAMPLES, dtype=np.float32))
        elif kind == "error":
            await self._handle_error(event)
        else:
            logger.trace(f"{self}: {kind}")

    async def _handle_error(self, event: dict[str, Any]):
        error = event.get("error")
        if isinstance(error, dict):
            details = str(error.get("message") or error.get("code") or error)
        else:
            details = str(error or event)
        if not self._session_started:
            # The model never got to listen, so this conversation will not start.
            await self.push_error(
                error_msg=f"Conversation failed to start: {details}",
                force_treat_as_permanent=True,
            )
        else:
            await self.push_error(error_msg=details)


def _audio_data_uri(path: str | os.PathLike) -> str:
    """Read an audio file as the ``data:`` URI the server takes a voice reference in."""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("audio/"):
        mime = "audio/wav"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
