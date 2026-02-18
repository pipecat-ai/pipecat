"""Hume.ai EVI (Empathic Voice Interface) WebSocket client.
Handles speech-to-speech conversation with Hume's API.
"""

import asyncio
import base64
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Optional

import websockets
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class SimpleHumeClient:
    """Simplified Hume client using direct WebSocket connection.
    More control over the connection for our POC.
    """

    def __init__(
        self,
        on_message: Callable[[Any], None],
        api_key: str | None = None,
        config_id: str | None = None,
        system_prompt: str | None = None,
    ):
        self.on_message = on_message
        self.api_key = api_key or os.getenv("HUME_API_KEY")
        self.config_id = config_id or os.getenv("HUME_CONFIG_ID")
        self.system_prompt = system_prompt

        self.ws = None
        self.is_connected = False
        self._receive_task = None
        self.chat_id: str | None = None

    async def connect(self):
        """Connect to Hume EVI via WebSocket."""
        try:
            # Hume EVI WebSocket URL
            url = "wss://api.hume.ai/v0/evi/chat"

            # Headers for authentication
            headers = {
                "X-Hume-Api-Key": self.api_key,
            }

            self.ws = await websockets.connect(
                url, extra_headers=headers, max_size=20 * 1024 * 1024
            )

            # Receive initial chat metadata
            raw = await self.ws.recv()
            meta = json.loads(raw)
            self.chat_id = meta.get("chat_id", "")
            logger.info(f"Hume session connected: chat_id={self.chat_id}")
            self.is_connected = True
            logger.info("Connected to Hume EVI WebSocket")

            # Send session settings
            await self._send_session_settings()

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Hume connection error: {e}")
            raise

    async def _send_session_settings(self):
        """Send initial session configuration."""
        settings = {
            "type": "session_settings",
            "audio": {
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
            },
        }

        if self.config_id:
            settings["config_id"] = self.config_id

        if self.system_prompt:
            settings["system_prompt"] = self.system_prompt

        await self.ws.send(json.dumps(settings))

    async def _receive_loop(self):
        """Receive and process messages from Hume."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                self.on_message(data)
        except Exception as e:
            if self.is_connected:
                logger.error(f"Receive loop error: {e}")
        finally:
            self.is_connected = False

    async def send_audio(self, audio_data: bytes):
        """Send audio to Hume."""
        if not self.is_connected or not self.ws:
            return

        try:
            message = {
                "type": "audio_input",
                "data": base64.b64encode(audio_data).decode("utf-8"),
            }
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def disconnect(self):
        """Disconnect from Hume."""
        self.is_connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("Disconnected from Hume")
