#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import io
from typing import Dict

import httpx
import numpy as np
from loguru import logger

from pipecat.audio.turn.base_smart_turn import BaseSmartTurn, SmartTurnTimeoutException


class SmartTurnAnalyzer(BaseSmartTurn):
    def __init__(self, url: str, **kwargs):
        super().__init__(**kwargs)
        self.remote_smart_turn_url = url

        if not self.remote_smart_turn_url:
            logger.error("remote_smart_turn_url is not set.")
            raise Exception("remote_smart_turn_url must be provided.")

        self.client = httpx.AsyncClient(
            headers={"Connection": "keep-alive"},
            timeout=httpx.Timeout(self._params.stop_secs),
        )

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        logger.trace("Serializing NumPy array to bytes...")
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        serialized_bytes = buffer.getvalue()
        logger.trace(f"Serialized size: {len(serialized_bytes)} bytes")
        return serialized_bytes

    async def _send_raw_request(self, data_bytes: bytes):
        headers = {"Content-Type": "application/octet-stream"}
        logger.trace(
            f"Sending {len(data_bytes)} bytes as raw body to {self.remote_smart_turn_url}..."
        )
        try:
            response = await self.client.post(
                self.remote_smart_turn_url,
                content=data_bytes,
                headers=headers,
            )

            logger.trace("\n--- Response ---")
            logger.trace(f"Status Code: {response.status_code}")

            if response.is_success:
                try:
                    json_data = response.json()
                    logger.trace("Response JSON:")
                    logger.trace(json_data)
                    return json_data
                except httpx.DecodingError:
                    logger.trace("Response Content (non-JSON):")
                    logger.trace(response.text)
            else:
                logger.trace("Response Content (Error):")
                logger.trace(response.text)
                response.raise_for_status()

        except httpx.TimeoutException:
            logger.error(f"Request timed out after {self._params.stop_secs} seconds")
            raise SmartTurnTimeoutException(f"Request exceeded {self._params.stop_secs} seconds.")
        except httpx.RequestError as e:
            logger.error(f"Failed to send raw request to Daily Smart Turn: {e}")
            raise Exception("Failed to send raw request to Daily Smart Turn.")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, any]:
        serialized_array = self._serialize_array(audio_array)
        return await self._send_raw_request(serialized_array)
