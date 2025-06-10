#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
from typing import Any, Dict, Optional

import aiohttp
import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnTimeoutException


class HttpSmartTurnAnalyzer(BaseSmartTurn):
    def __init__(
        self,
        *,
        url: str,
        aiohttp_session: aiohttp.ClientSession,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = url
        self._headers = headers or {}
        self._aiohttp_session = aiohttp_session

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        logger.trace("Serializing NumPy array to bytes...")
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        serialized_bytes = buffer.getvalue()
        logger.trace(f"Serialized size: {len(serialized_bytes)} bytes")
        return serialized_bytes

    async def _send_raw_request(self, data_bytes: bytes) -> Dict[str, Any]:
        headers = {"Content-Type": "application/octet-stream"}
        headers.update(self._headers)

        try:
            timeout = aiohttp.ClientTimeout(total=self._params.stop_secs)

            async with self._aiohttp_session.post(
                self._url, data=data_bytes, headers=headers, timeout=timeout
            ) as response:
                logger.trace("\n--- Response ---")
                logger.trace(f"Status Code: {response.status}")

                # Check if successful
                if response.status != 200:
                    error_text = await response.text()
                    logger.trace("Response Content (Error):")
                    logger.trace(error_text)

                    if response.status == 500:
                        logger.warning(f"Smart turn service returned 500 error: {error_text}")
                        raise Exception(f"Server returned HTTP 500: {error_text}")
                    else:
                        response.raise_for_status()

                # Process successful response
                try:
                    json_data = await response.json()
                    logger.trace("Response JSON:")
                    logger.trace(json_data)
                    return json_data
                except aiohttp.ContentTypeError:
                    # Non-JSON response
                    text = await response.text()
                    logger.trace("Response Content (non-JSON):")
                    logger.trace(text)
                    raise Exception(f"Non-JSON response: {text}")

        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self._params.stop_secs} seconds")
            raise SmartTurnTimeoutException(f"Request exceeded {self._params.stop_secs} seconds.")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to send raw request to Daily Smart Turn: {e}")
            raise Exception("Failed to send raw request to Daily Smart Turn.")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        try:
            serialized_array = self._serialize_array(audio_array)
            return await self._send_raw_request(serialized_array)
        except Exception as e:
            logger.error(f"Smart turn prediction failed: {str(e)}")
            # Return an incomplete prediction when a failure occurs
            return {
                "prediction": 0,
                "probability": 0.0,
                "metrics": {"inference_time": 0.0, "total_time": 0.0},
            }
