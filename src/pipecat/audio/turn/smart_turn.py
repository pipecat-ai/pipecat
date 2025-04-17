#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import io
import os
from typing import Dict

import numpy as np
import requests
from loguru import logger

from pipecat.audio.turn.base_smart_turn import BaseSmartTurn


class SmartTurnAnalyzer(BaseSmartTurn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remote_smart_turn_url = os.getenv("REMOTE_SMART_TURN_URL")

        if not self.remote_smart_turn_url:
            logger.error("REMOTE_SMART_TURN_URL is not set.")
            raise Exception("REMOTE_SMART_TURN_URL environment variable must be provided.")

        # Use a session to reuse connections (keep-alive)
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

    def _serialize_array(self, audio_array: np.ndarray) -> bytes:
        logger.trace("Serializing NumPy array to bytes...")
        buffer = io.BytesIO()
        np.save(buffer, audio_array)
        serialized_bytes = buffer.getvalue()
        logger.trace(f"Serialized size: {len(serialized_bytes)} bytes")
        return serialized_bytes

    def _send_raw_request(self, data_bytes: bytes):
        headers = {"Content-Type": "application/octet-stream"}
        logger.trace(
            f"Sending {len(data_bytes)} bytes as raw body to {self.remote_smart_turn_url}..."
        )
        try:
            response = self.session.post(
                self.remote_smart_turn_url,
                data=data_bytes,
                headers=headers,
                timeout=60,
            )

            logger.trace("\n--- Response ---")
            logger.trace(f"Status Code: {response.status_code}")

            if response.ok:
                try:
                    logger.trace("Response JSON:")
                    logger.trace(response.json())
                    return response.json()
                except requests.exceptions.JSONDecodeError:
                    logger.trace("Response Content (non-JSON):")
                    logger.trace(response.text)
            else:
                logger.trace("Response Content (Error):")
                logger.trace(response.text)
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send raw request to Daily Smart Turn: {e}")
            raise Exception("Failed to send raw request to Daily Smart Turn.")

    def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, any]:
        serialized_array = self._serialize_array(audio_array)
        return self._send_raw_request(serialized_array)
