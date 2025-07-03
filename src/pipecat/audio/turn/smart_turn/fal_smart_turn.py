#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fal.ai smart turn analyzer implementation.

This module provides a smart turn analyzer that uses Fal.ai's hosted smart-turn model
for end-of-turn detection in conversations.

Note: To learn more about the smart-turn model, visit:
    - https://fal.ai/models/fal-ai/smart-turn/playground
    - https://github.com/pipecat-ai/smart-turn
"""

from typing import Optional

import aiohttp

from pipecat.audio.turn.smart_turn.http_smart_turn import HttpSmartTurnAnalyzer


class FalSmartTurnAnalyzer(HttpSmartTurnAnalyzer):
    """Smart turn analyzer using Fal.ai's hosted smart-turn model.

    Extends HttpSmartTurnAnalyzer to provide integration with Fal.ai's
    smart turn detection API endpoint with proper authentication.
    """

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        url: str = "https://fal.run/fal-ai/smart-turn/raw",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Fal.ai smart turn analyzer.

        Args:
            aiohttp_session: HTTP client session for making API requests.
            url: Fal.ai API endpoint URL for smart turn detection.
            api_key: API key for authenticating with Fal.ai service.
            **kwargs: Additional arguments passed to parent HttpSmartTurnAnalyzer.
        """
        headers = {}
        if api_key:
            headers = {"Authorization": f"Key {api_key}"}
        super().__init__(url=url, aiohttp_session=aiohttp_session, headers=headers, **kwargs)
