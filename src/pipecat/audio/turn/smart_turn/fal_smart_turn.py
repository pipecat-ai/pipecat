#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

import aiohttp

from pipecat.audio.turn.smart_turn.http_smart_turn import HttpSmartTurnAnalyzer


class FalSmartTurnAnalyzer(HttpSmartTurnAnalyzer):
    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        url: str = "https://fal.run/fal-ai/smart-turn/raw",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        headers = {}
        if api_key:
            headers = {"Authorization": f"Key {api_key}"}
        super().__init__(url=url, aiohttp_session=aiohttp_session, headers=headers, **kwargs)
