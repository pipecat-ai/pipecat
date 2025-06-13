#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts



class HumeTTSService(TTSService):
    class UtteranceParams:
        def __init__(
            self,
            description: Optional[str] = None,
            speed: Optional[float] = None,
            trailing_silence: Optional[float] = None,
            voice_id: Optional[str] = None,
            voice_provider: Optional[str] = None,
        ):
            self.description = description
            self.speed = speed
            self.trailing_silence = trailing_silence
            self.voice_id = voice_id
            self.voice_provider = voice_provider

    def __init__(
        self,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        utterance_params: Optional[UtteranceParams] = None,
        format_type: str = "mp3",
        sample_rate: int = 48000,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )
        self._api_key = api_key
        self._session = aiohttp_session
        self._utterance_params = utterance_params or self.UtteranceParams()
        self._format_type = format_type
        self._base_url = "https://api.hume.ai/v0/tts/stream/json"
        self._cumulative_time = 0
        self._started = False

    def _build_utterance_dict(self, text: str) -> Dict[str, Any]:
        utt = {"text": text}
        if self._utterance_params.description:
            utt["description"] = self._utterance_params.description
        if self._utterance_params.speed:
            utt["speed"] = self._utterance_params.speed
        if self._utterance_params.trailing_silence:
            utt["trailing_silence"] = self._utterance_params.trailing_silence
        if self._utterance_params.voice_id:
            utt["voice"] = {"id": self._utterance_params.voice_id}
            if self._utterance_params.voice_provider == "HUME_AI":
                utt["voice"]["provider"] = "HUME_AI"
        return utt


    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        headers = {
            "X-Hume-Api-Key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload = {
            "utterances": [self._build_utterance_dict(text)],
            "format": {"type": self._format_type},
            "instant_mode": False,
            "strip_headers": False,
            "num_generations": 1,
            "split_utterances": True,
        }

        try:
            await self.start_ttfb_metrics()
            async with self._session.post(self._base_url, headers=headers, json=payload) as response:
                # print(json.dumps(payload, indent=2))
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Hume TTS API error: {error_text}")
                    return

                yield TTSStartedFrame()

                async for line in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                    line_str = line.decode("utf-8").strip()
                    # print("[SSE]", line)
                    if not line_str.startswith("data:"):
                        continue

                    try:
                        data = json.loads(line_str.removeprefix("data:").strip())

                        if "audio" in data:
                            await self.stop_ttfb_metrics()
                            audio = base64.b64decode(data["audio"])
                            yield TTSAudioRawFrame(audio, self.sample_rate, 1)

                    except Exception as e:
                        logger.error(f"HumeTTSService: Failed to parse line: {line_str}", exc_info=True)
                        continue

                yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"HumeTTSService error: {e}", exc_info=True)
            yield ErrorFrame(error=str(e))