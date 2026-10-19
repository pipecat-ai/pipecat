#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``on_word_timestamps`` reports a provider's word timings, ends included."""

import json
import unittest
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

from pipecat.frames.frames import Frame
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.tts_service import TTSService


class _TTS(TTSService):
    def __init__(self):
        super().__init__(sample_rate=24000)
        self.append_to_audio_context = AsyncMock()  # type: ignore[method-assign]
        self.live = {"ctx"}
        self.audio_context_available = lambda context_id: context_id in self.live  # type: ignore[method-assign]

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        yield None


def _record(service: TTSService) -> list:
    seen = []

    @service.event_handler("on_word_timestamps")
    async def on_word_timestamps(_service, context_id, words):
        seen.append((context_id, words))

    return seen


class TestWordTimestampsEvent(unittest.IsolatedAsyncioTestCase):
    async def test_words_carry_their_ends(self):
        tts = _TTS()
        seen = _record(tts)
        await tts.add_word_timestamps(
            [("Hello", 0.0), ("world", 0.4)], "ctx", word_ends=[0.35, 0.8]
        )
        self.assertEqual(seen, [("ctx", [("Hello", 0.0, 0.35), ("world", 0.4, 0.8)])])
        # The words still reach playback as before.
        self.assertEqual(tts.append_to_audio_context.await_count, 2)

    async def test_a_provider_without_ends_reports_none(self):
        tts = _TTS()
        seen = _record(tts)
        await tts.add_word_timestamps([("Hello", 0.0)], "ctx")
        self.assertEqual(seen, [("ctx", [("Hello", 0.0, None)])])

    async def test_a_stale_context_reports_nothing(self):
        tts = _TTS()
        seen = _record(tts)
        await tts.add_word_timestamps([("Hello", 0.0)], "gone", word_ends=[0.3])
        self.assertEqual(seen, [])

    async def test_mismatched_ends_are_refused(self):
        tts = _TTS()
        seen = _record(tts)
        with self.assertRaises(ValueError):
            await tts.add_word_timestamps([("Hello", 0.0), ("world", 0.4)], "ctx", word_ends=[0.3])
        with self.assertRaises(ValueError):
            await tts.add_word_timestamps(
                [("Hello", 0.0)], "ctx", word_ends=[0.3], pre_merge_tokens=True
            )
        self.assertEqual(seen, [])
        tts.append_to_audio_context.assert_not_awaited()


class TestCartesiaWordEnds(unittest.IsolatedAsyncioTestCase):
    """Cartesia reports each word's end; the service passes it on, aligned with
    the words that survive tag stripping."""

    async def test_ends_follow_the_normalized_words(self):
        tts = CartesiaTTSService(api_key="test", settings=CartesiaTTSService.Settings(voice="v"))
        tts.append_to_audio_context = AsyncMock()  # type: ignore[method-assign]
        tts.audio_context_available = lambda context_id: True  # type: ignore[method-assign]
        seen = _record(tts)
        message = {
            "type": "timestamps",
            "context_id": "ctx",
            "word_timestamps": {
                "words": ["Hello", '<break time="80ms"/>', "world"],
                "start": [0.0, 0.3, 0.4],
                "end": [0.3, 0.4, 0.8],
            },
        }

        async def websocket():
            yield json.dumps(message)

        tts._get_websocket = websocket  # type: ignore[method-assign]
        await tts._process_messages()
        self.assertEqual(seen, [("ctx", [("Hello", 0.0, 0.3), ("world", 0.4, 0.8)])])


if __name__ == "__main__":
    unittest.main()
