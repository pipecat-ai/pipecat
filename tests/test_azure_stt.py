#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Azure STT service init parameters and recognized handler."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.azure.stt import AzureSTTService


class TestAzureSTTProfanitySetting(unittest.TestCase):
    """The ``profanity`` setting surfaces ``SpeechConfig.set_profanity`` as a
    runtime-updatable ``AzureSTTSettings`` field so callers don't have to reach
    into the private ``_speech_config`` to disable Azure's profanity masking."""

    def _service(self, profanity):
        return AzureSTTService(
            api_key="fake",
            region="eastus",
            settings=AzureSTTService.Settings(profanity=profanity),
        )

    def test_profanity_omitted_is_no_op(self):
        # Omitted = keep Azure SDK default (Masked). Constructor should
        # not crash and ``_speech_config`` should still be set up.
        service = AzureSTTService(api_key="fake", region="eastus")
        self.assertIsNotNone(service._speech_config)

    def test_profanity_raw_accepted(self):
        self.assertIsNotNone(self._service("raw")._speech_config)

    def test_profanity_masked_accepted(self):
        self.assertIsNotNone(self._service("masked")._speech_config)

    def test_profanity_removed_accepted(self):
        self.assertIsNotNone(self._service("removed")._speech_config)

    def test_profanity_invalid_value_rejected(self):
        # Out-of-range value fails fast at init instead of silently
        # falling back to the SDK default.
        with self.assertRaises(KeyError):
            self._service("nope")  # type: ignore[arg-type]

    def test_profanity_calls_set_profanity_on_speech_config(self):
        # Spy on SpeechConfig.set_profanity to confirm the setting is wired
        # through. We can't easily read back the value (the SDK exposes
        # a setter but no public getter), so we patch and assert.
        with patch("pipecat.services.azure.stt.SpeechConfig.set_profanity") as mock_set:
            self._service("raw")
            mock_set.assert_called_once()

    def test_profanity_not_called_when_omitted(self):
        with patch("pipecat.services.azure.stt.SpeechConfig.set_profanity") as mock_set:
            AzureSTTService(api_key="fake", region="eastus")
            mock_set.assert_not_called()


class TestAzureSTTFinalizedFlag(unittest.IsolatedAsyncioTestCase):
    """Azure's ``RecognizedSpeech`` event is the final recognition for an
    utterance — the emitted ``TranscriptionFrame`` must carry
    ``finalized=True`` so downstream user-turn stop strategies can take
    their fast-path."""

    async def test_recognized_speech_emits_finalized_transcription_frame(self):
        # Intercept the ``TranscriptionFrame`` constructor to capture the
        # exact kwargs ``_on_handle_recognized`` uses. This is the most
        # direct way to pin the ``finalized=True`` invariant without
        # bringing up the full STT service lifecycle (TaskManager, event
        # loop registration) which would require a real pipeline.
        service = AzureSTTService(api_key="fake", region="eastus")

        async def noop(*_args, **_kwargs):
            pass

        service._handle_transcription = noop  # type: ignore[method-assign]
        # Short-circuit run_coroutine_threadsafe: we don't need the
        # coroutines to actually execute — only that the frame was
        # constructed with the right flag.
        fake_loop = SimpleNamespace()
        service.get_event_loop = lambda: fake_loop  # type: ignore[method-assign]

        def fake_run_threadsafe(coro, _loop):
            coro.close()  # Avoid "coroutine was never awaited" warnings.

            class _Dummy:
                def result(self, timeout=None):
                    return None

            return _Dummy()

        from azure.cognitiveservices.speech import ResultReason

        event = SimpleNamespace(
            result=SimpleNamespace(
                reason=ResultReason.RecognizedSpeech,
                text="hello world",
                language=None,
            )
        )

        constructed: list[dict] = []
        real_init = TranscriptionFrame.__init__

        def spy_init(self, *args, **kwargs):
            constructed.append({"args": args, "kwargs": dict(kwargs)})
            real_init(self, *args, **kwargs)

        with (
            patch(
                "pipecat.services.azure.stt.asyncio.run_coroutine_threadsafe",
                side_effect=fake_run_threadsafe,
            ),
            patch.object(TranscriptionFrame, "__init__", spy_init),
        ):
            service._on_handle_recognized(event)

        self.assertEqual(len(constructed), 1)
        # ``finalized`` is a kwarg of TranscriptionFrame; the recognized
        # path must pass it as True.
        self.assertTrue(
            constructed[0]["kwargs"].get("finalized"),
            f"finalized=True kwarg missing, got kwargs: {constructed[0]['kwargs']}",
        )


if __name__ == "__main__":
    unittest.main()
