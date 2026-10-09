#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Azure STT service init parameters and recognized handler."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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


class TestAzureSTTModelSetting(unittest.TestCase):
    """The ``model`` setting sets ``SpeechConfig.model``, which the Speech SDK
    sends to Azure to select a recognition model such as MAI-Transcribe-2-Streaming."""

    def test_model_set_on_init(self):
        service = AzureSTTService(
            api_key="fake",
            region="eastus",
            settings=AzureSTTService.Settings(model="mai-transcribe-2-streaming"),
        )
        self.assertEqual(service._speech_config.model, "mai-transcribe-2-streaming")
        self.assertEqual(
            service._speech_config.get_property_by_name("SPEECH-ModelName"),
            "mai-transcribe-2-streaming",
        )

    def test_model_none_leaves_default(self):
        service = AzureSTTService(
            api_key="fake",
            region="eastus",
            settings=AzureSTTService.Settings(model=None),
        )
        self.assertIsNone(service._settings.model)
        self.assertEqual(service._speech_config.get_property_by_name("SPEECH-ModelName"), "")


class TestAzureSTTModelUpdate(unittest.IsolatedAsyncioTestCase):
    """Live updates to ``model`` should apply the setting and reconnect when a stream is active."""

    async def test_model_change_triggers_reconnect_when_stream_active(self):
        service = AzureSTTService(api_key="fake", region="eastus")
        # Simulate an active audio stream so the service will reconnect.
        service._audio_stream = object()

        with (
            patch.object(service, "_disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(service, "_connect", new_callable=AsyncMock) as mock_connect,
            patch.object(service, "_apply_model") as mock_apply,
        ):
            await service._update_settings(AzureSTTService.Settings(model="new-model"))
            mock_apply.assert_called_once()
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    async def test_model_change_does_not_reconnect_when_stream_inactive(self):
        service = AzureSTTService(api_key="fake", region="eastus")
        # No active stream.
        service._audio_stream = None

        with (
            patch.object(service, "_disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(service, "_connect", new_callable=AsyncMock) as mock_connect,
            patch.object(service, "_apply_model") as mock_apply,
        ):
            await service._update_settings(AzureSTTService.Settings(model="new-model"))
            mock_apply.assert_called_once()
            mock_disconnect.assert_not_called()
            mock_connect.assert_not_called()

    async def test_model_change_updates_speech_config(self):
        service = AzureSTTService(
            api_key="fake",
            region="eastus",
            settings=AzureSTTService.Settings(model="mai-transcribe-2-streaming"),
        )

        await service._update_settings(AzureSTTService.Settings(model="other-model"))
        self.assertEqual(service._speech_config.model, "other-model")

        await service._update_settings(AzureSTTService.Settings(model=None))
        self.assertEqual(service._speech_config.get_property_by_name("SPEECH-ModelName"), "")


class TestAzureSTTSegmentationSilenceTimeout(unittest.TestCase):
    """The ``segmentation_silence_timeout_ms`` setting surfaces Azure's
    ``Speech_SegmentationSilenceTimeoutMs`` property, which decides how long a
    pause has to be before Azure finalizes a phrase."""

    def _segmentation_calls(self, mock_set):
        from azure.cognitiveservices.speech import PropertyId

        # SpeechConfig itself sets the recognition language through
        # ``set_property``, so look only at the segmentation property.
        return [
            call
            for call in mock_set.call_args_list
            if call.args[0] == PropertyId.Speech_SegmentationSilenceTimeoutMs
        ]

    def test_timeout_sets_speech_config_property(self):
        with patch("pipecat.services.azure.stt.SpeechConfig.set_property") as mock_set:
            AzureSTTService(
                api_key="fake",
                region="eastus",
                settings=AzureSTTService.Settings(segmentation_silence_timeout_ms=1200),
            )
            calls = self._segmentation_calls(mock_set)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].args[1], "1200")

    def test_timeout_omitted_is_no_op(self):
        with patch("pipecat.services.azure.stt.SpeechConfig.set_property") as mock_set:
            AzureSTTService(api_key="fake", region="eastus")
            self.assertEqual(self._segmentation_calls(mock_set), [])


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
