#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Azure STT service error and startup handling."""

import importlib.util
import pathlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _install_websockets_stub() -> None:
    """Install minimal websockets stubs required by STT service imports."""

    websockets_module = sys.modules.setdefault("websockets", types.ModuleType("websockets"))
    protocol_module = sys.modules.setdefault(
        "websockets.protocol", types.ModuleType("websockets.protocol")
    )
    exceptions_module = sys.modules.setdefault(
        "websockets.exceptions", types.ModuleType("websockets.exceptions")
    )

    class State:
        OPEN = "OPEN"
        CLOSED = "CLOSED"

    class ConnectionClosedError(Exception):
        pass

    class ConnectionClosedOK(Exception):
        pass

    class WebSocketClientProtocol:
        pass

    setattr(protocol_module, "State", State)
    setattr(exceptions_module, "ConnectionClosedError", ConnectionClosedError)
    setattr(exceptions_module, "ConnectionClosedOK", ConnectionClosedOK)
    setattr(websockets_module, "protocol", protocol_module)
    setattr(websockets_module, "exceptions", exceptions_module)
    setattr(websockets_module, "WebSocketClientProtocol", WebSocketClientProtocol)


def _install_azure_speech_stub() -> None:
    """Install minimal Azure Speech SDK stubs for tests.

    The Azure STT module imports Azure SDK symbols at module import time.
    These stubs allow tests to import the module without the optional dependency.
    """

    azure_module = sys.modules.setdefault("azure", types.ModuleType("azure"))
    cognitiveservices_module = sys.modules.setdefault(
        "azure.cognitiveservices", types.ModuleType("azure.cognitiveservices")
    )
    speech_module = sys.modules.setdefault(
        "azure.cognitiveservices.speech", types.ModuleType("azure.cognitiveservices.speech")
    )
    audio_module = sys.modules.setdefault(
        "azure.cognitiveservices.speech.audio", types.ModuleType("azure.cognitiveservices.speech.audio")
    )
    dialog_module = sys.modules.setdefault(
        "azure.cognitiveservices.speech.dialog", types.ModuleType("azure.cognitiveservices.speech.dialog")
    )

    class ResultReason:
        RecognizedSpeech = "RecognizedSpeech"
        RecognizingSpeech = "RecognizingSpeech"

    class SpeechConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.endpoint_id = None

    class _EventHook:
        def connect(self, _handler):
            return None

    class SpeechRecognizer:
        def __init__(self, speech_config=None, audio_config=None):
            self.speech_config = speech_config
            self.audio_config = audio_config
            self.recognizing = _EventHook()
            self.recognized = _EventHook()
            self.canceled = _EventHook()
            self.session_started = _EventHook()
            self.session_stopped = _EventHook()

        def start_continuous_recognition_async(self):
            future = MagicMock()
            future.get = MagicMock()
            return future

        def stop_continuous_recognition_async(self):
            return None

    class AudioStreamFormat:
        def __init__(self, samples_per_second=None, channels=None):
            self.samples_per_second = samples_per_second
            self.channels = channels

    class PushAudioInputStream:
        def __init__(self, _stream_format):
            pass

        def write(self, _audio):
            return None

        def close(self):
            return None

    class AudioConfig:
        def __init__(self, stream=None):
            self.stream = stream

    setattr(speech_module, "ResultReason", ResultReason)
    setattr(speech_module, "SpeechConfig", SpeechConfig)
    setattr(speech_module, "SpeechRecognizer", SpeechRecognizer)
    setattr(audio_module, "AudioStreamFormat", AudioStreamFormat)
    setattr(audio_module, "PushAudioInputStream", PushAudioInputStream)
    setattr(dialog_module, "AudioConfig", AudioConfig)

    setattr(azure_module, "cognitiveservices", cognitiveservices_module)
    setattr(cognitiveservices_module, "speech", speech_module)
    setattr(speech_module, "audio", audio_module)
    setattr(speech_module, "dialog", dialog_module)


_install_azure_speech_stub()
_install_websockets_stub()

azure_package_module = types.ModuleType("pipecat.services.azure")
azure_package_module.__path__ = []
common_module = types.ModuleType("pipecat.services.azure.common")
setattr(common_module, "language_to_azure_language", lambda _language: "en-US")

sys.modules["pipecat.services.azure"] = azure_package_module
sys.modules["pipecat.services.azure.common"] = common_module

stt_file = pathlib.Path(__file__).resolve().parents[1] / "src/pipecat/services/azure/stt.py"
spec = importlib.util.spec_from_file_location("pipecat.services.azure.stt", stt_file)
assert spec is not None and spec.loader is not None
stt_module = importlib.util.module_from_spec(spec)
sys.modules["pipecat.services.azure.stt"] = stt_module
spec.loader.exec_module(stt_module)

from pipecat.frames.frames import StartFrame


@pytest.mark.asyncio
async def test_start_connects_all_handlers_and_waits_for_future():
    """Verify startup wires event handlers and blocks for recognizer startup."""

    service = stt_module.AzureSTTService(api_key="test-key", region="eastus")

    loop = MagicMock()
    loop.run_in_executor = AsyncMock()
    service.get_event_loop = MagicMock(return_value=loop)

    recognizer = MagicMock()
    recognizer.recognizing = MagicMock()
    recognizer.recognized = MagicMock()
    recognizer.canceled = MagicMock()
    recognizer.session_started = MagicMock()
    recognizer.session_stopped = MagicMock()

    start_future = MagicMock()
    recognizer.start_continuous_recognition_async.return_value = start_future

    with patch.object(stt_module.STTService, "start", new=AsyncMock()):
        with patch.object(stt_module, "PushAudioInputStream") as mock_stream_cls:
            with patch.object(stt_module, "SpeechRecognizer", return_value=recognizer):
                await service.start(StartFrame())

    mock_stream_cls.assert_called_once()
    recognizer.recognizing.connect.assert_called_once_with(service._on_handle_recognizing)
    recognizer.recognized.connect.assert_called_once_with(service._on_handle_recognized)
    recognizer.canceled.connect.assert_called_once_with(service._on_handle_canceled)
    recognizer.session_started.connect.assert_called_once_with(service._on_handle_session_started)
    recognizer.session_stopped.connect.assert_called_once_with(service._on_handle_session_stopped)
    loop.run_in_executor.assert_awaited_once_with(None, start_future.get)


def test_canceled_handler_pushes_error_with_details():
    """Verify canceled events are surfaced through push_error."""

    service = stt_module.AzureSTTService(api_key="test-key", region="eastus")
    service.push_error = AsyncMock()
    service.get_event_loop = MagicMock(return_value=MagicMock())
    service._trace_cancellation = AsyncMock()

    canceled_event = MagicMock()
    canceled_event.cancellation_details.reason = "Error"
    canceled_event.cancellation_details.code = "AuthenticationFailure"
    canceled_event.cancellation_details.error_details = "401 Unauthorized"

    with patch("pipecat.services.azure.stt.asyncio.run_coroutine_threadsafe") as run_threadsafe:
        service._on_handle_canceled(canceled_event)

    service.push_error.assert_called_once_with(
        error_msg="Azure STT recognition canceled: Error (AuthenticationFailure)"
    )
    service._trace_cancellation.assert_called_once_with(
        reason="Error",
        code="AuthenticationFailure",
        recoverable=False,
        phase="startup",
    )
    assert run_threadsafe.call_count == 2

    for call in run_threadsafe.call_args_list:
        pending_coroutine = call.args[0]
        pending_coroutine.close()


def test_canceled_handler_uses_safe_defaults_when_details_missing():
    """Verify canceled handler does not crash on missing SDK fields."""

    service = stt_module.AzureSTTService(api_key="test-key", region="eastus")
    service.push_error = AsyncMock()
    service.get_event_loop = MagicMock(return_value=MagicMock())
    service._trace_cancellation = AsyncMock()

    canceled_event = MagicMock()
    canceled_event.cancellation_details = None

    with patch("pipecat.services.azure.stt.asyncio.run_coroutine_threadsafe") as run_threadsafe:
        service._on_handle_canceled(canceled_event)

    service.push_error.assert_called_once_with(
        error_msg="Azure STT recognition canceled: UNKNOWN (UNKNOWN)"
    )
    service._trace_cancellation.assert_called_once_with(
        reason="UNKNOWN",
        code="UNKNOWN",
        recoverable=False,
        phase="startup",
    )
    assert run_threadsafe.call_count == 2

    for call in run_threadsafe.call_args_list:
        pending_coroutine = call.args[0]
        pending_coroutine.close()


@pytest.mark.asyncio
async def test_run_stt_drops_audio_after_terminated_session():
    """Verify audio is not written after a terminated session."""

    service = stt_module.AzureSTTService(api_key="test-key", region="eastus")
    service.start_processing_metrics = AsyncMock()
    service._audio_stream = MagicMock()
    service._recognition_terminated = True

    frames = []
    async for frame in service.run_stt(b"audio"):
        frames.append(frame)

    service._audio_stream.write.assert_not_called()
    assert frames == [None]
