#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that a Pipecat server speaks RTVI as its specification says.

Each protocol version's document (``spec/<version>/rtvi.asyncapi.json``) is the
contract. The real ``RTVIProcessor`` and ``RTVIObserver`` run a session with a
client of that version, and every message they put on the wire is checked
against the document: it must be a message the server sends, its payload must
match the document's schema, and the client must be new enough to receive it.
Messages a client sends are checked the other way: the server accepts the
document's examples and answers a malformed message with an error response.
"""

import asyncio
import json
import unittest
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, PropertyMock, patch

from jsonschema import Draft202012Validator
from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerResponseFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import (
    RTVIClientMessageFrame,
    RTVIServerMessageFrame,
    RTVIServerResponseFrame,
    RTVIUICommandFrame,
    RTVIUIJobGroupFrame,
)
from pipecat.processors.frameworks.rtvi.observer import (
    RTVIFunctionCallReportLevel,
    RTVIObserverParams,
)
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.processors.frameworks.rtvi.spec import spec_path
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import TextType


def _versions() -> list[str]:
    """Every protocol version that has a document."""
    folders = (path.parent.name for path in spec_path().parent.parent.glob("*/rtvi.asyncapi.json"))
    return sorted(folders, key=_parse)


def _parse(version: str) -> list[int]:
    return [int(part) for part in version.split(".")]


class _Spec:
    """One protocol version's document, and checks of messages against it."""

    def __init__(self, version: str):
        self.version = version
        self.document = json.loads(spec_path(version).read_text(encoding="utf-8"))
        self.messages = {m["name"]: m for m in self.document["components"]["messages"].values()}
        operations = self.document["operations"]
        self.client_types = self._types(operations["receiveClientMessages"])
        self.server_types = self._types(operations["sendServerMessages"])

    def _types(self, operation: dict) -> set[str]:
        keys = [ref["$ref"].rsplit("/", 1)[1] for ref in operation["messages"]]
        return {self.document["components"]["messages"][key]["name"] for key in keys}

    def server_types_for(self, client_version: str) -> set[str]:
        """The messages the server may send to a client of ``client_version``."""
        return {
            type
            for type in self.server_types
            if _parse(self.messages[type].get("x-rtvi-min-client-version", "0.0.0"))
            <= _parse(client_version)
        }

    def violations(self, message: dict, client_version: str) -> list[str]:
        """What is wrong with a message the server sent to a client of ``client_version``."""
        type = message.get("type")
        if type not in self.server_types:
            return [f"{type!r} isn't a message the server sends in {self.version}"]
        problems = []
        if type not in self.server_types_for(client_version):
            minimum = self.messages[type]["x-rtvi-min-client-version"]
            problems.append(f"{type!r} needs a {minimum} client, sent to {client_version}")
        # The payload schemas point into the document's components.
        schema = {**self.messages[type]["payload"], "components": self.document["components"]}
        for error in Draft202012Validator(schema).iter_errors(message):
            path = "/".join(str(p) for p in error.absolute_path) or "(message)"
            problems.append(f"{type!r} at {path}: {error.message}")
        return problems


class _Session:
    """A real RTVIProcessor and RTVIObserver, with the wire messages they send."""

    def __init__(self, params: RTVIObserverParams | None = None):
        # The worker the processor would run in: bot-ready reports its
        # capabilities, and send-text flushes its pipeline.
        worker = SimpleNamespace(
            capabilities=BotCapabilities(audio_in=True, audio_out=True),
            flush_pipeline=AsyncMock(),
        )
        self._worker_patch = patch.object(
            RTVIProcessor, "pipeline_worker", new_callable=PropertyMock, return_value=worker
        )
        self._worker_patch.start()
        self.processor = RTVIProcessor()
        self.observer = self.processor.create_rtvi_observer(params=params)
        self.sent: list[dict] = []
        self.pushed: list[Any] = []
        self.output = BaseOutputTransport(TransportParams())
        self.source = FrameProcessor()

        async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, OutputTransportMessageUrgentFrame):
                self.sent.append(frame.message)
            else:
                self.pushed.append(frame)

        self.processor.push_frame = push_frame

    async def receive(self, type: str, data: Any = None, id: str = "msg_1"):
        """Handle a message from the client."""
        await self.processor._handle_message(RTVI.Message(type=type, id=id, data=data))

    async def push(self, frame, source: FrameProcessor | None = None):
        """Show the observer a frame pushed in the pipeline."""
        source = source or self.source
        await self.observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=True,
            )
        )

    def responses_to(self, id: str) -> list[dict]:
        return [m for m in self.sent if m.get("id") == id and m["type"] != "bot-ready"]

    async def cleanup(self):
        await self.observer.cleanup()
        await self.processor.cleanup()
        self._worker_patch.stop()


# Every message the observer can send, turned on.
_EVERYTHING = RTVIObserverParams(
    bot_llm_marker_enabled=True,
    bot_audio_level_enabled=True,
    user_audio_level_enabled=True,
    vad_user_speaking_enabled=True,
    system_logs_enabled=True,
    stt_raw_text_enabled=True,
    llm_raw_text_enabled=True,
    tts_raw_text_enabled=True,
    audio_level_period_secs=0,
    function_call_report_level={"*": RTVIFunctionCallReportLevel.FULL},
)


async def _run_session(client_version: str) -> list[dict]:
    """Hold a session that makes the server send every message it can.

    Returns:
        The messages the server sent, as they went on the wire.
    """
    session = _Session(_EVERYTHING)
    try:
        # Session: handshake, errors, and replies to client messages.
        await session.receive(
            "client-ready",
            {"version": client_version, "about": {"library": "conformance-test"}},
            id="ready",
        )
        await session.processor.set_bot_ready()
        await session.receive("not-a-message-type", id="unknown")
        await session.processor.send_error("The LLM service failed")
        await session.processor.send_server_message({"event": "order-updated"})
        client_message = RTVIClientMessageFrame(msg_id="cm", type="get-weather", data={})
        await session.push(RTVIServerResponseFrame(client_msg=client_message, data={"t": 21}))
        await session.push(RTVIServerResponseFrame(client_msg=client_message, error="No city"))
        await session.push(RTVIServerMessageFrame(data={"event": "order-shipped"}))
        await session.push(
            MetricsFrame(data=[TTFBMetricsData(processor="OpenAILLMService#0", value=0.4)])
        )
        logger.info("Conformance test log line")
        await logger.complete()

        # The user speaks.
        await session.push(VADUserStartedSpeakingFrame())
        await session.push(UserStartedSpeakingFrame())
        await session.push(
            InputAudioRawFrame(audio=b"\1\0" * 160, sample_rate=16000, num_channels=1)
        )
        await session.push(
            TranscriptionFrame(text="What's the weather?", user_id="user_1", timestamp="now")
        )
        await session.push(VADUserStoppedSpeakingFrame())
        await session.push(UserStoppedSpeakingFrame())
        await session.push(UserMuteStartedFrame())
        await session.push(UserMuteStoppedFrame())
        context = LLMContext(messages=[{"role": "user", "content": "What's the weather?"}])
        await session.push(LLMContextFrame(context=context))

        # The bot answers, with a function call.
        await session.push(LLMFullResponseStartFrame())
        await session.push(LLMTextFrame("It's sunny."))
        await session.push(LLMMarkerResponseFrame(raw="✓ It's sunny.", marker="✓"))
        await session.push(LLMFullResponseEndFrame())
        call = FunctionCallFromLLM(
            function_name="get_weather",
            tool_call_id="call_1",
            arguments={"city": "Lisbon"},
            context=context,
        )
        await session.push(FunctionCallsStartedFrame(function_calls=[call]))
        await session.push(
            FunctionCallInProgressFrame(
                function_name="get_weather", tool_call_id="call_1", arguments={}
            )
        )
        await session.push(
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="call_1",
                arguments={},
                result={"temperature": 21},
            )
        )
        # The deprecated way to send llm-function-call; it warns.
        with warnings.catch_warnings(record=True):
            await session.processor.handle_function_call(
                SimpleNamespace(function_name="get_weather", tool_call_id="call_1", arguments={})
            )
        await session.push(TTSStartedFrame())
        await session.push(BotStartedSpeakingFrame())
        await session.push(TTSAudioRawFrame(audio=b"\1\0" * 160, sample_rate=16000, num_channels=1))
        spoken = AggregatedTextFrame("It's sunny.", TextType.SENTENCE)
        spoken.will_be_spoken = True
        await session.push(spoken, source=session.output)
        await session.push(TTSTextFrame("It's sunny.", TextType.SENTENCE), source=session.output)
        await session.push(InterruptionFrame())
        await session.push(BotStoppedSpeakingFrame())
        await session.push(TTSStoppedFrame())

        # A UI worker drives the client's UI.
        await session.push(RTVIUICommandFrame(command="toast", payload={"title": "Shipped"}))
        await session.push(RTVIUIJobGroupFrame(data=RTVI.UIJobGroupStartedData(job_id="job_1")))

        # Let the system log sink deliver what it queued.
        await asyncio.sleep(0)
        return session.sent
    finally:
        await session.cleanup()


class TestRTVIServerSends(unittest.IsolatedAsyncioTestCase):
    """What the server sends obeys each protocol version's document."""

    async def test_every_message_obeys_its_version(self):
        for version in _versions():
            with self.subTest(version=version):
                spec = _Spec(version)
                sent = await _run_session(version)
                problems = [p for m in sent for p in spec.violations(m, version)]
                self.assertEqual(problems, [])

    async def test_the_session_sends_every_message_of_the_current_version(self):
        # A message the spec lists but this session never sends goes unchecked.
        # Add what makes the server send it to _run_session.
        spec = _Spec(RTVI.PROTOCOL_VERSION)
        sent = {m["type"] for m in await _run_session(RTVI.PROTOCOL_VERSION)}
        self.assertEqual(spec.server_types - sent, set())

    async def test_older_clients_only_get_the_messages_for_their_version(self):
        spec = _Spec(RTVI.PROTOCOL_VERSION)
        minimums = {
            m["x-rtvi-min-client-version"]
            for m in spec.messages.values()
            if "x-rtvi-min-client-version" in m
        }
        for minimum in sorted(minimums, key=_parse):
            # The newest client that is still too old for these messages.
            major, minor, _ = _parse(minimum)
            older = f"{major}.{minor - 1}.0" if minor else f"{major - 1}.0.0"
            with self.subTest(client_version=older):
                sent = await _run_session(older)
                problems = [p for m in sent for p in spec.violations(m, older)]
                self.assertEqual(problems, [])
                self.assertEqual(
                    {m["type"] for m in sent}, spec.server_types_for(older) & spec.server_types
                )


class TestRTVIServerReceives(unittest.IsolatedAsyncioTestCase):
    """What the server accepts obeys each protocol version's document."""

    async def _receive(self, version: str, type: str, data: Any) -> _Session:
        session = _Session()
        self.addAsyncCleanup(session.cleanup)
        await session.receive(type, data, id="under-test")
        spec = _Spec(version)
        problems = [p for m in session.sent for p in spec.violations(m, version)]
        self.assertEqual(problems, [])
        return session

    async def test_examples_are_accepted(self):
        for version in _versions():
            spec = _Spec(version)
            for type in sorted(spec.client_types):
                example = spec.messages[type]["examples"][0]["payload"]
                with self.subTest(version=version, type=type):
                    session = await self._receive(version, type, example.get("data"))
                    self.assertEqual(session.responses_to("under-test"), [])

    async def test_missing_data_gets_an_error_response(self):
        for version in _versions():
            spec = _Spec(version)
            for type in sorted(spec.client_types):
                if "data" not in spec.messages[type]["payload"]["required"]:
                    continue
                with self.subTest(version=version, type=type):
                    session = await self._receive(version, type, None)
                    responses = session.responses_to("under-test")
                    self.assertEqual([m["type"] for m in responses], ["error-response"])

    async def test_unknown_type_gets_an_error_response(self):
        session = await self._receive(RTVI.PROTOCOL_VERSION, "not-a-message-type", None)
        responses = session.responses_to("under-test")
        self.assertEqual([m["type"] for m in responses], ["error-response"])


if __name__ == "__main__":
    unittest.main()
