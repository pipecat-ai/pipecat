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
document's examples. Each of the document's ``x-rtvi-behaviors`` has a test
here of its own.
"""

import asyncio
import dataclasses
import json
import unittest
import warnings
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, PropertyMock, patch

from jsonschema import Draft7Validator
from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
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
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
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
        for error in Draft7Validator(schema).iter_errors(message):
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


async def _run_session(client_version: str, params: RTVIObserverParams = _EVERYTHING) -> list[dict]:
    """Hold a session that makes the server send every message it can.

    Args:
        client_version: The protocol version the client speaks.
        params: Which of the observer's messages are turned on.

    Returns:
        The messages the server sent, as they went on the wire.
    """
    session = _Session(params)
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
        # Every kind of metric, so the document's metrics schema is checked whole.
        await session.push(
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor="OpenAILLMService#0", value=0.4),
                    TTFAMetricsData(
                        processor="CartesiaTTSService#0", ttfa=0.6, ttfb=0.2, leading_silence=0.4
                    ),
                    TTFATMetricsData(
                        processor="OpenAILLMService#0", ttfat=0.9, ttfb=0.4, thinking_time=0.5
                    ),
                    ProcessingMetricsData(processor="OpenAILLMService#0", value=0.05),
                    LLMUsageMetricsData(
                        processor="OpenAILLMService#0",
                        value=LLMTokenUsage(
                            prompt_tokens=10, completion_tokens=20, total_tokens=30
                        ),
                    ),
                    STTUsageMetricsData(
                        processor="DeepgramSTTService#0", value=STTUsage(audio_seconds=1.5)
                    ),
                    TTSUsageMetricsData(processor="CartesiaTTSService#0", value=42),
                ]
            )
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
        await session.push(RTVIUICommandFrame(command="open-cart", payload={"items": 2}))
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


class TestRTVIBehaviors(unittest.IsolatedAsyncioTestCase):
    """The server follows each of the document's behaviors.

    A behavior's test is named after its id: ``test_<id>``, with underscores.
    """

    async def _session(self, params: RTVIObserverParams | None = None) -> _Session:
        session = _Session(params)
        self.addAsyncCleanup(session.cleanup)
        return session

    async def _ready(self, session: _Session, version: Any = RTVI.PROTOCOL_VERSION):
        data = {} if version is None else {"version": version}
        await session.receive("client-ready", data, id="ready")
        await session.processor.set_bot_ready()

    def test_every_behavior_has_a_test(self):
        for version in _versions():
            for behavior in _Spec(version).document["x-rtvi-behaviors"]:
                with self.subTest(version=version, behavior=behavior["id"]):
                    self.assertTrue(hasattr(self, "test_" + behavior["id"].replace("-", "_")))

    async def test_bot_ready_answers_client_ready(self):
        session = await self._session()
        await self._ready(session)
        bot_ready = [m for m in session.sent if m["type"] == "bot-ready"]
        self.assertEqual([m["id"] for m in bot_ready], ["ready"])

    async def test_bot_ready_version(self):
        for client, expected in (
            (RTVI.PROTOCOL_VERSION, RTVI.PROTOCOL_VERSION),
            (RTVI.PROTOCOL_VERSION.split(".")[0] + ".0.0", RTVI.PROTOCOL_VERSION),
            ("1.2.0", "1.2.0"),
        ):
            with self.subTest(client=client):
                session = await self._session()
                await self._ready(session, client)
                (bot_ready,) = [m for m in session.sent if m["type"] == "bot-ready"]
                self.assertEqual(bot_ready["data"]["version"], expected)

    async def test_client_ready_version_check(self):
        major = int(RTVI.PROTOCOL_VERSION.split(".")[0])
        for version in (None, "2.x", "2.3", f"{major + 1}.0.0"):
            with self.subTest(version=version):
                session = await self._session()
                await self._ready(session, version)
                self.assertEqual(
                    [m["type"] for m in session.sent if m.get("id") == "ready"],
                    ["error-response", "bot-ready"],
                )
        for version in (RTVI.PROTOCOL_VERSION, "1.0.0"):
            with self.subTest(version=version):
                session = await self._session()
                await self._ready(session, version)
                self.assertEqual(session.responses_to("ready"), [])

    async def test_failed_request_gets_error_response(self):
        spec = _Spec(RTVI.PROTOCOL_VERSION)
        malformed = [
            (type, None)
            for type in sorted(spec.client_types)
            if "data" in spec.messages[type]["payload"]["required"]
        ]
        for type, data in [("not-a-message-type", None), *malformed]:
            with self.subTest(type=type):
                session = await self._session()
                await session.receive(type, data, id="under-test")
                responses = session.responses_to("under-test")
                self.assertEqual([m["type"] for m in responses], ["error-response"])

    async def test_invalid_message_gets_error(self):
        session = await self._session()
        no_id = {"label": RTVI.MESSAGE_LABEL, "type": "send-text", "data": {"content": "Hi"}}
        await session.processor._handle_transport_message(InputTransportMessageFrame(no_id))
        self.assertEqual([m["type"] for m in session.sent], ["error"])

        session = await self._session()
        not_rtvi = {"label": "other", "type": "send-text", "id": "1"}
        await session.processor._handle_transport_message(InputTransportMessageFrame(not_rtvi))
        self.assertEqual(session.sent, [])

    async def test_client_message_answer(self):
        session = await self._session()
        await session.receive("client-message", {"t": "get-weather", "d": {}}, id="cm")
        (request,) = [f for f in session.pushed if isinstance(f, RTVIClientMessageFrame)]
        await session.push(RTVIServerResponseFrame(client_msg=request, data={"temperature": 21}))
        await session.push(RTVIServerResponseFrame(client_msg=request, error="No city"))
        client_message = RTVI.ClientMessage(msg_id="cm", type="get-weather", data={})
        await session.processor.send_server_response(client_message, {"temperature": 21})
        await session.processor.send_error_response(client_message, "No city")
        answers = session.responses_to("cm")
        self.assertEqual(
            [m["type"] for m in answers],
            ["server-response", "error-response", "server-response", "error-response"],
        )
        for answer in answers:
            if answer["type"] == "server-response":
                self.assertEqual(answer["data"]["t"], "get-weather")

    async def test_disconnect_bot_ends_session(self):
        session = await self._session()
        await session.receive("disconnect-bot")
        self.assertTrue(any(isinstance(f, EndWorkerFrame) for f in session.pushed))

    async def test_messages_turned_off(self):
        spec = _Spec(RTVI.PROTOCOL_VERSION)
        # The param that turns each message on.
        switches = {
            m["name"]: m["x-rtvi-enabled-by"]
            for m in spec.messages.values()
            if "x-rtvi-enabled-by" in m
        }
        for field in sorted(set(switches.values())):
            with self.subTest(off=field):
                params = dataclasses.replace(_EVERYTHING, **{field: False})
                sent = {m["type"] for m in await _run_session(RTVI.PROTOCOL_VERSION, params)}
                turned_off = {type for type, f in switches.items() if f == field}
                self.assertEqual(sent & turned_off, set())
                # Turning one off leaves the rest on.
                self.assertTrue(set(switches) - turned_off <= sent)

    def _spoken(self, text: str) -> AggregatedTextFrame:
        frame = AggregatedTextFrame(text, TextType.SENTENCE)
        frame.will_be_spoken = True
        return frame

    async def test_spoken_output_waits_for_speech(self):
        session = await self._session()
        await self._ready(session)
        await session.push(self._spoken("It's sunny."), source=session.output)
        await session.push(self._spoken("In Lisbon."), source=session.output)
        self.assertNotIn("bot-output", [m["type"] for m in session.sent])
        await session.push(BotStartedSpeakingFrame())
        after = [m for m in session.sent if m["type"] in ("bot-started-speaking", "bot-output")]
        self.assertEqual(
            [(m["type"], m.get("data", {}).get("text")) for m in after],
            [
                ("bot-started-speaking", None),
                ("bot-output", "It's sunny."),
                ("bot-output", "In Lisbon."),
            ],
        )

    async def test_interruption_drops_unspoken_output(self):
        session = await self._session()
        await self._ready(session)
        await session.push(self._spoken("It's sunny."), source=session.output)
        await session.push(InterruptionFrame())
        await session.push(BotStartedSpeakingFrame())
        types = [m["type"] for m in session.sent]
        self.assertIn("bot-interrupted", types)
        self.assertNotIn("bot-output", types)


if __name__ == "__main__":
    unittest.main()
