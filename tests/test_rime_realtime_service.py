#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism frame contracts through the real SDK and a local protocol peer."""

import asyncio

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import (
    AudioPlaybackReportFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMInputTextFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMOperationResultFrame,
    LLMRunFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService

from .rime_realtime_peer import conversation, until

pytestmark = pytest.mark.asyncio


def results(frames, request):
    return [
        frame
        for frame in frames
        if isinstance(frame, LLMOperationResultFrame) and frame.request_id == request.id
    ]


async def test_typed_history_audio_and_second_turn():
    context = LLMContext(
        [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior reply"},
        ]
    )
    async with conversation(context=context) as (peer, service, worker, context, output, up, down):
        assert [item["text"] for item in peer.history] == ["Prior question", "Prior reply"]
        assert not [e for e in peer.events if e["type"] == "response.create"]
        typed = LLMInputTextFrame("A typed question", client_message_id="client-1")
        await worker.queue_frame(typed)
        request = await peer.next()
        assert request["response"]["prism_input_text"] == typed.text
        ref = await peer.start(request)
        await peer.message(ref, samples=241)
        await peer.end(ref)
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["played_ms"] == pytest.approx(241 / 24, abs=0.1)
        await until(lambda: results(up, typed))
        assert [f.outcome for f in results(up, typed)] == ["accepted"]
        assert context.messages == [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior reply"},
            {"role": "user", "content": "A typed question"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert any(isinstance(frame, LLMTextFrame) for frame in down)
        from pipecat.frames.frames import LLMFullResponseStartFrame

        started = next(frame for frame in down if isinstance(frame, LLMFullResponseStartFrame))
        assert started.metadata["rime"]["client_message_id"] == "client-1"
        assert started.metadata["rime"]["input_request_id"] == typed.id
        assert output.writes[0].context_id
        await worker.queue_frame(InputAudioRawFrame(b"\0\0" * 160, 16000, 1))
        await peer.next("input_audio_buffer.append")
        await worker.queue_frame(LLMInputTextFrame("Next"))
        request = await peer.next()
        ref = await peer.start(request)
        await peer.end(ref)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == 0


@pytest.mark.parametrize("measured,enabled,expected", [(False, True, None), (True, False, 0)])
async def test_unknown_and_disabled_playback(measured, enabled, expected):
    async with conversation(measured=measured, enabled=enabled) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await peer.message(ref)
        await peer.end(ref)
        assert (await asyncio.wait_for(peer.receipts.get(), 3))["played_ms"] == expected


async def test_settings_acceptance_and_rejection_preserve_state():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        peer.gates["session.update"] = asyncio.Event()
        update = LLMUpdateSettingsFrame(
            delta=service.Settings(system_instruction="Be brief", truncation="disabled")
        )
        await worker.queue_frame(update)
        await until(lambda: len([e for e in peer.events if e["type"] == "session.update"]) == 2)
        assert service.settings.system_instruction is None
        assert not results(up, update)
        peer.gates["session.update"].set()
        await until(lambda: results(up, update))
        assert service.settings.system_instruction == "Be brief"
        assert service.settings.truncation == "disabled"
        assert peer.settings["instructions"] == "Be brief"
        assert service.session_info.request_id == "actual-peer-request"
        peer.refuse["session.update"] = "invalid_request"
        rejected = LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="Rejected"))
        await worker.queue_frame(rejected)
        await until(lambda: results(up, rejected))
        assert results(up, rejected)[0].outcome == "rejected"
        assert service.settings.system_instruction == "Be brief"
        assert service.is_usable


async def test_history_import_never_generates_a_turn_and_run_is_separate():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        append = LLMMessagesAppendFrame(messages=[{"role": "user", "content": "History only"}])
        await worker.queue_frame(append)
        await until(lambda: results(up, append))
        assert ctx.messages == [{"role": "user", "content": "History only"}]
        assert not any(e["type"] == "response.create" for e in peer.events)
        await worker.queue_frame(LLMRunFrame())
        request = await peer.next()
        assert "prism_input_text" not in request["response"]
        ref = await peer.start(request)
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)


async def test_history_validation_precedes_any_mutation():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        request = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": "Valid"}, {"role": "user", "content": "x" * 4001}]
        )
        await worker.queue_frame(request)
        await until(lambda: results(up, request))
        assert results(up, request)[0].outcome == "rejected"
        assert ctx.messages == []
        assert peer.history == []
        assert service.is_usable


async def test_late_transcript_keeps_input_before_reply():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await peer.emit("input_audio_buffer.speech_started", item_id="speech-1", audio_start_ms=0)
        await peer.emit("input_audio_buffer.committed", item_id="speech-1")
        ref = await peer.start(input_ids=["speech-1"])
        await peer.message(ref)
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)
        await peer.emit(
            "conversation.item.input_audio_transcription.completed",
            item_id="speech-1",
            transcript="Late text",
        )
        await until(lambda: ctx.messages and ctx.messages[0].get("content") == "Late text")
        assert ctx.messages == [
            {"role": "user", "content": "Late text"},
            {"role": "assistant", "content": "Hello"},
        ]


async def test_combined_tool_settings_fillers_and_metadata():
    from rimelabs_sdk.realtime import PrismFiller, RetentionRatio

    async def lookup_invoice(params, invoice: str):
        """Look up an invoice.

        Args:
            invoice: Invoice number.
        """
        await params.result_callback({"invoice": invoice})

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        update = LLMUpdateSettingsFrame(
            delta=service.Settings(
                system_instruction="Handle invoices",
                tools=[lookup_invoice],
                tool_fillers={
                    "lookup_invoice": PrismFiller(
                        type="auto", instructions="Say that you are checking."
                    )
                },
                truncation=RetentionRatio(0.75),
                response_metadata={"step": "billing"},
            )
        )
        await worker.queue_frame(update)
        await until(lambda: len(results(up, update)) == 2)
        assert {f.operation for f in results(up, update)} == {"settings", "response_metadata"}
        assert all(f.outcome == "accepted" for f in results(up, update))
        updates = [e["session"] for e in peer.events if e["type"] == "session.update"]
        assert len(updates) == 3
        assert {"instructions", "tools", "truncation"} <= updates[1].keys()
        assert updates[1]["tools"][0]["prism_filler"]["type"] == "auto"
        assert updates[2] == {"prism_response_metadata": {"step": "billing"}}
        assert service.has_function("lookup_invoice")
        rejected = LLMUpdateSettingsFrame(
            delta=service.Settings(tool_fillers={"absent": PrismFiller(type="none")})
        )
        await worker.queue_frame(rejected)
        await until(lambda: results(up, rejected))
        assert results(up, rejected)[0].outcome == "rejected"
        from pipecat.frames.frames import LLMSetToolsFrame

        clear = LLMSetToolsFrame(tools=[])
        await worker.queue_frame(clear)
        await until(lambda: results(up, clear))
        assert service.settings.tool_fillers == {}
        assert not service.has_function("lookup_invoice")
        assert not ctx.tools


async def test_tool_executes_once_with_response_handler_after_tool_removal():
    from pipecat.services.llm_service import FunctionCallParams

    calls = []

    async def lookup(params: FunctionCallParams, invoice: str):
        """Get an invoice.

        Args:
            invoice: Invoice number.
        """
        calls.append(invoice)
        await params.result_callback({"status": "paid"})

    async with conversation(context=LLMContext(tools=[lookup])) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        await worker.queue_frame(LLMRunFrame())
        parent = await peer.start(await peer.next())
        update = LLMUpdateSettingsFrame(delta=service.Settings(tools=[]))
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        assert not service.has_function("lookup")
        body = dict(
            response_id=parent,
            item_id="call-item",
            call_id="call-1",
            name="lookup",
            arguments='{"invoice":"123"}',
        )
        await peer.emit("response.function_call_arguments.done", **body)
        await peer.end(parent)
        continuation = await peer.next()
        assert calls == ["123"]
        assert continuation["response"]["metadata"]["prism_cause"] == "tool_continuation"
        sent = [e for e in peer.events if e["type"] == "conversation.item.create"]
        assert len(sent) == 1
        assert sent[0]["item"]["call_id"] == "call-1"
        assert sent[0]["item"]["output"] == '{"status": "paid"}'
        ref = await peer.start(continuation)
        await peer.end(ref)
        await until(lambda: len([m for m in ctx.messages if m.get("role") == "tool"]) == 1)
        assert calls == ["123"]


async def test_audio_and_event_processing_continue_during_settings_acknowledgment():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        gate = peer.gates["session.update"] = asyncio.Event()
        update = LLMUpdateSettingsFrame(delta=service.Settings(voice="voice-two"))
        await worker.queue_frame(update)
        await until(lambda: len([e for e in peer.events if e["type"] == "session.update"]) == 2)
        await worker.queue_frame(InputAudioRawFrame(b"\0\0" * 320, 16000, 1))
        await peer.next("input_audio_buffer.append")
        ref = await peer.start()
        await peer.message(ref)
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)
        assert not results(up, update)
        gate.set()
        await until(lambda: results(up, update))
        assert results(up, update)[0].outcome == "accepted"


async def test_rebuild_preserves_arrivals_and_partial_failure_requires_recovery():

    async with conversation(context=LLMContext([{"role": "user", "content": "Old"}])) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        # Reject a rebuild after one deletion has already committed remotely.
        peer.history_failure_at = peer.history_mutations + 2
        update = LLMMessagesUpdateFrame(messages=[{"role": "user", "content": "Replacement"}])
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        assert results(up, update)[0].outcome == "partial"
        assert not service.is_usable
        assert ctx.messages == [{"role": "user", "content": "Old"}]


async def test_repeated_history_records_keep_identity_for_explicit_delete():
    from pipecat.frames.frames import LLMMessagesTransformFrame

    ctx = LLMContext([{"role": "user", "content": "Same"}, {"role": "user", "content": "Same"}])
    async with conversation(context=ctx) as (peer, service, worker, ctx, output, up, down):
        retained_id = peer.history[1]["id"]
        update = LLMMessagesTransformFrame(transform=lambda messages: messages[1:])
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        assert results(up, update)[0].outcome == "accepted"
        assert [item["id"] for item in peer.history] == [retained_id]
        assert len(ctx.messages) == 1
        assert len([e for e in peer.events if e["type"] == "conversation.item.create"]) == 2


async def test_unsafe_rebuild_during_active_response_is_rejected():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMInputTextFrame("Live"))
        ref = await peer.start(await peer.next())
        await until(lambda: bool(ctx.messages))
        update = LLMMessagesUpdateFrame(messages=[])
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        assert results(up, update)[0].outcome == "rejected"
        assert service.is_usable
        assert ctx.messages[0]["content"] == "Live"
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)


async def test_output_ids_multiple_messages_and_duplicate_reports():
    from pipecat.processors.frame_processor import FrameDirection

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await peer.message(ref, item_id="first", samples=241)
        await peer.message(ref, item_id="second", samples=240)
        await peer.end(ref)
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["played_ms"] == pytest.approx(481 / 24, abs=0.1)
        reports = [f for f in up if isinstance(f, AudioPlaybackReportFrame)]
        assert len(reports) == 2
        assert reports[0].context_id != reports[1].context_id
        assert {f.context_id for f in output.writes} == {f.context_id for f in reports}
        await worker.queue_frame(reports[0], FrameDirection.UPSTREAM)
        await asyncio.sleep(0.02)
        assert peer.receipts.empty()


@pytest.mark.parametrize("cause,truncate", [("user", True), ("application", False)])
async def test_interruption_reports_cause_and_rejects_late_audio(cause, truncate):
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        common = await peer.message(ref, done=False, samples=1920)
        await until(lambda: bool(output.writes))
        await worker.queue_frame(InterruptionFrame(cause=cause))
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        count = len(output.writes)
        await peer.emit("response.audio.delta", **common, delta="EAAQABAAEAA=")
        await asyncio.sleep(0.02)
        assert len(output.writes) == count
        assert (
            bool([e for e in peer.events if e["type"] == "conversation.item.truncate"]) == truncate
        )
        assert receipt["played_ms"] >= 0
        await worker.queue_frame(LLMInputTextFrame("After interruption"))
        ref = await peer.start(await peer.next())
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)


async def test_append_wait_preserves_a_late_speech_record():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        gate = peer.gates["conversation.item.create"] = asyncio.Event()
        append = LLMMessagesAppendFrame(messages=[{"role": "system", "content": "A note"}])
        await worker.queue_frame(append)
        await until(lambda: any(e["type"] == "conversation.item.create" for e in peer.events))
        assert ctx.messages == []
        await peer.emit(
            "input_audio_buffer.speech_started", item_id="speech-late", audio_start_ms=0
        )
        await peer.emit(
            "conversation.item.input_audio_transcription.completed",
            item_id="speech-late",
            transcript="New speech",
        )
        await until(lambda: bool(ctx.messages))
        gate.set()
        await until(lambda: results(up, append))
        assert ctx.messages == [
            {"role": "user", "content": "New speech"},
            {"role": "system", "content": "A note"},
        ]


async def test_missing_acknowledgment_is_unknown_and_never_replayed():
    from rimelabs_sdk.realtime import RealtimeTimeouts

    async with conversation(timeouts=RealtimeTimeouts(request_s=0.08)) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        gate = peer.gates["session.update"] = asyncio.Event()
        update = LLMUpdateSettingsFrame(delta=service.Settings(voice="unknown-voice"))
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        assert results(up, update)[0].outcome == "unknown"
        assert not service.is_usable
        assert service.settings.voice is None
        assert len([e for e in peer.events if e["type"] == "session.update"]) == 2
        gate.set()


async def test_unsent_typed_admission_can_be_retried_with_a_new_request():
    from rimelabs_sdk.realtime import RealtimeTimeouts

    async with conversation(timeouts=RealtimeTimeouts(ready_s=0.08)) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        blocked = LLMInputTextFrame("Wait for admission")
        await worker.queue_frame(blocked)
        await until(lambda: results(up, blocked))
        assert results(up, blocked)[0].outcome == "rejected"
        assert service.is_usable
        assert len([e for e in peer.events if e["type"] == "response.create"]) == 1
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)
        retry = LLMInputTextFrame(blocked.text)
        await worker.queue_frame(retry)
        ref = await peer.start(await peer.next())
        await peer.end(ref)
        await until(lambda: results(up, retry))
        assert results(up, retry)[0].outcome == "accepted"
        assert [m for m in ctx.messages if m.get("role") == "user"] == [
            {"role": "user", "content": blocked.text}
        ]


@pytest.mark.parametrize(
    "rate,channels", [(8000, 1), (16000, 2), (24000, 1), (48000, 2), (44100, 2), (32000, 3)]
)
async def test_microphone_pcm_formats(rate, channels):
    import base64

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(
            InputAudioRawFrame(b"\x01\x00" * (rate * channels // 10), rate, channels)
        )
        wire = await peer.next("input_audio_buffer.append")
        pcm = base64.b64decode(wire["audio"])
        assert pcm and len(pcm) % 2 == 0
        assert service.is_usable


async def test_external_context_snapshot_rebuilds_before_reply():
    from pipecat.frames.frames import LLMContextFrame

    async with conversation(context=LLMContext([{"role": "user", "content": "Old"}])) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        ctx.set_messages([{"role": "user", "content": "External replacement"}])
        frame = LLMContextFrame(ctx)
        await worker.queue_frame(frame)
        request = await peer.next()
        assert "prism_input_text" not in request["response"]
        assert [item["text"] for item in peer.history] == ["External replacement"]
        ref = await peer.start(request)
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)
        assert ctx.messages == [{"role": "user", "content": "External replacement"}]


async def test_queued_replacements_apply_in_request_order():
    async with conversation(context=LLMContext([{"role": "user", "content": "Old"}])) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        gate = peer.gates["conversation.item.delete"] = asyncio.Event()
        first = LLMMessagesUpdateFrame(messages=[{"role": "user", "content": "First"}])
        second = LLMMessagesUpdateFrame(messages=[{"role": "user", "content": "Second"}])
        await worker.queue_frame(first)
        await worker.queue_frame(second)
        await until(lambda: any(e["type"] == "conversation.item.delete" for e in peer.events))
        gate.set()
        await until(lambda: results(up, second))
        assert results(up, first)[0].outcome == results(up, second)[0].outcome == "accepted"
        assert ctx.messages == [{"role": "user", "content": "Second"}]
        assert [item["text"] for item in peer.history] == ["Second"]


async def test_cancel_during_update_retains_its_acknowledged_outcome():
    from pipecat.frames.frames import CancelFrame

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        gate = peer.gates["session.update"] = asyncio.Event()
        update = LLMUpdateSettingsFrame(
            delta=service.Settings(system_instruction="Accepted during cancel")
        )
        outcomes = []

        @service.event_handler("on_operation_result")
        async def on_result(service, frame):
            if frame.request_id == update.id:
                outcomes.append(frame.outcome)

        await worker.queue_frame(update)
        await until(lambda: len([e for e in peer.events if e["type"] == "session.update"]) == 2)
        await worker.queue_frame(CancelFrame())
        await asyncio.sleep(0.02)
        gate.set()
        await until(lambda: bool(outcomes))
        assert outcomes == ["accepted"]
        assert service.settings.system_instruction == "Accepted during cancel"


async def test_failed_startup_closes_session_but_keeps_injected_client_usable():
    from rimelabs_sdk import Rime
    from websockets.asyncio.server import serve

    from pipecat.frames.frames import ErrorFrame
    from pipecat.tests.utils import SleepFrame, run_test

    from .rime_realtime_peer import Peer

    peer = Peer()
    peer.refuse["session.update"] = "invalid_request"
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with Rime(api_key="local-peer") as client:
            service = RimeRealtimeLLMService(client=client, endpoint=endpoint)
            down, up = await run_test(service, frames_to_send=[LLMRunFrame(), SleepFrame(0.05)])
            assert not service.is_usable
            assert any(isinstance(frame, ErrorFrame) for frame in up)
            async with client.realtime.connect(endpoint=endpoint) as session:
                assert session.info.request_id == "actual-peer-request"


async def test_cleanup_leaves_no_owned_service_tasks():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        name = service.name
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith(name + "::")]


async def test_completed_imported_calls_are_history_and_never_execute_tools():
    from pipecat.frames.frames import FunctionCallInProgressFrame

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "past",
                    "type": "function",
                    "function": {"name": "old_tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "A later history note"},
        {"role": "tool", "tool_call_id": "past", "content": ""},
    ]
    async with conversation(context=LLMContext(messages)) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        assert [item["type"] for item in peer.history] == [
            "function_call",
            "function_call_output",
            "message",
        ]
        assert not any(isinstance(frame, FunctionCallInProgressFrame) for frame in down)
        assert not any(event["type"] == "response.create" for event in peer.events)
        snapshot = await service.get_history()
        assert len(snapshot) == 3


async def test_recorded_tool_result_reply_does_not_execute_tool_again():
    from pipecat.services.llm_service import FunctionCallParams

    executed = []

    async def lookup(params: FunctionCallParams):
        """Get a recorded result."""
        executed.append(True)
        await params.result_callback({"found": True})

    async with conversation(context=LLMContext(tools=[lookup])) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        await worker.queue_frame(LLMRunFrame())
        parent = await peer.start(await peer.next())
        peer.refuse["response.create"] = "tool_continuation_unavailable"
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=parent,
            item_id="call-item",
            call_id="expired",
            name="lookup",
            arguments="{}",
        )
        await peer.end(parent)
        report = await peer.next()
        assert report["response"]["prism_call_id"] == "expired"
        assert executed == [True]
        ref = await peer.start(report)
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)


async def test_microphone_overflow_fails_before_sending_audio():
    from pipecat.frames.frames import ErrorFrame

    async with conversation(max_input_audio_bytes=192000) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        await worker.queue_frame(InputAudioRawFrame(bytes(192002), 16000, 1))
        await until(lambda: not service.is_usable)
        assert not any(event["type"] == "input_audio_buffer.append" for event in peer.events)
        assert any(isinstance(frame, ErrorFrame) and "buffer limit" in frame.error for frame in up)


async def test_graceful_end_drains_generation_and_playback():
    from pipecat.frames.frames import EndFrame

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await worker.queue_frame(EndFrame())
        await asyncio.sleep(0.02)
        assert not any(event["type"] == "response.cancel" for event in peer.events)
        await peer.message(ref)
        await peer.end(ref)
        receipt = await asyncio.wait_for(peer.receipts.get(), 3)
        assert receipt["played_ms"] > 0


async def test_cancel_deadline_resolves_queued_operation_outcomes():
    from pipecat.frames.frames import CancelFrame

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        peer.gates["session.update"] = asyncio.Event()
        first = LLMUpdateSettingsFrame(delta=service.Settings(voice="awaiting-ack"))
        second = LLMUpdateSettingsFrame(delta=service.Settings(voice="not-submitted"))
        outcomes = {}

        @service.event_handler("on_operation_result")
        async def on_result(service, frame):
            outcomes[frame.request_id] = frame.outcome

        await worker.queue_frame(first)
        await worker.queue_frame(second)
        await until(lambda: len([e for e in peer.events if e["type"] == "session.update"]) == 2)
        await worker.queue_frame(CancelFrame())
        await until(lambda: first.id in outcomes and second.id in outcomes)
        assert outcomes[first.id] == "unknown"
        assert outcomes[second.id] == "rejected"
        assert len([e for e in peer.events if e["type"] == "session.update"]) == 2


async def test_unversioned_policy_change_requires_explicit_recovery():
    from pipecat.frames.frames import ErrorFrame

    async with conversation(unversioned=True) as (peer, service, worker, ctx, output, up, down):
        assert service.session_info.settings_revision is None
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 3)
        update = LLMUpdateSettingsFrame(delta=service.Settings(interrupt_on_speech=False))
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        assert results(up, update)[0].outcome == "accepted"
        await worker.queue_frame(LLMRunFrame())
        await peer.start(await peer.next())
        await until(
            lambda: any(
                isinstance(frame, ErrorFrame) and "revision is unknown" in frame.error
                for frame in up
            )
        )
        assert not service.is_usable


async def test_cancel_active_output_sends_terminal_sdk_playback_report():
    from pipecat.frames.frames import CancelFrame

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await peer.message(ref, samples=1920, done=False)
        await until(lambda: bool(output.writes))
        await worker.queue_frame(CancelFrame())
        receipt = await asyncio.wait_for(peer.receipts.get(), 1)
        assert receipt["response_id"] == ref


async def test_tool_continuation_keeps_typed_text_only_output():
    from pipecat.services.llm_service import FunctionCallParams

    async def lookup(params: FunctionCallParams):
        """Get a result."""
        await params.result_callback({"ok": True})

    async with conversation(context=LLMContext(tools=[lookup])) as (
        peer,
        service,
        worker,
        ctx,
        output,
        up,
        down,
    ):
        await worker.queue_frame(LLMInputTextFrame("Use lookup", audio_response=False))
        parent = await peer.start(await peer.next())
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=parent,
            item_id="call-item",
            call_id="text-only",
            name="lookup",
            arguments="{}",
        )
        await peer.end(parent)
        child = await peer.start(await peer.next())
        await peer.message(child)
        await peer.end(child)
        await until(lambda: peer.receipts.qsize() == 2)
        receipts = [peer.receipts.get_nowait(), peer.receipts.get_nowait()]
        assert [receipt["played_ms"] for receipt in receipts] == [0, 0]
        assert not [frame for frame in output.writes if frame.context_id]


async def test_metadata_acknowledgment_revision_can_start_a_reply():
    async with conversation() as (peer, service, worker, ctx, output, up, down):
        update = LLMUpdateSettingsFrame(
            delta=service.Settings(response_metadata={"test": "revision"})
        )
        await worker.queue_frame(update)
        await until(lambda: results(up, update))
        await worker.queue_frame(LLMRunFrame())
        ref = await peer.start(await peer.next())
        await peer.end(ref)
        await asyncio.wait_for(peer.receipts.get(), 1)
        assert service.is_usable


async def test_cancel_preserves_pending_empty_output_receipt():
    from pipecat.frames.frames import AudioOutputStartFrame, CancelFrame

    async with conversation() as (peer, service, worker, ctx, output, up, down):
        started, release = asyncio.Event(), asyncio.Event()

        @output.event_handler("on_before_process_frame")
        async def delay_output_start(processor, frame):
            if isinstance(frame, AudioOutputStartFrame):
                started.set()
                await release.wait()

        await worker.queue_frame(LLMInputTextFrame("Text only", audio_response=False))
        ref = await peer.start(await peer.next())
        await peer.message(ref, samples=0, done=False)
        await asyncio.wait_for(started.wait(), 1)
        await worker.queue_frame(CancelFrame())
        await asyncio.sleep(0.02)
        release.set()
        receipt = await asyncio.wait_for(peer.receipts.get(), 1)
        assert receipt["response_id"] == ref
        assert receipt["played_ms"] == 0
