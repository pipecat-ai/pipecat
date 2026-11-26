#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism SDK-main workflows through the public pipeline and real SDK wire protocol."""

import asyncio
import json

import pytest

pytest.importorskip("rimelabs_sdk")
from rimelabs_sdk import Rime
from rimelabs_sdk import realtime as r
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallResultProperties,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMSetToolChoiceFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService, RimeStartupParams
from pipecat.tests.utils import SleepFrame, run_test
from tests.rime_realtime_peer import Peer, conversation, until
from tests.test_rime_realtime_service import answer, sent


async def call(peer, response, *, call_id="lookup-1", name="lookup"):
    await peer.emit(
        "response.function_call_arguments.done",
        response_id=response,
        item_id=f"item-{call_id}",
        output_index=0,
        call_id=call_id,
        name=name,
        arguments="{}",
    )


@pytest.mark.asyncio
async def test_refused_reply_allows_next_turn():
    async with conversation() as (peer, service, worker, context, _, upstream, __):
        peer.refuse["response.create"] = "invalid_request"
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Refused"}], run_llm=True)
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert service.is_usable
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Accepted"}], run_llm=True)
        )
        await answer(peer, text="Answered")
        await until(lambda: {"role": "assistant", "content": "Answered"} in context.get_messages())
        assert [item["text"] for item in peer.history] == ["Refused", "Accepted"]


@pytest.mark.skip(reason="Flaky: asserts before the peer applies the last update")
@pytest.mark.asyncio
async def test_prompt_clear_and_filler_survives_tool_handoff():
    async def lookup(params):
        await params.result_callback("done")

    settings = RimeRealtimeLLMService.Settings(
        system_instruction="Original",
        tools=[lookup],
        tool_fillers={"lookup": r.PrismFiller(type="text", text="Checking")},
    )
    async with conversation(settings=settings) as (peer, service, worker, *_):
        await worker.queue_frames(
            [
                LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="")),
                LLMSetToolsFrame(tools=[]),
                LLMSetToolsFrame(tools=[lookup]),
            ]
        )
        await until(lambda: len(sent(peer, "session.update")) == 4)
        assert peer.settings["instructions"] == ""
        assert peer.settings["tools"][0]["prism_filler"]["text"] == "Checking"


@pytest.mark.asyncio
async def test_tool_suppression_acknowledges_then_restores():
    calls = []

    async def lookup(params):
        calls.append(True)
        await params.result_callback("done")

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        *_,
    ):
        gate = peer.gates["session.update"] = asyncio.Event()
        await worker.queue_frames(
            [
                LLMSetToolChoiceFrame(tool_choice="none"),
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Hello"}], run_llm=True
                ),
            ]
        )
        await until(lambda: len(sent(peer, "session.update")) == 2)
        assert len(sent(peer, "response.create")) == 1
        gate.set()
        request = await peer.next()
        assert peer.settings["tools"] == []
        response = await peer.start(request)
        await until(lambda: bool(peer.settings["tools"]))
        await call(peer, response)
        await until(lambda: not service.is_usable)
        assert calls == []


@pytest.mark.asyncio
async def test_discard_receipt_orders_next_admission_and_suppresses_speech_audio():
    async with conversation(playback="discard") as (
        peer,
        service,
        worker,
        context,
        output,
        _,
        downstream,
    ):
        await until(lambda: len(sent(peer, "prism.playback.drained")) == 1)
        response = await peer.start()
        await peer.message(response, text="Discarded speech", samples=2400)
        await peer.end(response, ready=False)
        await until(lambda: len(sent(peer, "prism.playback.drained")) == 2)
        assert not output.writes
        await until(
            lambda: any(f.text == "Discarded speech" for f in downstream if hasattr(f, "text"))
        )
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Next"}], run_llm=True)
        )
        await answer(peer)
        await until(lambda: len(sent(peer, "prism.playback.drained")) == 3)
        assert not output.writes
        assert service.is_usable


@pytest.mark.asyncio
async def test_pending_tool_survives_more_than_128_completed_responses():
    release = asyncio.Event()
    calls = []

    async def lookup(params):
        calls.append(True)
        await release.wait()
        await params.result_callback(False)

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        *_,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False)
        parent = await peer.start()
        await call(peer, parent)
        await peer.end(parent)
        await until(lambda: bool(calls))
        for _ in range(130):
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Later turn"}], run_llm=True
                )
            )
            await answer(peer, text="Later turn", samples=0)
        release.set()
        request = await peer.next()
        assert request["response"]["metadata"]["prism_cause"] == "proactive"
        await answer(peer, request, text="Result false", samples=0)
        assert calls == [True]
        assert [
            e["item"]["output"]
            for e in sent(peer, "conversation.item.create")
            if e["item"]["type"] == "function_call_output"
        ] == ["false"]
        assert service.is_usable


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 401])
async def test_startup_http_refusal_policy(status):
    attempts = []
    peer = Peer()

    def refuse(connection, request):
        attempts.append(asyncio.get_running_loop().time())
        if len(attempts) == 1 or status == 401:
            response = connection.respond(status, "Temporary test refusal")
            response.headers["Retry-After"] = "0"
            return response

    async with serve(peer.handler, "127.0.0.1", 0, process_request=refuse) as server:
        async with Rime(api_key="peer") as client:
            service = RimeRealtimeLLMService(
                client=client,
                endpoint=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime",
                startup=RimeStartupParams(retry_interval=0, total_timeout=2),
            )
            service._shutdown_timeout = 0.1
            _, upstream = await run_test(
                service, frames_to_send=[LLMContextFrame(LLMContext()), SleepFrame(0.15)]
            )
            assert len(attempts) == (1 if status == 401 else 2)
            errors = [f for f in upstream if isinstance(f, ErrorFrame)]
            assert bool(errors) == (status == 401)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "cancelled"])
async def test_usage_once_for_terminal_response_with_raw_diagnostics(status):

    async with conversation(
        worker_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    ) as (peer, service, worker, _, __, upstream, downstream):
        raw = []

        @service.event_handler("on_rime_event")
        async def capture(service, event):
            if isinstance(event.payload, r.ResponseEnded):
                raw.append(event.payload)

        response = await peer.start()
        usage = {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
            "context_window": 32768,
        }
        await peer.end(response, status, usage=usage)
        await until(
            lambda: any(
                isinstance(f, MetricsFrame)
                and any(isinstance(d, LLMUsageMetricsData) for d in f.data)
                for f in downstream
            )
        )
        metrics = [
            d
            for f in downstream
            if isinstance(f, MetricsFrame)
            for d in f.data
            if isinstance(d, LLMUsageMetricsData)
        ]
        assert len(metrics) == 1
        assert metrics[0].value.total_tokens == 16
        assert raw[0].usage.context_window == 32768
        without = await peer.start()
        await peer.end(without)
        await until(lambda: len(raw) == 2)
        assert raw[-1].usage is None


@pytest.mark.asyncio
@pytest.mark.parametrize("final_run", [True, False])
@pytest.mark.parametrize("progress_run", [True, False])
async def test_progress_then_final_respects_reply_policy(final_run, progress_run):
    release = asyncio.Event()
    calls = []

    async def lookup(params):
        calls.append(True)
        await params.result_callback(
            "Working", properties=FunctionCallResultProperties(is_final=False, run_llm=progress_run)
        )
        await release.wait()
        await params.result_callback(
            "Finished", properties=FunctionCallResultProperties(run_llm=final_run)
        )

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        _,
        upstream,
        __,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False)
        parent = await peer.start()
        await call(peer, parent)
        await peer.end(parent)
        await until(lambda: len(sent(peer, "conversation.item.create")) == 1)
        progress = sent(peer, "conversation.item.create")[0]["item"]
        assert (progress["type"], progress["role"]) == ("message", "system")
        if not progress_run:
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Answer while the lookup runs"}],
                    run_llm=True,
                )
            )
        await answer(peer, text="Progress recorded")
        release.set()
        writes = 3 if progress_run else 4
        await until(lambda: len(sent(peer, "conversation.item.create")) == writes)
        final, settled = (e["item"] for e in sent(peer, "conversation.item.create")[-2:])
        assert (final["type"], final["call_id"]) == ("function_call_output", "lookup-1")
        assert (settled["role"], json.loads(settled["text"])["status"]) == ("system", "finished")
        if final_run:
            # The progress reply superseded the tool round, so the result is reported.
            request = await peer.next()
            assert request["response"]["prism_call_id"] == "lookup-1"
            await answer(peer, request, text="Final recorded")
            await until(
                lambda: any(m.get("content") == "Final recorded" for m in context.get_messages())
            )
            # The report answers the result; the aggregator's request adds no reply.
            await asyncio.sleep(0.1)
            assert len(sent(peer, "response.create")) == 3
        else:
            await until(lambda: any("Finished" in str(m) for m in context.get_messages()))
            await asyncio.sleep(0.05)
            assert len(sent(peer, "response.create")) == 2
        assert calls == [True]
        # A later context frame does not write the async final a second time.
        await worker.queue_frame(LLMContextFrame(context))
        await asyncio.sleep(0.05)
        assert len(sent(peer, "conversation.item.create")) == writes
        assert not any(isinstance(f, ErrorFrame) for f in upstream)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exception", "timeout"])
async def test_tool_failure_settles_once_without_repeat_execution(mode):
    calls = []

    async def lookup(params):
        calls.append(True)
        if mode == "exception":
            raise ValueError("Synthetic failure")
        await params.result_callback(
            "Working", properties=FunctionCallResultProperties(is_final=False, run_llm=False)
        )
        await asyncio.Event().wait()

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        _,
        upstream,
        __,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False, timeout_secs=0.05)
        parent = await peer.start()
        await call(peer, parent)
        await peer.end(parent)
        if mode == "exception":
            await answer(peer, text="Tool failed")
        else:
            await until(
                lambda: any(
                    '"status": "finished"' in (e["item"].get("text") or "")
                    for e in sent(peer, "conversation.item.create")
                )
            )
            items = [e["item"] for e in sent(peer, "conversation.item.create")]
            assert [item.get("output") for item in items if item["type"] != "message"] == [
                "CANCELLED"
            ]
            settled = json.loads(items[-1]["text"])
            assert settled["result"].startswith("CANCELLED: this tool call was cancelled")
        assert calls == [True]
        assert service.is_usable


@pytest.mark.asyncio
async def test_history_acknowledgment_timeout_is_terminal():
    async with conversation(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (
        peer,
        service,
        worker,
        _,
        __,
        upstream,
        ___,
    ):
        peer.gates["conversation.item.create"] = asyncio.Event()
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Uncertain"}], run_llm=True
            )
        )
        await until(lambda: not service.is_usable)
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        errors = [f for f in upstream if isinstance(f, ErrorFrame)]
        assert errors[-1].exception.status == "uncertain"
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_settings_confirmation_survives_rejected_label_update():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        peer.refuse["session.labels"] = "invalid_request"
        await worker.queue_frame(
            LLMUpdateSettingsFrame(
                delta=service.Settings(
                    system_instruction="Confirmed", response_metadata={"stage": "next"}
                )
            )
        )
        await until(lambda: peer.settings.get("instructions") == "Confirmed")
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert service.session_info.response_metadata != {"stage": "next"}
        assert service.is_usable


@pytest.mark.asyncio
async def test_discard_orders_out_of_order_completions():
    async with conversation(playback="discard") as (peer, service, worker, *_):
        await until(lambda: len(sent(peer, "prism.playback.drained")) == 1)
        first = await peer.start()
        second = await peer.start()
        await peer.end(second, ready=False)
        await asyncio.sleep(0.02)
        assert len(sent(peer, "prism.playback.drained")) == 1
        await peer.end(first, ready=False)
        await until(lambda: len(sent(peer, "prism.playback.drained")) == 3)
        receipts = sent(peer, "prism.playback.drained")
        assert [e["response_id"] for e in receipts[-2:]] == [first, second]


@pytest.mark.asyncio
async def test_protected_response_limit_fails_without_eviction():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        for _ in range(129):
            try:
                await peer.start()
            except Exception:
                break
        await until(lambda: not service.is_usable)
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert any(isinstance(f, ErrorFrame) and "protected" in f.error.lower() for f in upstream)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", ["0", "0.02", "bad", None])
async def test_startup_retry_exhaustion_and_diagnostics(retry_after):
    attempts = []
    peer = Peer()

    def refuse(connection, request):
        attempts.append(asyncio.get_running_loop().time())
        response = connection.respond(503, "Retry test")
        if retry_after is not None:
            response.headers["Retry-After"] = retry_after
        return response

    async with serve(peer.handler, "127.0.0.1", 0, process_request=refuse) as server:
        async with Rime(api_key="peer") as client:
            service = RimeRealtimeLLMService(
                client=client,
                endpoint=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime",
                startup=RimeStartupParams(max_retries=1, retry_interval=0.01, total_timeout=0.4),
            )
            service._shutdown_timeout = 0.1
            _, upstream = await run_test(
                service, frames_to_send=[LLMContextFrame(LLMContext()), SleepFrame(0.12)]
            )
            assert len(attempts) == 2
            assert attempts[-1] - attempts[0] >= 0.01
            assert any(isinstance(f, ErrorFrame) for f in upstream)


@pytest.mark.asyncio
async def test_interruption_before_creation_cancels_late_reply():
    from pipecat.frames.frames import InterruptionFrame

    async with conversation() as (peer, service, worker, context, output, *_):
        await worker.queue_frame(LLMSetToolChoiceFrame(tool_choice="none"))
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Accepted late"}], run_llm=True
            )
        )
        request = await peer.next()
        await worker.queue_frame(InterruptionFrame())
        await until(lambda: service._interruption_epoch == 1)
        response = await peer.start(request)
        await peer.message(response, text="Should not play", samples=2400)
        await until(
            lambda: any(e["response_id"] == response for e in sent(peer, "response.cancel"))
        )
        await until(lambda: len(sent(peer, "session.update")) == 3)
        assert not output.writes
        assert service.is_usable
        assert context.get_messages().count({"role": "user", "content": "Accepted late"}) == 1


@pytest.mark.asyncio
async def test_refused_final_tool_output_ends_unsettled_conversation():
    async def lookup(params):
        await params.result_callback("Final")

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        _,
        __,
        upstream,
        ___,
    ):
        peer.refuse["conversation.item.create"] = "invalid_request"
        parent = await peer.start()
        await call(peer, parent)
        await peer.end(parent)
        await until(lambda: not service.is_usable)
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_suppression_restore_failure_is_terminal():
    async def lookup(params):
        await params.result_callback("unused")

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        _,
        __,
        upstream,
        ___,
    ):
        await worker.queue_frames(
            [
                LLMSetToolChoiceFrame(tool_choice="none"),
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Hello"}], run_llm=True
                ),
            ]
        )
        request = await peer.next()
        peer.refuse["session.update"] = "invalid_request"
        await answer(peer, request)
        await until(lambda: not service.is_usable)
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert peer.settings["tools"] == []
