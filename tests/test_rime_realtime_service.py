#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism protocol regressions through standard Pipecat aggregation and output."""

import asyncio
import json

import pytest

pytest.importorskip("rimelabs_sdk")

from rimelabs_sdk import RimeInputError
from rimelabs_sdk import realtime as r

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService, RimeReplyFrame
from tests.rime_realtime_peer import conversation, until


def sent(peer, kind):
    return [event for event in peer.events if event["type"] == kind]


async def answer(peer, request=None, *, text="Hello", samples=480):
    response = await peer.start(request or await peer.next())
    await peer.message(response, text=text, samples=samples)
    await peer.end(response)
    return response


@pytest.mark.asyncio
async def test_typed_text_uses_history_and_standard_output_without_receipts():
    async with conversation() as (peer, service, worker, context, output, upstream, downstream):
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Hello Prism"}], run_llm=True
            )
        )
        request = await peer.next()
        assert "prism_input_text" not in request["response"]
        assert peer.history[-1]["text"] == "Hello Prism"
        await answer(peer, request, text="Hello caller", samples=960)
        await until(
            lambda: {"role": "assistant", "content": "Hello caller"} in context.get_messages()
        )
        await until(lambda: bool(output.writes))
        assert context.get_messages().count({"role": "user", "content": "Hello Prism"}) == 1
        assert not sent(peer, "prism.playback.drained")
        assert any(isinstance(f, LLMTextFrame) and f.text == "Hello caller" for f in downstream)
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Second reply")
        await until(
            lambda: {"role": "assistant", "content": "Second reply"} in context.get_messages()
        )
        # The existing local turns must not be re-imported on the next run.
        assert len(sent(peer, "conversation.item.create")) == 1


@pytest.mark.asyncio
async def test_initial_history_import_and_explicit_run():
    context = LLMContext([{"role": "user", "content": "Earlier turn"}])
    async with conversation(context=context) as (peer, service, worker, *_):
        assert len(peer.history) == 1
        assert len(sent(peer, "response.create")) == 1
        await worker.queue_frame(LLMRunFrame())
        await answer(peer)
        assert len(sent(peer, "conversation.item.create")) == 1


@pytest.mark.asyncio
async def test_labeled_reply_and_saved_instruction_deletion():
    async with conversation() as (peer, service, worker, _, _, _, downstream):
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Still there?"}], run_llm=False
            )
        )
        labels = {"call": "this-reply", "step": "2"}
        await worker.queue_frame(RimeReplyFrame(instruction="Check in", response_metadata=labels))
        labels["step"] = "changed"
        request = await peer.next()
        assert peer.history[-1]["text"] == "Still there?"
        assert request["response"]["prism_instruction"] == "Check in"
        assert request["response"]["metadata"] == {
            "prism_cause": "proactive",
            "call": "this-reply",
            "step": "2",
        }
        request["response"]["metadata"]["prism_instruction_item_id"] = "saved-instruction"
        peer.history.append(
            {"id": "saved-instruction", "type": "message", "role": "user", "text": "Check in"}
        )
        response = await peer.start(request)
        await until(
            lambda: any(
                isinstance(frame, LLMFullResponseStartFrame)
                and frame.metadata.get("rime", {}).get("instruction_item_id") == "saved-instruction"
                for frame in downstream
            )
        )
        with pytest.raises(RimeInputError, match="before its reply ends"):
            await service.delete_instruction_item("saved-instruction")
        await peer.message(response)
        await peer.end(response)
        await until(lambda: service._responses[response].ended)
        service._responses.pop(response)
        await service.delete_instruction_item("saved-instruction")
        assert sent(peer, "conversation.item.delete")[-1]["item_id"] == "saved-instruction"
        assert all(item["id"] != "saved-instruction" for item in peer.history)
        with pytest.raises(RimeInputError, match="Unknown saved"):
            await service.delete_instruction_item("saved-instruction")


@pytest.mark.asyncio
async def test_voice_transcript_and_reply_are_recorded_once_without_reimport():
    async with conversation() as (peer, service, worker, context, output, upstream, downstream):
        await peer.emit("input_audio_buffer.speech_started", item_id="speech-1", audio_start_ms=0)
        await peer.emit("input_audio_buffer.speech_stopped", item_id="speech-1", audio_end_ms=100)
        await peer.emit(
            "conversation.item.input_audio_transcription.completed",
            item_id="speech-1",
            transcript="Voice question",
        )
        response = await peer.start(input_ids=["speech-1"])
        await peer.message(response, text="Voice answer")
        await peer.end(response)
        await until(
            lambda: {"role": "assistant", "content": "Voice answer"} in context.get_messages()
        )
        assert context.get_messages().count({"role": "user", "content": "Voice question"}) == 1
        assert len([f for f in downstream if isinstance(f, TranscriptionFrame)]) == 1
        await asyncio.sleep(0.05)
        assert len(sent(peer, "response.create")) == 1
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next")
        assert not sent(peer, "conversation.item.create")


@pytest.mark.asyncio
async def test_interrupted_local_text_does_not_replace_provider_history():
    from pipecat.frames.frames import TTSTextFrame

    async with conversation() as (peer, service, worker, context, output, upstream, _):
        push_frame = output.push_frame

        async def heard_text(frame, direction=None):
            if isinstance(frame, TTSTextFrame):
                frame.text = "Heard portion"
            if direction is None:
                await push_frame(frame)
            else:
                await push_frame(frame, direction)

        output.push_frame = heard_text
        response = await peer.start()
        await peer.message(response, text="Complete generated answer", samples=24000, done=False)
        await until(lambda: bool(output.writes))
        await worker.queue_frame(InterruptionFrame())
        await until(lambda: bool(sent(peer, "conversation.item.truncate")))
        await until(
            lambda: {"role": "assistant", "content": "Heard portion"} in context.get_messages()
        )
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next answer")
        assert not sent(peer, "conversation.item.delete")
        assert not sent(peer, "conversation.item.create")
        assert service.is_usable
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_llm", [False, True])
@pytest.mark.parametrize("direction", list(FrameDirection))
async def test_application_append_does_not_reconcile_unmatched_live_text(run_llm, direction):
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, text="Provider answer", samples=0)
        await peer.end(response)
        await until(lambda: bool(context.get_messages()))
        # Local playback records can differ from generated provider text.
        context.get_messages()[0]["content"] = "Heard portion"
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Next question"}], run_llm=run_llm
            ),
            direction,
        )
        if not run_llm:
            await until(lambda: len(context.get_messages()) == 2)
            assert not sent(peer, "conversation.item.create")
            # Both halves must share the deferred edit.
            opposite = (
                FrameDirection.UPSTREAM
                if direction == FrameDirection.DOWNSTREAM
                else FrameDirection.DOWNSTREAM
            )
            await worker.queue_frame(LLMRunFrame(), opposite)
        await answer(peer)
        assert [event["item"]["text"] for event in sent(peer, "conversation.item.create")] == [
            "Next question"
        ]
        assert not sent(peer, "conversation.item.delete")
        assert service.is_usable


@pytest.mark.asyncio
@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("direction", list(FrameDirection))
async def test_explicit_replacement_removes_live_history(transform, direction):
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, text="Old provider answer", samples=0)
        await peer.end(response)
        await until(lambda: bool(context.get_messages()))
        messages = [{"role": "user", "content": "Replacement"}]
        edit = (
            LLMMessagesTransformFrame(transform=lambda _: messages, run_llm=False)
            if transform
            else LLMMessagesUpdateFrame(messages=messages, run_llm=False)
        )
        await worker.queue_frame(edit, direction)
        await until(lambda: context.get_messages() == messages)
        assert not sent(peer, "conversation.item.delete")
        await worker.queue_frame(LLMRunFrame(), direction)
        await answer(peer)
        assert [event["item_id"] for event in sent(peer, "conversation.item.delete")] == [
            f"message-{response}"
        ]
        assert peer.history[-1]["text"] == "Replacement"
        assert service.is_usable


@pytest.mark.asyncio
async def test_assistant_aggregation_keeps_a_deferred_application_append():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Deferred question"}])
        )
        await until(lambda: len(context.get_messages()) == 1)
        await peer.message(response, text="Current answer", samples=0)
        await peer.end(response)
        await until(lambda: len(context.get_messages()) == 2)
        assert not sent(peer, "conversation.item.create")
        assert len(sent(peer, "response.create")) == 1
        await worker.queue_frame(LLMRunFrame())
        await answer(peer)
        assert [event["item"]["text"] for event in sent(peer, "conversation.item.create")] == [
            "Deferred question"
        ]


@pytest.mark.asyncio
async def test_reply_start_during_explicit_rebuild_keeps_partial_failure_guard():
    context = LLMContext([{"role": "user", "content": "Original"}])
    async with conversation(context=context) as (peer, service, worker, *_):
        gate = peer.gates["conversation.item.delete"] = asyncio.Event()
        await worker.queue_frame(
            LLMMessagesUpdateFrame(
                messages=[{"role": "user", "content": "Replacement"}], run_llm=True
            )
        )
        await until(lambda: bool(sent(peer, "conversation.item.delete")))
        response = await peer.start()
        await until(lambda: response in service._ledger.active_responses)
        gate.set()
        await until(lambda: not service.is_usable)
        assert str(service._failure) == "Prism history edit partially applied"
        assert len(sent(peer, "conversation.item.create")) == 1


@pytest.mark.asyncio
async def test_rejected_history_edit_leaves_local_context_changed():
    async with conversation() as (peer, service, worker, context, _, upstream, __):
        peer.refuse["conversation.item.create"] = "invalid_request"
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Rejected input"}], run_llm=True
            )
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert {"role": "user", "content": "Rejected input"} in context.get_messages()
        assert not peer.history
        assert service.is_usable
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_history_replacement_and_partial_failure_require_recovery():
    context = LLMContext([{"role": "user", "content": "Original"}])
    async with conversation(context=context) as (peer, service, worker, context, _, upstream, __):
        peer.history_failure_at = peer.history_mutations + 2
        await worker.queue_frame(
            LLMMessagesUpdateFrame(
                messages=[{"role": "user", "content": "Replacement"}], run_llm=True
            )
        )
        await until(lambda: not service.is_usable)
        assert context.get_messages() == [{"role": "user", "content": "Replacement"}]
        assert len(sent(peer, "conversation.item.delete")) == 1
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_settings_tools_fillers_truncation_and_metadata_update():
    async def lookup(params):
        await params.result_callback({"status": "paid"})

    async with conversation() as (peer, service, worker, context, *_):
        await worker.queue_frame(
            LLMUpdateSettingsFrame(
                delta=service.Settings(
                    system_instruction="Billing support",
                    tools=[lookup],
                    tool_fillers={"lookup": r.PrismFiller(type="text", text="Checking")},
                    truncation="auto",
                    response_metadata={"department": "billing"},
                )
            )
        )
        await until(lambda: service.session_info.response_metadata == {"department": "billing"})
        assert peer.settings["instructions"] == "Billing support"
        assert service.has_function("lookup")
        assert peer.settings["tools"][0]["prism_filler"]["text"] == "Checking"
        assert peer.settings["truncation"] == "auto"
        await worker.queue_frame(LLMSetToolsFrame(tools=[]))
        await until(lambda: peer.settings["tools"] == [])
        assert not context.tools


@pytest.mark.asyncio
async def test_rejected_settings_preserve_acknowledged_provider_settings():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        peer.refuse["session.update"] = "invalid_request"
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="Rejected"))
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert service._settings.system_instruction is None
        assert service.is_usable


@pytest.mark.asyncio
async def test_audio_and_events_continue_while_settings_acknowledgment_waits():
    async with conversation() as (peer, service, worker, *_):
        peer.gates["session.update"] = asyncio.Event()
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="Waiting"))
        )
        await worker.queue_frame(InputAudioRawFrame(b"\x00\x00" * 480, 24000, 1))
        event = await peer.next("input_audio_buffer.append")
        assert event["audio"]
        peer.gates["session.update"].set()
        await until(lambda: service._settings.system_instruction == "Waiting")


@pytest.mark.asyncio
async def test_tool_result_continues_once_and_is_not_imported_on_next_turn():
    calls = []

    async def lookup(params):
        calls.append(params.arguments)
        await params.result_callback({"status": "paid"})

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        *_,
    ):
        response = await peer.start()
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="call-item",
            output_index=0,
            call_id="call-1",
            name="lookup",
            arguments="{}",
        )
        await peer.end(response)
        continuation = await peer.next()
        assert continuation["response"]["metadata"]["prism_cause"] == "tool_continuation"
        assert len(calls) == 1
        assert len(sent(peer, "conversation.item.create")) == 1
        next_response = await peer.start(continuation)
        await peer.message(next_response, text="It is paid")
        await peer.end(next_response)
        await until(
            lambda: {"role": "assistant", "content": "It is paid"} in context.get_messages()
        )
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Anything else?")
        assert len(calls) == 1
        # Tool output is submitted once, never imported again as history.
        creates = sent(peer, "conversation.item.create")
        assert sum(event["item"]["type"] == "function_call_output" for event in creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rate,channels", [(8000, 1), (16000, 2), (24000, 1), (44100, 2), (48000, 3)]
)
async def test_microphone_pcm_formats(rate, channels):
    async with conversation() as (peer, service, worker, *_):
        await worker.queue_frame(
            InputAudioRawFrame(b"\x00\x00" * (rate // 10) * channels, rate, channels)
        )
        assert (await peer.next("input_audio_buffer.append"))["audio"]


@pytest.mark.asyncio
async def test_interruption_cancels_reply_and_sends_estimated_position():
    async with conversation() as (peer, service, worker, _, output, upstream, downstream):
        response = await peer.start()
        await peer.message(response, samples=24000, done=False)
        await until(lambda: bool(output.writes))
        await worker.queue_frame(InterruptionFrame())
        await until(lambda: bool(sent(peer, "conversation.item.truncate")))
        position = sent(peer, "conversation.item.truncate")[-1]["audio_end_ms"]
        assert 0 <= position <= 1000
        assert sent(peer, "response.cancel")[-1]["response_id"] == response
        before = len(output.writes)
        await peer.emit(
            "response.audio.delta",
            response_id=response,
            output_index=0,
            item_id=f"message-{response}",
            content_index=0,
            delta="EAA=",
        )
        await worker.queue_frame(LLMRunFrame())
        request = await peer.next()
        assert len(output.writes) == before
        await answer(peer, request, text="After interruption")
        assert not sent(peer, "prism.playback.drained")


@pytest.mark.asyncio
@pytest.mark.parametrize("kept_text", [None, "", "Retained words"])
async def test_retained_text_event_is_forwarded_without_replacing_context(kept_text):
    observed = []
    async with conversation() as (peer, service, worker, context, *_):

        @service.event_handler("on_rime_event")
        async def on_event(service, event):
            if isinstance(event.payload, r.ItemTruncated):
                observed.append(event.payload)

        response = await peer.start()
        await peer.message(response, text="Generated answer", samples=0)
        await peer.end(response)
        await until(lambda: bool(context.get_messages()))
        await peer.emit(
            "conversation.item.truncated",
            item_id=f"message-{response}",
            content_index=0,
            audio_end_ms=120,
            **({} if kept_text is None else {"prism_kept_text": kept_text}),
        )
        await until(lambda: bool(observed))
        assert observed == [
            r.ItemTruncated(
                item_id=f"message-{response}",
                content_index=0,
                audio_end_ms=120,
                kept_text=kept_text,
            )
        ]
        assert context.get_messages() == [{"role": "assistant", "content": "Generated answer"}]
        assert not sent(peer, "conversation.item.create")
        assert not sent(peer, "conversation.item.delete")
        assert service.is_usable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary",
    [None, {"items_left_out": 3}, {"items_left_out": 3, "first_kept_item_id": "kept-item"}],
)
async def test_prompt_truncation_summary_is_forwarded_without_deleting_history(summary):
    observed = []
    context = LLMContext([{"role": "user", "content": "Keep this history"}])
    async with conversation(context=context) as (peer, service, worker, context, *_):

        @service.event_handler("on_rime_event")
        async def on_event(service, event):
            if isinstance(event.payload, r.ResponseEnded):
                observed.append(event.payload)

        response = await peer.start()
        await peer.message(response, text="Answer", samples=0)
        await peer.emit(
            "response.done",
            response={
                "id": response,
                "status": "completed",
                **({} if summary is None else {"prism_truncation": summary}),
            },
        )
        await peer.emit("prism.typed_input.ready")
        await until(lambda: any(event.response.response_id == response for event in observed))
        ended = next(event for event in observed if event.response.response_id == response)
        assert ended.truncation == (r.ResponseTruncation(**summary) if summary else None)
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next answer")
        assert peer.history[0]["text"] == "Keep this history"
        assert context.get_messages()[0] == {"role": "user", "content": "Keep this history"}
        assert len(sent(peer, "conversation.item.create")) == 1
        assert not sent(peer, "conversation.item.delete")
        assert service.is_usable


@pytest.mark.asyncio
async def test_unversioned_settings_changes_do_not_require_revision_tracking():
    async with conversation(unversioned=True) as (peer, service, worker, *_):
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(interrupt_on_speech=False))
        )
        await until(lambda: service._settings.interrupt_on_speech is False)
        response = await peer.start()
        await peer.message(response)
        await peer.end(response)
        assert service.is_usable


@pytest.mark.asyncio
async def test_cleanup_leaves_no_owned_tasks():
    async with conversation() as (peer, service, worker, *_):
        pass
    assert all(task.done() for task in service._tasks)
    assert all(task.done() for task in service._requests)
    assert service._connection_task.done()


@pytest.mark.asyncio
async def test_append_without_inference_syncs_on_the_next_run():
    async with conversation() as (peer, service, worker, context, *_):
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": "Remember NOVA"}])
        )
        await until(lambda: bool(context.get_messages()))
        assert not sent(peer, "conversation.item.create")
        assert len(sent(peer, "response.create")) == 1
        await worker.queue_frame(LLMRunFrame())
        await answer(peer)
        assert len(sent(peer, "conversation.item.create")) == 1


@pytest.mark.asyncio
async def test_history_deletion_preserves_retained_server_id():
    context = LLMContext(
        [{"role": "user", "content": "First"}, {"role": "user", "content": "Second"}]
    )
    async with conversation(context=context) as (peer, service, worker, *_):
        retained_id = peer.history[1]["id"]
        await worker.queue_frame(
            LLMMessagesUpdateFrame(messages=[{"role": "user", "content": "Second"}], run_llm=True)
        )
        await answer(peer)
        assert peer.history[0]["id"] == retained_id
        assert len(sent(peer, "conversation.item.delete")) == 1
        assert len(sent(peer, "conversation.item.create")) == 2


@pytest.mark.asyncio
async def test_invalid_replacement_does_not_delete_provider_history():
    context = LLMContext([{"role": "user", "content": "Keep me"}])
    async with conversation(context=context) as (peer, service, worker, _, __, upstream, ___):
        await worker.queue_frame(
            LLMMessagesUpdateFrame(
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "invalid"}}],
                    }
                ],
                run_llm=True,
            )
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert service.is_usable
        assert not sent(peer, "conversation.item.delete")
        assert peer.history[0]["text"] == "Keep me"


@pytest.mark.asyncio
async def test_rebuild_during_active_reply_is_rejected():
    context = LLMContext([{"role": "user", "content": "Keep me"}])
    async with conversation(context=context) as (peer, service, worker, _, __, upstream, ___):
        response = await peer.start()
        await worker.queue_frame(
            LLMMessagesUpdateFrame(
                messages=[{"role": "user", "content": "Replacement"}], run_llm=True
            )
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert service.is_usable
        assert not sent(peer, "conversation.item.delete")
        await peer.end(response)


@pytest.mark.asyncio
async def test_missing_request_acknowledgment_fails_without_replay():
    async with conversation(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (
        peer,
        service,
        worker,
        *_,
    ):
        peer.gates["response.create"] = asyncio.Event()
        await worker.queue_frame(LLMRunFrame())
        await until(lambda: not service.is_usable)
        assert len(sent(peer, "response.create")) == 2


@pytest.mark.asyncio
async def test_text_only_output_survives_restoration_of_output_settings():
    from pipecat.frames.frames import LLMConfigureOutputFrame

    async with conversation() as (peer, service, worker, context, output, _, downstream):
        await worker.queue_frames(
            [
                LLMConfigureOutputFrame(skip_tts=True),
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Text only"}], run_llm=True
                ),
                LLMConfigureOutputFrame(skip_tts=False),
            ]
        )
        await answer(peer, text="Text reply", samples=960)
        await until(
            lambda: {"role": "assistant", "content": "Text reply"} in context.get_messages()
        )
        assert not output.writes
        assert any(isinstance(f, LLMTextFrame) and f.text == "Text reply" for f in downstream)


@pytest.mark.asyncio
async def test_multiple_tool_results_start_one_continuation_after_generation_ends():
    completed = []

    async def lookup(params, value: str):
        completed.append(value)
        await params.result_callback({"value": value})

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        *_,
    ):
        response = await peer.start()
        for index in range(2):
            await peer.emit(
                "response.function_call_arguments.done",
                response_id=response,
                item_id=f"call-item-{index}",
                output_index=index,
                call_id=f"call-{index}",
                name="lookup",
                arguments=json.dumps({"value": str(index)}),
            )
            await until(lambda expected=index + 1: len(completed) == expected)
        await until(lambda: len(sent(peer, "conversation.item.create")) == 2)
        assert len(sent(peer, "response.create")) == 1
        await peer.end(response)
        continuation = await peer.next()
        assert continuation["response"]["metadata"]["prism_cause"] == "tool_continuation"
        await answer(peer, continuation)
        await until(lambda: {"role": "assistant", "content": "Hello"} in context.get_messages())
        assert len(sent(peer, "response.create")) == 2


@pytest.mark.asyncio
async def test_removed_tool_uses_standard_missing_handler():
    completed = []

    async def lookup(params):
        completed.append(True)
        await params.result_callback("Done")

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        *_,
    ):
        response = await peer.start()
        await worker.queue_frame(LLMSetToolsFrame(tools=[]))
        await until(lambda: not service.has_function("lookup"))
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="missing-call",
            output_index=0,
            call_id="call-1",
            name="lookup",
            arguments="{}",
        )
        await peer.end(response)
        request = await peer.next()
        assert completed == []
        result = sent(peer, "conversation.item.create")[-1]["item"]["output"]
        assert "not currently available" in result
        await answer(peer, request)


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["done", None, False, 0, {"status": "paid"}])
async def test_tool_result_encoding_matches_normal_context_aggregation(result):
    async def lookup(params):
        await params.result_callback(result)

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        *_,
    ):
        response = await peer.start()
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="call-item",
            output_index=0,
            call_id="call-1",
            name="lookup",
            arguments="{}",
        )
        await peer.end(response)
        await answer(peer, await peer.next(), text="Tool answer")
        await until(
            lambda: {"role": "assistant", "content": "Tool answer"} in context.get_messages()
        )
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next answer")
        assert len(sent(peer, "conversation.item.create")) == 1
        assert service.is_usable


@pytest.mark.asyncio
async def test_async_final_result_envelope_does_not_duplicate_provider_history():
    finished = asyncio.Event()

    async def lookup(params):
        await finished.wait()
        await params.result_callback({"status": "paid"})

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        *_,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False)
        response = await peer.start()
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="call-item",
            output_index=0,
            call_id="call-1",
            name="lookup",
            arguments="{}",
        )
        await until(lambda: "call-1" in service._calls)
        await peer.end(response)
        finished.set()
        await answer(peer, await peer.next(), text="Async answer")
        await until(
            lambda: {"role": "assistant", "content": "Async answer"} in context.get_messages()
        )
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next")
        assert len(sent(peer, "conversation.item.create")) == 1
        assert service.is_usable


@pytest.mark.asyncio
async def test_multiple_live_messages_match_one_aggregated_assistant_turn():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, item_id="message-1", text="First sentence.", samples=0)
        await peer.message(response, item_id="message-2", text="Second sentence.", samples=0)
        await peer.end(response)
        await until(
            lambda: any("Second sentence." in m.get("content", "") for m in context.get_messages())
        )
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next")
        assert not sent(peer, "conversation.item.create")
        assert not sent(peer, "conversation.item.delete")


@pytest.mark.asyncio
async def test_late_transcript_is_flushed_before_the_assistant_record():
    async with conversation() as (peer, service, worker, context, *_):
        await peer.emit("input_audio_buffer.speech_started", item_id="speech-1", audio_start_ms=0)
        await peer.emit("input_audio_buffer.speech_stopped", item_id="speech-1", audio_end_ms=100)
        response = await peer.start(input_ids=["speech-1"])
        await peer.message(response, text="Answer")
        await peer.emit(
            "conversation.item.input_audio_transcription.completed",
            item_id="speech-1",
            transcript="Late question",
        )
        await peer.end(response)
        await until(lambda: len(context.get_messages()) == 2)
        assert context.get_messages() == [
            {"role": "user", "content": "Late question"},
            {"role": "assistant", "content": "Answer"},
        ]
        await worker.queue_frame(LLMRunFrame())
        await answer(peer, text="Next")
        assert not sent(peer, "conversation.item.create")


@pytest.mark.asyncio
async def test_intermediate_result_is_rejected_but_final_result_continues():
    from pipecat.frames.frames import FunctionCallResultProperties

    async def lookup(params):
        await params.result_callback(
            "Working", properties=FunctionCallResultProperties(is_final=False)
        )
        await params.result_callback("Done")

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        _,
        __,
        upstream,
        ___,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False)
        response = await peer.start()
        await peer.emit(
            "response.function_call_arguments.done",
            response_id=response,
            item_id="call-item",
            output_index=0,
            call_id="call-1",
            name="lookup",
            arguments="{}",
        )
        await peer.end(response)
        await answer(peer, await peer.next())
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        creates = sent(peer, "conversation.item.create")
        assert len(creates) == 1
        assert creates[0]["item"]["output"] == '"Done"'
        assert service.is_usable


@pytest.mark.asyncio
async def test_server_playback_admission_wait_does_not_block_microphone_frames():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, text="Waiting for server playback estimate")
        await peer.end(response, ready=False)
        await until(lambda: bool(context.get_messages()))
        await worker.queue_frames([LLMRunFrame(), InputAudioRawFrame(b"\x00\x00" * 480, 24000, 1)])
        await peer.next("input_audio_buffer.append")
        assert len(sent(peer, "response.create")) == 1
        await peer.emit("prism.typed_input.ready")
        await answer(peer)
        assert service.is_usable


@pytest.mark.asyncio
async def test_failed_startup_closes_session_but_preserves_injected_client():
    from rimelabs_sdk import Rime
    from websockets.asyncio.server import serve

    from pipecat.tests.utils import SleepFrame, run_test
    from tests.rime_realtime_peer import Peer

    peer = Peer()
    peer.refuse["session.update"] = "invalid_request"
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with Rime(api_key="local-peer") as client:
            service = RimeRealtimeLLMService(client=client, endpoint=endpoint)
            _, upstream = await run_test(service, frames_to_send=[LLMRunFrame(), SleepFrame(0.05)])
            assert not service.is_usable
            assert any(isinstance(frame, ErrorFrame) for frame in upstream)
            async with client.realtime.connect(endpoint=endpoint) as session:
                assert session.info.request_id == "actual-peer-request"


@pytest.mark.asyncio
async def test_completed_tool_history_import_does_not_execute_tools():
    calls = []

    async def lookup(params):
        calls.append(True)
        await params.result_callback("Unexpected")

    context = LLMContext(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "imported-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "imported-call", "content": "Completed result"},
        ],
        tools=[lookup],
    )
    async with conversation(context=context) as (peer, service, worker, *_):
        assert calls == []
        assert len(sent(peer, "conversation.item.create")) == 2
        await worker.queue_frame(LLMRunFrame())
        await answer(peer)
        assert calls == []
        assert len(sent(peer, "conversation.item.create")) == 2


@pytest.mark.asyncio
async def test_unexpected_connection_close_requires_explicit_recovery():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        await peer.socket.close()
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert not service.is_usable
        assert len(sent(peer, "session.update")) == 1
