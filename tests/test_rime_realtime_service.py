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

from rimelabs_sdk import realtime as r

from pipecat.adapters.llm.schemas.direct_function import tool_options
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService
from pipecat.utils.async_tool_cancellation import (
    ASYNC_TOOL_CANCELLATION_INSTRUCTIONS,
    cancel_tool_name,
)
from tests.rime_realtime_peer import conversation, until


def sent(peer, kind):
    return [event for event in peer.events if event["type"] == kind]


async def answer(peer, request=None, *, text="Hello", samples=480):
    response = await peer.start(request or await peer.next())
    await peer.message(response, text=text, samples=samples)
    await peer.end(response)
    return response


def created_texts(peer):
    return [event["item"].get("text") for event in sent(peer, "conversation.item.create")]


def hold_audio(output):
    """Keep the bot speaking until the returned event is set."""
    release = asyncio.Event()
    write_audio = output.write_audio_frame

    async def held_write(frame):
        await release.wait()
        return await write_audio(frame)

    output.write_audio_frame = held_write
    return release


async def speak(peer, downstream):
    response = await peer.start()
    await peer.message(response, text="Speaking", samples=2400)
    await peer.end(response)
    await until(lambda: any(isinstance(f, BotStartedSpeakingFrame) for f in downstream))
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
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Again"}], run_llm=True)
        )
        await answer(peer, text="Second reply")
        await until(
            lambda: {"role": "assistant", "content": "Second reply"} in context.get_messages()
        )
        # Each append writes only its own messages.
        assert created_texts(peer) == ["Hello Prism", "Again"]


@pytest.mark.asyncio
async def test_initial_history_import_starts_one_reply():
    context = LLMContext(
        [
            {"role": "system", "content": "Be brief"},
            {"role": "developer", "content": "Speak English"},
            {"role": "user", "content": "Earlier turn"},
        ]
    )
    async with conversation(context=context) as (peer, *_):
        assert [(item["role"], item["text"]) for item in peer.history] == [
            ("system", "Be brief"),
            ("system", "Speak English"),
            ("user", "Earlier turn"),
        ]
        assert len(sent(peer, "response.create")) == 1


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
        assert not sent(peer, "conversation.item.delete")
        assert not sent(peer, "conversation.item.create")
        assert service.is_usable
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_llm", [False, True])
async def test_append_is_written_at_once_and_replies_only_when_asked(run_llm):
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, text="Provider answer", samples=0)
        await peer.end(response)
        await until(lambda: bool(context.get_messages()))
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": "Next question"}], run_llm=run_llm
            )
        )
        await until(lambda: created_texts(peer) == ["Next question"])
        await until(lambda: len(context.get_messages()) == 2)
        if run_llm:
            await answer(peer)
        else:
            await asyncio.sleep(0.05)
        assert len(sent(peer, "response.create")) == (2 if run_llm else 1)
        assert not sent(peer, "conversation.item.delete")
        assert service.is_usable


@pytest.mark.asyncio
async def test_context_replacement_leaves_provider_history():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, text="Old provider answer", samples=0)
        await peer.end(response)
        await until(lambda: bool(context.get_messages()))
        messages = [{"role": "user", "content": "Replacement"}]
        await worker.queue_frame(LLMMessagesUpdateFrame(messages=messages, run_llm=False))
        await until(lambda: context.get_messages() == messages)
        await asyncio.sleep(0.05)
        assert not sent(peer, "conversation.item.create")
        assert not sent(peer, "conversation.item.delete")
        assert len(sent(peer, "response.create")) == 1
        assert service.is_usable


@pytest.mark.asyncio
async def test_append_during_reply_is_written_at_once():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Side note"}])
        )
        await until(lambda: created_texts(peer) == ["Side note"])
        await peer.message(response, text="Current answer", samples=0)
        await peer.end(response)
        await until(lambda: len(context.get_messages()) == 2)
        assert len(sent(peer, "response.create")) == 1


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
        error = next(f for f in upstream if isinstance(f, ErrorFrame)).exception
        assert error.status == "rejected"
        assert {"role": "user", "content": "Rejected input"} in context.get_messages()
        assert not peer.history
        assert service.is_usable
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_partial_append_failure_is_recoverable():
    async with conversation() as (peer, service, worker, context, _, upstream, __):
        peer.history_failure_at = peer.history_mutations + 2
        await worker.queue_frame(
            LLMMessagesAppendFrame(
                messages=[
                    {"role": "user", "content": "First"},
                    {"role": "user", "content": "Second"},
                ],
                run_llm=True,
            )
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        error = next(f for f in upstream if isinstance(f, ErrorFrame)).exception
        assert error.status == "partial"
        assert error.recoverable
        assert service.is_usable
        assert [item["text"] for item in peer.history] == ["First"]
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("run_llm", [False, True])
async def test_append_queued_with_the_opening_reaches_prism(run_llm):
    async def opening(worker, context):
        await worker.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Early"}], run_llm=run_llm
                ),
                LLMRunFrame(),
            ]
        )

    context = LLMContext([{"role": "system", "content": "Be brief"}])
    async with conversation(context=context, opening=opening) as (peer, service, *_):
        await until(lambda: {"role": "user", "content": "Early"} in context.get_messages())
        await asyncio.sleep(0.05)
        assert created_texts(peer) == ["Be brief", "Early"]
        assert len(sent(peer, "response.create")) == 1
        assert service.is_usable


@pytest.mark.asyncio
async def test_append_recorded_before_the_opening_is_written_once():
    async def opening(worker, context):
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Early"}])
        )
        await until(lambda: bool(context.get_messages()))
        await worker.queue_frame(LLMRunFrame())

    async with conversation(opening=opening) as (peer, *_):
        assert created_texts(peer) == ["Early"]
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("prism_answers", [False, True])
async def test_reply_held_while_the_bot_speaks_is_skipped_once_prism_answers(prism_answers):
    async with conversation() as (peer, service, worker, context, output, _, downstream):
        release = hold_audio(output)
        try:
            await speak(peer, downstream)
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Appended"}], run_llm=True
                )
            )
            await until(lambda: created_texts(peer) == ["Appended"])
            await asyncio.sleep(0.05)
            if prism_answers:
                response = await peer.start()
                await peer.message(response, text="Answer that took it in", samples=0)
                await peer.end(response)
            release.set()
            await until(lambda: any(isinstance(f, BotStoppedSpeakingFrame) for f in downstream))
            if prism_answers:
                await asyncio.sleep(0.1)
                assert len(sent(peer, "response.create")) == 1
            else:
                await answer(peer)
                assert len(sent(peer, "response.create")) == 2
            assert service.is_usable
        finally:
            release.set()


@pytest.mark.asyncio
async def test_response_started_during_the_append_write_is_not_its_reply():
    async with conversation() as (peer, service, worker, context, output, _, downstream):
        release = hold_audio(output)
        written = peer.gates["conversation.item.create"] = asyncio.Event()
        try:
            await speak(peer, downstream)
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Appended"}], run_llm=True
                )
            )
            await until(lambda: created_texts(peer) == ["Appended"])
            response = await peer.start()
            await peer.message(response, text="Earlier answer", samples=0)
            await peer.end(response)
            await asyncio.sleep(0.05)
            written.set()
            await until(lambda: [item["text"] for item in peer.history] == ["Appended"])
            release.set()
            await answer(peer)
            assert len(sent(peer, "response.create")) == 2
            assert service.is_usable
        finally:
            written.set()
            release.set()


@pytest.mark.asyncio
async def test_failed_append_keeps_the_reply_owed_to_an_earlier_one():
    async with conversation() as (peer, service, worker, context, output, upstream, downstream):
        release = hold_audio(output)
        try:
            await speak(peer, downstream)
            peer.history_failure_at = peer.history_mutations + 2
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "First"}], run_llm=True
                )
            )
            await worker.queue_frame(
                LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Second"}])
            )
            await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
            release.set()
            await answer(peer)
            assert [item["text"] for item in peer.history] == ["First"]
            assert len(sent(peer, "response.create")) == 2
            assert service.is_usable
        finally:
            release.set()


@pytest.mark.asyncio
async def test_interruption_drops_a_held_reply():
    async with conversation() as (peer, service, worker, context, output, _, downstream):
        release = hold_audio(output)
        try:
            await speak(peer, downstream)
            await worker.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Appended"}], run_llm=True
                )
            )
            await until(lambda: created_texts(peer) == ["Appended"])
            await worker.queue_frame(InterruptionFrame())
            release.set()
            await until(lambda: not service._requests)
            await worker.queue_frame(
                LLMMessagesUpdateFrame(messages=context.get_messages(), run_llm=True)
            )
            await asyncio.sleep(0.1)
            assert len(sent(peer, "response.create")) == 1
            assert service.is_usable
        finally:
            release.set()


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


@pytest.mark.skip(reason="Flaky: asserts before the session update reaches the peer")
@pytest.mark.asyncio
async def test_speed_and_interruption_rules_open_and_update():
    settings = RimeRealtimeLLMService.Settings(
        speed=1.1, interrupt_min_words=2, acknowledgment_phrases=["uh huh"]
    )
    async with conversation(settings=settings) as (peer, service, worker, *_):
        opening = sent(peer, "session.update")[0]["session"]
        assert opening["speed"] == 1.1
        assert opening["turn_detection"]["interrupt_min_words"] == 2
        assert opening["turn_detection"]["acknowledgment_phrases"] == ["uh huh"]
        assert "interrupt_phrases" not in opening["turn_detection"]

        await worker.queue_frame(
            LLMUpdateSettingsFrame(
                delta=service.Settings(speed=0.9, interrupt_phrases=["stop", "wait"])
            )
        )
        await until(lambda: service._settings.speed == 0.9)
        update = sent(peer, "session.update")[-1]["session"]
        assert update["speed"] == 0.9
        assert update["turn_detection"] == {"interrupt_phrases": ["stop", "wait"]}
        assert service._settings.interrupt_phrases == ["stop", "wait"]
        assert service._settings.interrupt_min_words == 2
        detection = service.session_info.turn_detection
        assert detection.interrupt_min_words == 2
        assert detection.interrupt_phrases == ("stop", "wait")
        assert detection.acknowledgment_phrases == ("uh huh",)

        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(acknowledgment_phrases=[]))
        )
        await until(lambda: service._settings.acknowledgment_phrases == [])
        assert sent(peer, "session.update")[-1]["session"]["turn_detection"] == {
            "acknowledgment_phrases": []
        }
        assert service.is_usable


@pytest.mark.asyncio
async def test_opening_turn_detection_rules_join_the_settings():
    turn_detection = r.TurnDetection(eagerness="low", interrupt_min_words=3)
    async with conversation(turn_detection=turn_detection) as (peer, service, *_):
        opening = sent(peer, "session.update")[0]["session"]["turn_detection"]
        assert opening["eagerness"] == "low"
        assert opening["interrupt_min_words"] == 3
        assert service._settings.interrupt_min_words == 3


def test_interruption_rule_set_twice_is_rejected():
    with pytest.raises(ValueError, match="interrupt_min_words"):
        RimeRealtimeLLMService(
            turn_detection=r.TurnDetection(interrupt_min_words=3),
            settings=RimeRealtimeLLMService.Settings(interrupt_min_words=2),
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"speed": 2.0},
        {"speed": True},
        {"interrupt_min_words": 0},
        {"interrupt_phrases": "stop"},
        {"acknowledgment_phrases": [1]},
    ],
)
def test_invalid_speed_and_interruption_rules_are_rejected(settings):
    with pytest.raises(ValueError):
        RimeRealtimeLLMService(settings=RimeRealtimeLLMService.Settings(**settings))


@pytest.mark.asyncio
async def test_invalid_interruption_rule_update_is_rejected_without_a_write():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        updates = len(sent(peer, "session.update"))
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(interrupt_min_words=11))
        )
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert len(sent(peer, "session.update")) == updates
        assert service._settings.interrupt_min_words is None
        assert service.is_usable


@pytest.mark.asyncio
async def test_full_partial_transcripts_replace_earlier_partials():
    async with conversation() as (peer, service, worker, context, output, upstream, downstream):
        event = "conversation.item.input_audio_transcription.delta"
        await peer.emit(event, item_id="speech-1", delta="Wether")
        await peer.emit(event, item_id="speech-1", delta=" in", transcript="Weather in")
        await peer.emit(event, item_id="speech-1", delta=" Paris")
        await until(
            lambda: len([f for f in downstream if isinstance(f, InterimTranscriptionFrame)]) == 3
        )
        texts = [f.text for f in downstream if isinstance(f, InterimTranscriptionFrame)]
        assert texts == ["Wether", "Weather in", "Weather in Paris"]


@pytest.mark.asyncio
async def test_long_history_message_is_left_to_prism():
    text = "word " * 1200
    context = LLMContext([{"role": "user", "content": text}])
    async with conversation(context=context) as (peer, service, *_):
        assert created_texts(peer) == [text]
        assert service.is_usable


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


@tool_options(cancel_on_interruption=False)
async def slow_lookup(params):
    """Look up an order in the background."""
    await params.result_callback({"status": "paid"})


@tool_options(cancel_on_interruption=False, cancellable_by_llm=True)
async def stoppable_lookup(params):
    """Look up an order the model may stop."""
    await params.result_callback({"status": "paid"})


def tool_names(peer):
    return [tool["function"]["name"] for tool in peer.settings.get("tools", [])]


@pytest.mark.asyncio
async def test_runtime_instruction_keeps_the_composed_guidance():
    settings = RimeRealtimeLLMService.Settings(system_instruction="Support", tools=[slow_lookup])
    async with conversation(settings=settings) as (peer, service, worker, *_):
        assert peer.settings["instructions"] == f"Support\n\n{ASYNC_TOOL_INSTRUCTIONS}"
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="Billing"))
        )
        await until(lambda: peer.settings["instructions"].startswith("Billing"))
        assert peer.settings["instructions"] == f"Billing\n\n{ASYNC_TOOL_INSTRUCTIONS}"
        assert service._settings.system_instruction == peer.settings["instructions"]


@pytest.mark.asyncio
async def test_tools_update_sends_cancel_tools_and_their_guidance():
    settings = RimeRealtimeLLMService.Settings(system_instruction="Support")
    async with conversation(settings=settings) as (peer, service, worker, context, *_):
        await worker.queue_frame(LLMSetToolsFrame(tools=[stoppable_lookup]))
        await until(lambda: "stoppable_lookup" in tool_names(peer))
        assert tool_names(peer) == ["stoppable_lookup", cancel_tool_name("stoppable_lookup")]
        assert ASYNC_TOOL_CANCELLATION_INSTRUCTIONS in peer.settings["instructions"]
        assert ASYNC_TOOL_INSTRUCTIONS in peer.settings["instructions"]

        await worker.queue_frame(LLMSetToolsFrame(tools=[]))
        await until(lambda: peer.settings["tools"] == [])
        assert peer.settings["instructions"] == "Support"
        assert not context.tools


@pytest.mark.asyncio
async def test_refused_tools_update_restores_tools_and_instruction():
    settings = RimeRealtimeLLMService.Settings(system_instruction="Support")
    async with conversation(settings=settings) as (peer, service, worker, context, _, upstream, __):
        peer.refuse["session.update"] = "invalid_request"
        await worker.queue_frame(LLMSetToolsFrame(tools=[stoppable_lookup]))
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert not service.has_function("stoppable_lookup")
        assert not service.has_function(cancel_tool_name("stoppable_lookup"))
        assert service._settings.system_instruction == "Support"
        # The user aggregator set the refused tools on the context; they're cleared.
        assert not context.tools
        assert service.is_usable

        # The next update compares against what Prism holds, so it sends nothing stale.
        await worker.queue_frame(LLMUpdateSettingsFrame(delta=service.Settings(speed=1.1)))
        await until(lambda: peer.settings.get("speed") == 1.1)
        assert "tools" not in sent(peer, "session.update")[-1]["session"]
        assert "instructions" not in sent(peer, "session.update")[-1]["session"]


@pytest.mark.asyncio
async def test_context_frame_during_tools_update_keeps_the_new_handlers():
    settings = RimeRealtimeLLMService.Settings(system_instruction="Support")
    context = LLMContext(tools=[slow_lookup])
    async with conversation(settings=settings, context=context) as (peer, service, worker, *_):
        assert tool_names(peer) == ["slow_lookup"]
        peer.gates["session.update"] = asyncio.Event()
        updates = len(sent(peer, "session.update"))
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(tools=[stoppable_lookup]))
        )
        await until(lambda: len(sent(peer, "session.update")) > updates)
        # The base service syncs handlers to the context's tools on this frame.
        await worker.queue_frame(LLMContextFrame(context))
        await asyncio.sleep(0.1)
        peer.gates["session.update"].set()
        await until(lambda: "stoppable_lookup" in tool_names(peer))
        await asyncio.sleep(0.1)
        assert service.has_function("stoppable_lookup")
        assert service.has_function(cancel_tool_name("stoppable_lookup"))
        assert service._settings.system_instruction == peer.settings["instructions"]


@pytest.mark.skip(reason="Flaky: asserts before the session update reaches the peer")
@pytest.mark.asyncio
async def test_update_without_instruction_or_tool_changes_sends_neither():
    settings = RimeRealtimeLLMService.Settings(system_instruction="Support", tools=[slow_lookup])
    async with conversation(settings=settings) as (peer, service, worker, *_):
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="Support", speed=0.9))
        )
        await until(lambda: service._settings.speed == 0.9)
        update = sent(peer, "session.update")[-1]["session"]
        assert update == {"speed": 0.9}


@pytest.mark.skip(reason="Flaky: asserts before the session update reaches the peer")
@pytest.mark.asyncio
async def test_settings_for_another_service_pass_through_unapplied():
    async with conversation() as (peer, service, worker, _, __, ___, downstream):
        updates = len(sent(peer, "session.update"))
        other = RimeRealtimeLLMService()
        await worker.queue_frame(
            LLMUpdateSettingsFrame(
                delta=service.Settings(system_instruction="Elsewhere"), service=other
            )
        )
        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=service.Settings(system_instruction="Here"))
        )
        await until(lambda: service._settings.system_instruction == "Here")
        assert len(sent(peer, "session.update")) == updates + 1
        passed = [f for f in downstream if isinstance(f, LLMUpdateSettingsFrame)]
        assert [f.service for f in passed] == [other]


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
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Go on"}], run_llm=True)
        )
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
async def test_system_append_without_inference_is_written_without_a_reply():
    async with conversation() as (peer, service, worker, context, *_):
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": "Remember NOVA"}])
        )
        await until(lambda: bool(sent(peer, "conversation.item.create")))
        assert sent(peer, "conversation.item.create")[0]["item"]["role"] == "system"
        await until(lambda: bool(context.get_messages()))
        await asyncio.sleep(0.05)
        assert len(sent(peer, "response.create")) == 1


@pytest.mark.asyncio
async def test_invalid_append_is_rejected_without_writes():
    context = LLMContext([{"role": "user", "content": "Keep me"}])
    async with conversation(context=context) as (peer, service, worker, _, __, upstream, ___):
        await worker.queue_frame(
            LLMMessagesAppendFrame(
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
        assert [item["text"] for item in peer.history] == ["Keep me"]


@pytest.mark.asyncio
async def test_missing_request_acknowledgment_fails_without_replay():
    async with conversation(timeouts=r.RealtimeTimeouts(request_s=0.05)) as (
        peer,
        service,
        worker,
        *_,
    ):
        peer.gates["response.create"] = asyncio.Event()
        await worker.queue_frame(
            LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Hi"}], run_llm=True)
        )
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


@pytest.mark.skip(reason="Flaky: the second call can arrive before the first reply request")
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
@pytest.mark.parametrize("result", ["done", "", None, False, 0, {}, {"status": "paid"}])
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
        if result is not None:
            await answer(peer, await peer.next(), text="Tool answer")
            await until(
                lambda: {"role": "assistant", "content": "Tool answer"} in context.get_messages()
            )
        else:
            await until(lambda: len(sent(peer, "conversation.item.create")) == 1)
            assert len(sent(peer, "response.create")) == 1
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
        assert len(sent(peer, "conversation.item.create")) == 1
        assert service.is_usable


@pytest.mark.asyncio
async def test_later_context_frames_do_not_reimport_history_or_reply():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, item_id="message-1", text="First sentence.", samples=0)
        await peer.message(response, item_id="message-2", text="Second sentence.", samples=0)
        await peer.end(response)
        await until(
            lambda: any("Second sentence." in m.get("content", "") for m in context.get_messages())
        )
        # A run requested mid-conversation sends the context downstream again.
        await worker.queue_frame(LLMRunFrame())
        await asyncio.sleep(0.05)
        assert not sent(peer, "conversation.item.create")
        assert not sent(peer, "conversation.item.delete")
        assert len(sent(peer, "response.create")) == 1


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
        assert not sent(peer, "conversation.item.create")


@pytest.mark.asyncio
async def test_silent_progress_keeps_the_call_open_until_the_final_result():
    from pipecat.frames.frames import FunctionCallResultProperties

    async def lookup(params):
        await params.result_callback(
            "Working", properties=FunctionCallResultProperties(is_final=False, run_llm=False)
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
        continuation = await peer.next()
        assert continuation["response"]["metadata"]["prism_cause"] == "tool_continuation"
        await answer(peer, continuation)
        creates = sent(peer, "conversation.item.create")
        assert [event["item"]["type"] for event in creates] == [
            "message",
            "function_call_output",
            "message",
        ]
        progress = json.loads(creates[0]["item"]["text"])
        assert creates[0]["item"]["role"] == "system"
        assert (progress["tool_call_id"], progress["status"]) == ("call-1", "running")
        assert progress["result"] == '"Working"'
        assert (creates[1]["item"]["call_id"], creates[1]["item"]["output"]) == ("call-1", '"Done"')
        # Prism files the output beside the call, so the result follows the progress too.
        settled = json.loads(creates[2]["item"]["text"])
        assert creates[2]["item"]["role"] == "system"
        assert (settled["tool_call_id"], settled["status"]) == ("call-1", "finished")
        assert settled["result"] == '"Done"'
        assert len(sent(peer, "response.create")) == 2
        assert not any(isinstance(f, ErrorFrame) for f in upstream)
        assert service.is_usable


async def emit_call(peer, response, call_id="call-1", name="lookup"):
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
async def test_tool_result_waits_for_the_bot_to_stop_speaking():
    finished = asyncio.Event()

    async def lookup(params):
        await finished.wait()
        await params.result_callback({"status": "paid"})

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        output,
        _,
        downstream,
    ):
        release = hold_audio(output)
        try:
            parent = await peer.start()
            await peer.message(parent, text="Checking", samples=2400)
            await emit_call(peer, parent)
            await peer.end(parent)
            await until(lambda: any(isinstance(f, BotStartedSpeakingFrame) for f in downstream))
            finished.set()
            await until(lambda: len(sent(peer, "conversation.item.create")) == 1)
            await asyncio.sleep(0.1)
            assert len(sent(peer, "response.create")) == 1
            release.set()
            continuation = await peer.next()
            assert continuation["response"]["metadata"]["prism_cause"] == "tool_continuation"
            await answer(peer, continuation)
            assert service.is_usable
        finally:
            finished.set()
            release.set()


@pytest.mark.asyncio
async def test_tool_result_frames_keep_their_reply_policy():
    from pipecat.frames.frames import FunctionCallResultFrame, FunctionCallResultProperties

    async def lookup(params):
        await params.result_callback(
            {"status": "paid"}, properties=FunctionCallResultProperties(run_llm=False)
        )

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        context,
        _,
        __,
        downstream,
    ):
        parent = await peer.start()
        await emit_call(peer, parent)
        await peer.end(parent)
        await until(lambda: any(isinstance(f, FunctionCallResultFrame) for f in downstream))
        result = next(f for f in downstream if isinstance(f, FunctionCallResultFrame))
        assert result.properties.run_llm is False
        await asyncio.sleep(0.05)
        assert len(sent(peer, "response.create")) == 1
        assert service._responses[parent].hold is not None
        # A later response, such as the user's next turn, takes the result in.
        turn = await peer.start()
        await peer.end(turn)
        await until(lambda: service._responses[parent].hold is None)
        assert service.is_usable


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [True, False])
async def test_tool_result_deadline_is_quiet_while_the_tool_runs(running):
    finished = asyncio.Event()

    async def lookup(params):
        await finished.wait()
        await params.result_callback({"status": "paid"})

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
        parent = await peer.start()
        await emit_call(peer, parent)
        await peer.end(parent)
        await until(lambda: "call-1" in service._calls)
        if not running:
            finished.set()
            await until(lambda: service._calls["call-1"].submitted)
        await peer.emit(
            "error",
            error={
                "code": "tool_result_timeout",
                "message": "tool results did not arrive in time",
                "scope": "tool_roundtrip",
                "owner": {"kind": "response", "response_id": parent, "call_ids": ["call-1"]},
            },
        )
        await asyncio.sleep(0.05)
        errors = [f for f in upstream if isinstance(f, ErrorFrame)]
        assert bool(errors) is not running
        if running:
            finished.set()
            request = await peer.next()
            await answer(peer, request)
        assert service.is_usable


@pytest.mark.asyncio
@pytest.mark.parametrize("moved_on", [False, True])
async def test_async_result_after_its_round_closed_is_reported(moved_on):
    finished = asyncio.Event()

    async def lookup(params):
        await finished.wait()
        await params.result_callback({"status": "paid"})

    async with conversation(settings=RimeRealtimeLLMService.Settings(tools=[lookup])) as (
        peer,
        service,
        worker,
        *_,
    ):
        service.register_function("lookup", lookup, cancel_on_interruption=False)
        parent = await peer.start()
        await emit_call(peer, parent)
        await peer.end(parent)
        await until(lambda: "call-1" in service._calls)
        if moved_on:
            turn = await peer.start()
            await peer.end(turn)
            await until(lambda: turn in service._responses)
        finished.set()
        request = await peer.next()
        outputs = sent(peer, "conversation.item.create")
        assert [event["item"]["output"] for event in outputs] == ['{"status": "paid"}']
        if moved_on:
            assert request["response"]["metadata"]["prism_cause"] == "proactive"
            assert request["response"]["prism_call_id"] == "call-1"
        else:
            assert request["response"]["metadata"]["prism_cause"] == "tool_continuation"
        await answer(peer, request)
        await asyncio.sleep(0.1)
        assert len(sent(peer, "response.create")) == 2
        assert service.is_usable


async def late_result(peer, service, finished):
    """Close a running async call's round with a later turn, then finish the call."""
    parent = await peer.start()
    await emit_call(peer, parent)
    await peer.end(parent)
    await until(lambda: "call-1" in service._calls)
    turn = await peer.start()
    await peer.end(turn)
    await until(lambda: turn in service._responses)
    finished.set()


@pytest.mark.asyncio
async def test_late_result_report_survives_an_interruption():
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
        peer.gates["response.create"] = asyncio.Event()
        await late_result(peer, service, finished)
        await until(lambda: bool(service._requests))
        await worker.queue_frame(InterruptionFrame())
        await asyncio.sleep(0.05)
        peer.gates["response.create"].set()
        request = await peer.next()
        assert request["response"]["prism_call_id"] == "call-1"
        report = await peer.start(request)
        await peer.message(report, text="It is paid")
        await peer.end(report)
        # An interrupted reply would be stopped, and its words left out of the context.
        await until(
            lambda: {"role": "assistant", "content": "It is paid"} in context.get_messages()
        )
        assert not service._responses[report].stopped
        assert service.is_usable


@pytest.mark.asyncio
async def test_refused_late_result_report_is_requested_once_more():
    finished = asyncio.Event()

    async def lookup(params):
        await finished.wait()
        await params.result_callback({"status": "paid"})

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
        peer.refuse["response.create"] = "nudge_superseded"
        await late_result(peer, service, finished)
        await until(lambda: len(sent(peer, "response.create")) == 3)
        retry = sent(peer, "response.create")[-1]
        assert retry["response"]["prism_call_id"] == "call-1"
        await answer(peer, retry)
        assert not any(isinstance(f, ErrorFrame) for f in upstream)
        assert service.is_usable


@pytest.mark.asyncio
async def test_server_playback_admission_wait_does_not_block_microphone_frames():
    async with conversation() as (peer, service, worker, context, *_):
        response = await peer.start()
        await peer.message(response, text="Waiting for server playback estimate")
        await peer.end(response, ready=False)
        await until(lambda: bool(context.get_messages()))
        await worker.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Waiting"}], run_llm=True
                ),
                InputAudioRawFrame(b"\x00\x00" * 480, 24000, 1),
            ]
        )
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
            _, upstream = await run_test(
                service, frames_to_send=[LLMContextFrame(LLMContext()), SleepFrame(0.05)]
            )
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
        await asyncio.sleep(0.05)
        assert calls == []


@pytest.mark.asyncio
async def test_unexpected_connection_close_requires_explicit_recovery():
    async with conversation() as (peer, service, worker, _, __, upstream, ___):
        await peer.socket.close()
        await until(lambda: any(isinstance(f, ErrorFrame) for f in upstream))
        assert not service.is_usable
        assert len(sent(peer, "session.update")) == 1
