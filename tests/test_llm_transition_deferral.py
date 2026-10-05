"""Replay provider streams and playback/interruption ordering without network calls."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from google.genai import types

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    StopFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest_asyncio.fixture(params=["openai", "google"])
async def service(request):
    if request.param == "google":
        llm = GoogleLLMService(api_key="test-key")
    else:
        with patch.object(OpenAILLMService, "create_client"):
            llm = OpenAILLMService(api_key="test-key")
    await llm.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None, worker_runner=None),
        )
    )
    llm.push_frame = AsyncMock()
    llm.run_function_calls = AsyncMock()
    for name in ("end_call", "transfer_agent"):
        llm.register_function(name, AsyncMock(), is_node_transition=True)
    llm.register_function("save_booking", AsyncMock())
    yield llm
    await llm.cleanup()


async def respond(
    service, names, text="Your booking is confirmed.", during_stream=None, arguments=None
):
    if isinstance(service, GoogleLLMService):

        async def stream(context):
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=[types.Part(text=text)])
                    )
                ]
            )
            if during_stream:
                await during_stream()
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=name, id=f"call-{i}", args=arguments or {}
                                    )
                                )
                                for i, name in enumerate(names)
                            ],
                        )
                    )
                ]
            )

        service._stream_response = stream
    else:

        class Stream:
            def __aiter__(self):
                return self.iterate()

            async def iterate(self):
                yield SimpleNamespace(
                    usage=None,
                    model=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
                )
                if during_stream:
                    await during_stream()
                for i, name in enumerate(names):
                    yield SimpleNamespace(
                        usage=None,
                        model=None,
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=i,
                                            id=f"call-{i}",
                                            function=SimpleNamespace(
                                                name=name, arguments=json.dumps(arguments or {})
                                            ),
                                        )
                                    ],
                                )
                            )
                        ],
                    )

            async def close(self):
                pass

        service.get_chat_completions = AsyncMock(return_value=Stream())
    await service._process_context(LLMContext())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names",
    [
        ["save_booking"],
        ["save_booking", "end_call"],
        ["end_call", "save_booking"],
        ["end_call", "transfer_agent"],
    ],
)
async def test_only_a_single_transition_can_be_deferred(service, names):
    await respond(service, names)
    service.run_function_calls.assert_awaited_once()
    assert [c.function_name for c in service.run_function_calls.await_args.args[0]] == names
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert service.run_function_calls.await_count == 1


@pytest.mark.asyncio
async def test_single_transition_waits_for_normal_playback_completion(service):
    await respond(service, ["end_call"])
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["interruption", "interrupted_stop", "cancel", "end", "stop"])
async def test_interrupted_or_cancelled_transition_never_executes(service, first):
    await respond(service, ["end_call"])
    service.run_function_calls.assert_not_awaited()
    if first == "interrupted_stop":
        stopped = BotStoppedSpeakingFrame()
        stopped.interrupted = True
        await service.process_frame(stopped, FrameDirection.UPSTREAM)
    else:
        frame = {
            "interruption": InterruptionFrame,
            "cancel": CancelFrame,
            "end": EndFrame,
            "stop": StopFrame,
        }[first]()
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_interruption_during_stream_cannot_arm_a_late_transition(service):
    async def interrupt():
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    await respond(service, ["end_call"], during_stream=interrupt)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()
    # A fresh response can still decide to end the call.
    await respond(service, ["end_call"])
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "...", "\n"])
async def test_transition_without_speech_does_not_wait(service, text):
    await respond(service, ["end_call"], text=text)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_transition_without_tts_runs_immediately(service):
    await service.process_frame(LLMConfigureOutputFrame(skip_tts=True), FrameDirection.DOWNSTREAM)
    await respond(service, ["end_call"])
    service.run_function_calls.assert_awaited_once()

    service.run_function_calls.reset_mock()
    await service.process_frame(LLMConfigureOutputFrame(skip_tts=False), FrameDirection.DOWNSTREAM)
    await respond(service, ["end_call"])
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("old_names", [["end_call"], ["end_call", "save_booking"]])
async def test_fresh_response_does_not_revive_an_interrupted_stream(service, old_names):
    async def interrupt_and_answer_again():
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await respond(service, ["transfer_agent"])

    await respond(service, old_names, during_stream=interrupt_and_answer_again)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    dispatched = [
        fc.function_name
        for call in service.run_function_calls.await_args_list
        for fc in call.args[0]
    ]
    assert dispatched == (["save_booking"] if "save_booking" in old_names else []) + [
        "transfer_agent"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_during_stream", [False, True])
async def test_delayed_interrupted_stop_preserves_the_fresh_response(service, stop_during_stream):
    await respond(service, ["end_call"])
    # The service and output transport may receive opposite broadcast siblings.
    upstream, downstream = InterruptionFrame(), InterruptionFrame()
    upstream.broadcast_sibling_id = downstream.id
    downstream.broadcast_sibling_id = upstream.id
    await service.process_frame(upstream, FrameDirection.UPSTREAM)
    stopped = BotStoppedSpeakingFrame(interrupted=True)
    stopped.interruption_id = min(upstream.id, downstream.id)

    async def delayed_stop():
        await service.process_frame(stopped, FrameDirection.UPSTREAM)

    await respond(
        service,
        ["transfer_agent"],
        during_stream=delayed_stop if stop_during_stream else None,
    )
    if not stop_during_stream:
        await delayed_stop()
    service.run_function_calls.assert_not_awaited()
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()
    assert service.run_function_calls.await_args.args[0][0].function_name == "transfer_agent"


@pytest.mark.asyncio
async def test_late_suppressed_transition_is_logged_without_arguments(service):
    async def interrupt():
        await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    sentinel = "private-booking-details-must-not-be-logged"
    with patch("pipecat.services.llm_service.logger") as logger:
        await respond(
            service, ["end_call"], during_stream=interrupt, arguments={"booking": sentinel}
        )
    service.run_function_calls.assert_not_awaited()
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert any(
        "end_call" in message and "call-0" in message and "interrupted" in message
        for message in messages
    )
    assert sentinel not in str(logger.info.call_args_list)


@pytest.mark.asyncio
async def test_each_new_interruption_cancels_only_its_own_response(service):
    await respond(service, ["end_call"])
    first = InterruptionFrame()
    await service.process_frame(first, FrameDirection.DOWNSTREAM)
    await respond(service, ["transfer_agent"])
    second = InterruptionFrame()
    await service.process_frame(second, FrameDirection.DOWNSTREAM)
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_not_awaited()

    await respond(service, ["end_call"])
    for interruption in (second, first):
        await service.process_frame(
            BotStoppedSpeakingFrame(interrupted=True, interruption_id=interruption.id),
            FrameDirection.UPSTREAM,
        )
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("more_speech", [None, "text", "playback"])
async def test_playback_can_finish_before_the_transition_is_parsed(service, more_speech):
    async def finish_speech():
        await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        if more_speech == "text":
            await service._push_llm_text("One more sentence.")
        elif more_speech == "playback":
            await service.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    await respond(service, ["end_call"], during_stream=finish_speech)
    if more_speech:
        service.run_function_calls.assert_not_awaited()
        await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    service.run_function_calls.assert_awaited_once()

    # A completed response's playback state must not release the next one.
    service.run_function_calls.reset_mock()
    await respond(service, ["transfer_agent"])
    service.run_function_calls.assert_not_awaited()
