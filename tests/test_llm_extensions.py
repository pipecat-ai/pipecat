#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how an LLM service runs its LLM extensions."""

import unittest
from collections.abc import Callable

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    TextFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.llm_extensions.base_llm_extension import BaseLLMExtension, LLMExtensionTool
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _MockLLMService(LLMService):
    """Minimal LLM service that pushes every frame onward, as a concrete service does."""

    def __init__(self, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=kwargs.pop("system_instruction", None),
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _Recorder(BaseLLMExtension):
    """An extension that records what it sees and does what it is told."""

    def __init__(
        self,
        *,
        order: int = 0,
        instruction: str | None = None,
        filter: Callable[[Frame], list[Frame]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.order = order
        self._instruction = instruction
        self._filter = filter
        self.processed: list[tuple[Frame, FrameDirection]] = []
        self.filtered: list[Frame] = []
        self.set_up = False
        self.cleaned_up = False

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        self.set_up = True

    async def cleanup(self):
        await super().cleanup()
        self.cleaned_up = True

    @property
    def system_instruction(self) -> str | None:
        return self._instruction

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        self.processed.append((frame, direction))

    async def filter_push(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        self.filtered.append(frame)
        return self._filter(frame) if self._filter else [frame]


class _Installed(BaseLLMExtension):
    installed_by = "SomeStrategy"


def _append(suffix: str) -> Callable[[Frame], list[Frame]]:
    """A filter that appends ``suffix`` to the text of every LLMTextFrame."""

    def filter(frame: Frame) -> list[Frame]:
        if isinstance(frame, LLMTextFrame):
            return [LLMTextFrame(frame.text + suffix)]
        return [frame]

    return filter


def _texts(frames: list[Frame]) -> list[str]:
    return [frame.text for frame in frames if isinstance(frame, LLMTextFrame)]


class TestLLMExtensions(unittest.IsolatedAsyncioTestCase):
    async def test_extensions_see_every_frame_the_service_processes(self):
        extension = _Recorder()
        service = _MockLLMService(extensions=[extension])
        text = TextFrame("hi")

        await run_test(service, frames_to_send=[text], ignore_start=False)

        self.assertEqual(
            [(type(frame), direction) for frame, direction in extension.processed],
            [
                (StartFrame, FrameDirection.DOWNSTREAM),
                (TextFrame, FrameDirection.DOWNSTREAM),
                (EndFrame, FrameDirection.DOWNSTREAM),
            ],
        )

    async def test_extensions_filter_what_the_service_pushes_in_order(self):
        later = _Recorder(order=1, filter=_append("-later"))
        earlier = _Recorder(order=0, filter=_append("-earlier"))
        # Attached in the other order: the order decides.
        service = _MockLLMService(extensions=[later, earlier])

        received_down, _ = await run_test(
            service,
            frames_to_send=[LLMTextFrame("x")],
            expected_down_frames=[LLMServiceMetadataFrame, LLMTextFrame],
        )

        self.assertEqual(_texts(received_down), ["x-earlier-later"])

    async def test_frames_an_extension_adds_continue_through_the_ones_after_it(self):
        def add_one(frame: Frame) -> list[Frame]:
            if isinstance(frame, LLMTextFrame):
                return [frame, LLMTextFrame("added")]
            return [frame]

        adder = _Recorder(order=0, filter=add_one)
        after = _Recorder(order=1)
        service = _MockLLMService(extensions=[adder, after])

        received_down, _ = await run_test(
            service,
            frames_to_send=[LLMTextFrame("x")],
            expected_down_frames=[LLMServiceMetadataFrame, LLMTextFrame, LLMTextFrame],
        )

        self.assertEqual(_texts(after.filtered), ["x", "added"])
        self.assertEqual(_texts(received_down), ["x", "added"])

    async def test_frames_go_out_in_the_order_the_extensions_produce_them(self):
        """A frame one extension adds goes through the rest and out before the next one."""

        def split(frame: Frame) -> list[Frame]:
            if isinstance(frame, LLMTextFrame):
                return [LLMTextFrame("first"), LLMTextFrame("second")]
            return [frame]

        class Announcer(BaseLLMExtension):
            async def filter_push(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
                if isinstance(frame, LLMTextFrame) and frame.text == "first":
                    await self.push_frame(LLMTextFrame("announced"))
                return [frame]

        service = _MockLLMService(extensions=[_Recorder(filter=split), Announcer()])

        received_down, _ = await run_test(
            service,
            frames_to_send=[LLMTextFrame("x")],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMTextFrame,
                LLMTextFrame,
                LLMTextFrame,
            ],
        )

        self.assertEqual(_texts(received_down), ["announced", "first", "second"])

    async def test_an_extension_can_hold_a_frame_back(self):
        dropper = _Recorder(filter=lambda frame: [] if isinstance(frame, LLMTextFrame) else [frame])
        service = _MockLLMService(extensions=[dropper])

        received_down, _ = await run_test(
            service,
            frames_to_send=[LLMTextFrame("x")],
            expected_down_frames=[LLMServiceMetadataFrame],
        )

        self.assertEqual(_texts(received_down), [])

    async def test_what_an_extension_pushes_goes_out_as_it_is(self):
        class Speaker(BaseLLMExtension):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, TextFrame):
                    await self.push_frame(LLMTextFrame("spoken"))
                    await self.push_frame(LLMTextFrame("up"), FrameDirection.UPSTREAM)
                    await self.broadcast_frame(UserTurnInferenceCompletedFrame)

        # A filter that would drop the text, were it filtered.
        dropper = _Recorder(filter=lambda frame: [] if isinstance(frame, LLMTextFrame) else [frame])
        service = _MockLLMService(extensions=[Speaker(), dropper])

        received_down, received_up = await run_test(
            service,
            frames_to_send=[TextFrame("hi")],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMTextFrame,
                UserTurnInferenceCompletedFrame,
                TextFrame,
            ],
            expected_up_frames=[
                LLMServiceMetadataFrame,
                LLMTextFrame,
                UserTurnInferenceCompletedFrame,
            ],
        )

        self.assertEqual(_texts(received_down), ["spoken"])
        self.assertEqual(_texts(received_up), ["up"])
        self.assertEqual(_texts(dropper.filtered), [])

    async def test_extensions_are_set_up_and_cleaned_up_with_the_service(self):
        extension = _Recorder()
        service = _MockLLMService(extensions=[extension])

        await run_test(service, frames_to_send=[])

        self.assertTrue(extension.set_up)
        self.assertTrue(extension.cleaned_up)
        # The service's handlers are gone with it.
        self.assertEqual(extension._event_handlers["on_push_frame"].handlers, [])

    async def test_an_extension_added_in_hand_is_set_up_with_the_service(self):
        service = _MockLLMService(system_instruction="APP")
        extension = _Recorder(instruction="ADDED")

        service.add_extension(extension)
        await service.setup(frame_processor_setup(TaskManager()))

        self.assertTrue(extension.set_up)
        service._compose_system_instruction()
        self.assertEqual(service._settings.system_instruction, "APP\n\nADDED")
        await service.cleanup()
        self.assertTrue(extension.cleaned_up)

    async def test_adding_an_extension_to_a_running_service_is_refused(self):
        service = _MockLLMService()
        await service.setup(frame_processor_setup(TaskManager()))

        with self.assertRaises(RuntimeError):
            service.add_extension(_Recorder())
        await service.cleanup()

    async def test_instructions_compose_after_the_base_in_order(self):
        service = _MockLLMService(
            system_instruction="APP",
            extensions=[
                _Recorder(order=1, instruction="LATER"),
                _Recorder(order=0, instruction="EARLIER"),
                _Recorder(order=0),
            ],
        )

        service._compose_system_instruction()

        self.assertEqual(service._settings.system_instruction, "APP\n\nEARLIER\n\nLATER")

    async def test_an_extensions_tools_are_advertised_and_withdrawn_with_the_registry(self):
        """A tool an extension returns is built in; one it stops returning goes."""
        calls = []

        async def handler(params):
            calls.append(params.function_name)
            await params.result_callback("ok")

        class Companion(BaseLLMExtension):
            """Adds a tool for every registered tool whose name starts with "go"."""

            def builtin_tools(self, registered):
                return [
                    LLMExtensionTool(
                        FunctionSchema(
                            name=f"stop_{name}",
                            description="",
                            properties={},
                            required=[],
                            handler=handler,
                        ),
                        cancel_on_interruption=False,
                    )
                    for name in registered
                    if name and name.startswith("go")
                ]

        service = _MockLLMService(extensions=[Companion()])
        service.register_function("go_fetch", handler)
        service._sync_registered_tool_handlers(NOT_GIVEN)

        self.assertEqual(set(service.get_llm_adapter().builtin_tools), {"stop_go_fetch"})
        self.assertFalse(service._functions["stop_go_fetch"].cancel_on_interruption)
        # The name is reserved while the tool is built in.
        with self.assertRaises(ValueError):
            service.register_function("stop_go_fetch", handler)

        service.unregister_function("go_fetch")
        service._sync_registered_tool_handlers(NOT_GIVEN)

        self.assertEqual(set(service.get_llm_adapter().builtin_tools), set())
        self.assertNotIn("stop_go_fetch", service._functions)

    async def test_a_users_tool_keeps_its_name_over_an_extensions(self):
        async def handler(params):
            pass

        class Companion(BaseLLMExtension):
            def builtin_tools(self, registered):
                return [
                    LLMExtensionTool(
                        FunctionSchema(
                            name="helper",
                            description="",
                            properties={},
                            required=[],
                            handler=handler,
                        )
                    )
                ]

        service = _MockLLMService(extensions=[Companion()])
        service.register_function("helper", handler)
        service._sync_registered_tool_handlers(NOT_GIVEN)

        self.assertEqual(set(service.get_llm_adapter().builtin_tools), set())
        self.assertIs(service._functions["helper"].handler, handler)

    async def test_the_constructor_refuses_an_extension_another_component_installs(self):
        with self.assertRaises(ValueError) as ctx:
            _MockLLMService(extensions=[_Installed()])
        self.assertIn("SomeStrategy", str(ctx.exception))

    async def test_a_settings_update_builds_and_adds_extensions(self):
        first = _Recorder(instruction="FIRST")
        service = _MockLLMService(extensions=[first])

        await run_test(
            service,
            frames_to_send=[
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(extensions=[lambda: _Recorder(instruction="SECOND")])
                )
            ],
        )

        recorders = [e for e in service._extensions if isinstance(e, _Recorder)]
        self.assertEqual(len(recorders), 2)
        self.assertIs(recorders[0], first)
        second = recorders[1]
        self.assertTrue(second.set_up)
        self.assertEqual(service._settings.system_instruction, "FIRST\n\nSECOND")

    async def test_an_extension_in_a_settings_update_is_refused(self):
        service = _MockLLMService()

        with self.assertRaises(TypeError) as ctx:
            await service._update_settings(LLMSettings(extensions=[_Recorder()]))
        self.assertIn("add_extension", str(ctx.exception))

    async def test_a_settings_update_builds_an_extension_per_service(self):
        delta = LLMSettings(extensions=[_Recorder])
        one = _MockLLMService()
        another = _MockLLMService()

        await one._update_settings(delta)
        await another._update_settings(delta)

        (built_for_one,) = [e for e in one._extensions if isinstance(e, _Recorder)]
        (built_for_another,) = [e for e in another._extensions if isinstance(e, _Recorder)]
        self.assertIsNot(built_for_one, built_for_another)


if __name__ == "__main__":
    unittest.main()
