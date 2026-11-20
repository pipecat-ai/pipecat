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
from pipecat.services.base_llm_extension import BaseLLMExtension, LLMExtensionTool
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
            extensions=kwargs.pop("extensions", []),
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

    async def filter_push_frame(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        self.filtered.append(frame)
        return self._filter(frame) if self._filter else [frame]


class _Alpha(_Recorder):
    """A recorder of a class of its own, for a service that holds several: one of each class."""


class _Beta(_Recorder):
    """A recorder of a class of its own, for a service that holds several: one of each class."""


class _Gamma(_Recorder):
    """A recorder of a class of its own, for a service that holds several: one of each class."""


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
        later = _Beta(order=1, filter=_append("-later"))
        earlier = _Alpha(order=0, filter=_append("-earlier"))
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

        adder = _Alpha(order=0, filter=add_one)
        after = _Beta(order=1)
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
            async def filter_push_frame(
                self, frame: Frame, direction: FrameDirection
            ) -> list[Frame]:
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

    async def test_what_an_extension_pushes_passes_the_extensions_outside_it(self):
        class Speaker(BaseLLMExtension):
            order = 1

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, TextFrame):
                    await self.push_frame(LLMTextFrame("spoken"))
                    await self.push_frame(LLMTextFrame("up"), FrameDirection.UPSTREAM)
                    await self.broadcast_frame(UserTurnInferenceCompletedFrame)

        # A filter that would drop the text, were it handed it.
        inside = _Alpha(
            order=0, filter=lambda frame: [] if isinstance(frame, LLMTextFrame) else [frame]
        )
        outside = _Beta(order=2, filter=_append("-outside"))
        service = _MockLLMService(extensions=[inside, Speaker(), outside])

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

        self.assertEqual(_texts(received_down), ["spoken-outside"])
        self.assertEqual(_texts(received_up), ["up-outside"])
        self.assertEqual(_texts(inside.filtered), [])
        self.assertEqual(_texts(outside.filtered), ["spoken", "up"])
        self.assertEqual(
            len([f for f in outside.filtered if isinstance(f, UserTurnInferenceCompletedFrame)]), 2
        )
        self.assertEqual(
            [f for f in inside.filtered if isinstance(f, UserTurnInferenceCompletedFrame)], []
        )

    async def test_extensions_process_a_frame_from_the_outside_in(self):
        """Each extension processes a frame before it filters any frame pushed in response."""
        log: list[str] = []

        class Responder(BaseLLMExtension):
            """Answers a TextFrame with a frame of its own, and logs what it is handed."""

            def __init__(self, name: str, order: int):
                super().__init__()
                self._name = name
                self.order = order

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, TextFrame):
                    log.append(f"{self._name} processes")
                    await self.push_frame(LLMTextFrame(f"from {self._name}"))

            async def filter_push_frame(
                self, frame: Frame, direction: FrameDirection
            ) -> list[Frame]:
                if isinstance(frame, LLMTextFrame):
                    log.append(f"{self._name} filters {frame.text}")
                return [frame]

        class A(Responder):
            pass

        class B(Responder):
            pass

        class C(Responder):
            pass

        service = _MockLLMService(extensions=[A("A", 0), B("B", 1), C("C", 2)])

        received_down, _ = await run_test(service, frames_to_send=[TextFrame("hi")])

        self.assertEqual(
            log,
            [
                "C processes",
                "B processes",
                "C filters from B",
                "A processes",
                "B filters from A",
                "C filters from A",
            ],
        )
        self.assertEqual(_texts(received_down), ["from C", "from B", "from A"])

    async def test_what_an_extension_pushes_is_past_the_output_gate(self):
        class ClosedGate:
            """Stands in for the gate predictive delegation installs: lets no text out."""

            async def process_frame(self, service, frame):
                pass

            def allow_push(self, frame):
                return not isinstance(frame, LLMTextFrame)

            def allow_function_calls(self):
                return True

        class Speaker(BaseLLMExtension):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, UserTurnInferenceCompletedFrame):
                    await self.push_frame(LLMTextFrame("from the extension"))

        service = _MockLLMService(extensions=[Speaker()])
        service._output_gate = ClosedGate()  # type: ignore[assignment]

        received_down, _ = await run_test(
            service,
            frames_to_send=[UserTurnInferenceCompletedFrame(), LLMTextFrame("from the service")],
        )

        self.assertEqual(_texts(received_down), ["from the extension"])

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
                _Alpha(order=1, instruction="LATER"),
                _Beta(order=0, instruction="EARLIER"),
                _Gamma(order=0),
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

    async def test_a_settings_update_builds_and_adds_extensions(self):
        first = _Alpha(instruction="FIRST")
        service = _MockLLMService(extensions=[first])

        await run_test(
            service,
            frames_to_send=[
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(extensions=[lambda: _Beta(instruction="SECOND")])
                )
            ],
        )

        self.assertEqual(len(service._extensions), 2)
        self.assertIs(service._extensions[0], first)
        second = service._extensions[1]
        self.assertIsInstance(second, _Beta)
        self.assertTrue(second.set_up)
        self.assertEqual(service._settings.system_instruction, "FIRST\n\nSECOND")

    async def test_get_extension_finds_the_attached_extension_of_a_class(self):
        alpha = _Alpha()
        service = _MockLLMService(extensions=[alpha])

        self.assertIs(service.get_extension(_Alpha), alpha)
        self.assertIsNone(service.get_extension(_Recorder))
        self.assertIsNone(service.get_extension(_Beta))

    async def test_a_second_extension_of_a_class_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _MockLLMService(extensions=[_Recorder(), _Recorder()])
        self.assertIn("subclass", str(ctx.exception))

        service = _MockLLMService(extensions=[_Recorder()])
        with self.assertRaises(ValueError):
            service.add_extension(_Recorder())
        with self.assertRaises(ValueError):
            await service._update_settings(LLMSettings(extensions=[_Recorder]))
        self.assertEqual(
            [e for e in service._extensions if isinstance(e, _Recorder)], service._extensions[:1]
        )

    async def test_constructor_settings_build_a_callable_entry(self):
        service = _MockLLMService(extensions=[_Recorder])

        self.assertIsInstance(service.get_extension(_Recorder), _Recorder)

    async def test_a_settings_update_takes_an_extension_in_hand(self):
        service = _MockLLMService()
        extension = _Recorder()

        await service._update_settings(LLMSettings(extensions=[extension]))

        self.assertIs(service.get_extension(_Recorder), extension)

    async def test_an_extension_belongs_to_one_service(self):
        """A second service that an update with an extension in hand reaches refuses it."""
        extension = _Recorder()
        one = _MockLLMService()
        another = _MockLLMService()
        await one.setup(frame_processor_setup(TaskManager()))
        await another.setup(frame_processor_setup(TaskManager()))

        await one._update_settings(LLMSettings(extensions=[extension]))
        self.assertIs(extension.attached_service, one)
        with self.assertRaises(ValueError):
            await another._update_settings(LLMSettings(extensions=[extension]))

        await one.cleanup()
        await another.cleanup()

    async def test_a_settings_update_builds_an_extension_per_service(self):
        delta = LLMSettings(extensions=[_Recorder])
        one = _MockLLMService()
        another = _MockLLMService()

        await one._update_settings(delta)
        await another._update_settings(delta)

        (built_for_one,) = one._extensions
        (built_for_another,) = another._extensions
        self.assertIsInstance(built_for_one, _Recorder)
        self.assertIsNot(built_for_one, built_for_another)


if __name__ == "__main__":
    unittest.main()
