#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings
from functools import partial

from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMarkerFrame,
    LLMMarkerResponseFrame,
    LLMTextFrame,
    UserTurnCompletionFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_extensions.user_turn_completion import UserTurnCompletionLLMExtension
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_completion import (
    USER_TURN_COMPLETE_MARKER,
    USER_TURN_COMPLETION_INSTRUCTIONS,
    USER_TURN_INCOMPLETE_LONG_MARKER,
    USER_TURN_INCOMPLETE_SHORT_MARKER,
    TurnMarker,
    UserTurnCompletionConfig,
)


class _Reader:
    """Runs the extension the way its service does, recording what goes out.

    ``pushed`` holds every frame in order, with its direction: what the
    extension lets through or adds in ``filter_push``, and what it pushes or
    broadcasts itself.
    """

    def __init__(self, config: UserTurnCompletionConfig | None = None):
        self.extension = UserTurnCompletionLLMExtension(config)
        self.pushed: list[tuple[Frame, FrameDirection]] = []
        self.extension.add_event_handler("on_push_frame", self._on_push_frame)
        self.extension.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)

    async def _on_push_frame(self, _, frame: Frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))

    async def _on_broadcast_frame(self, _, frame_cls: type[Frame], **kwargs):
        self.pushed.append((frame_cls(**kwargs), FrameDirection.DOWNSTREAM))
        self.pushed.append((frame_cls(**kwargs), FrameDirection.UPSTREAM))

    async def push(self, frame: Frame):
        """Push a frame as the service would, through the extension's filter."""
        for filtered in await self.extension.filter_push(frame, FrameDirection.DOWNSTREAM):
            self.pushed.append((filtered, FrameDirection.DOWNSTREAM))

    async def text(self, text: str):
        await self.push(LLMTextFrame(text))

    async def process(self, frame: Frame):
        await self.extension.process_frame(frame, FrameDirection.DOWNSTREAM)

    @property
    def frames(self) -> list[Frame]:
        return [frame for frame, _ in self.pushed]

    def of(self, frame_cls: type[Frame]) -> list:
        return [frame for frame in self.frames if isinstance(frame, frame_cls)]


class TestUserTurnCompletionLLMExtension(unittest.IsolatedAsyncioTestCase):
    """Tests for reading the marker from the model's text."""

    async def test_complete_marker_pushes_text(self):
        """Test that ● marker is detected and text after it is pushed normally."""
        reader = _Reader()

        # Simulate LLM generating: "● Hello there!"
        await reader.text(f"{USER_TURN_COMPLETE_MARKER} Hello there!")

        # The marker rides as LLMMarkerFrame(append_to_context_immediately=False);
        # only the spoken text is pushed as an LLMTextFrame.
        text_frames = reader.of(LLMTextFrame)
        self.assertEqual(len(text_frames), 1)
        self.assertEqual(text_frames[0].text, "Hello there!")
        self.assertFalse(text_frames[0].skip_tts)

        marker_frames = reader.of(LLMMarkerFrame)
        self.assertEqual(len(marker_frames), 1)
        self.assertEqual(marker_frames[0].marker, USER_TURN_COMPLETE_MARKER)
        self.assertFalse(marker_frames[0].append_to_context_immediately)

        # UserTurnInferenceCompletedFrame broadcast in both directions.
        self.assertEqual(len(reader.of(UserTurnInferenceCompletedFrame)), 2)

    async def test_incomplete_short_marker_suppresses_text(self):
        """Test that ◐ marker suppresses text and is emitted as a stand-alone marker frame."""
        reader = _Reader()

        await reader.text(USER_TURN_INCOMPLETE_SHORT_MARKER)

        # No LLMTextFrame: response is suppressed.
        self.assertEqual(reader.of(LLMTextFrame), [])

        marker_frames = reader.of(LLMMarkerFrame)
        self.assertEqual(len(marker_frames), 1)
        self.assertEqual(marker_frames[0].marker, USER_TURN_INCOMPLETE_SHORT_MARKER)
        self.assertTrue(marker_frames[0].append_to_context_immediately)

        # Incomplete markers do not emit UserTurnInferenceCompletedFrame.
        self.assertEqual(reader.of(UserTurnInferenceCompletedFrame), [])

    async def test_incomplete_long_marker_suppresses_text(self):
        """Test that ○ marker suppresses text and is emitted as a stand-alone marker frame."""
        reader = _Reader()

        await reader.text(USER_TURN_INCOMPLETE_LONG_MARKER)

        self.assertEqual(reader.of(LLMTextFrame), [])

        marker_frames = reader.of(LLMMarkerFrame)
        self.assertEqual(len(marker_frames), 1)
        self.assertEqual(marker_frames[0].marker, USER_TURN_INCOMPLETE_LONG_MARKER)
        self.assertTrue(marker_frames[0].append_to_context_immediately)

        self.assertEqual(reader.of(UserTurnInferenceCompletedFrame), [])

    async def test_text_buffered_until_marker_found(self):
        """Test that text is buffered until a marker is detected."""
        reader = _Reader()

        # Simulate token-by-token streaming without marker
        await reader.text("Hello")
        await reader.text(" there")

        # No frames should be pushed yet (buffering)
        self.assertEqual(reader.frames, [])

        # Now send the complete marker
        await reader.text(f" {USER_TURN_COMPLETE_MARKER} How are you?")

        # One LLMTextFrame for the spoken portion; one LLMMarkerFrame for
        # the marker.
        self.assertEqual(len(reader.of(LLMTextFrame)), 1)
        self.assertEqual(len(reader.of(LLMMarkerFrame)), 1)

    async def test_turn_state_reset_after_llm_full_response_end_frame(self):
        """Test that the turn marker is reset when LLMFullResponseEndFrame is pushed."""
        reader = _Reader()

        # Simulate first LLM response: complete marker sets the marker to COMPLETE
        await reader.text(f"{USER_TURN_COMPLETE_MARKER} Hello!")
        self.assertEqual(reader.extension._turn_marker, TurnMarker.COMPLETE)

        await reader.push(LLMFullResponseEndFrame())

        # The marker must now be cleared — ready for the next response
        self.assertIsNone(reader.extension._turn_marker)
        self.assertEqual(reader.extension._turn_text_buffer, "")

    async def test_response_end_reports_the_raw_text_and_marker(self):
        """When the response ends, the extension reports what the LLM wrote and the marker it read."""
        reader = _Reader()
        await reader.push(LLMFullResponseStartFrame())
        await reader.text("Well, ")
        await reader.text(f"{USER_TURN_INCOMPLETE_SHORT_MARKER} go on")
        await reader.text(" please")
        await reader.push(LLMFullResponseEndFrame())

        reports = reader.of(LLMMarkerResponseFrame)
        self.assertEqual(len(reports), 1)
        report = reports[0]
        # Everything the LLM produced, including what the extension held back.
        self.assertEqual(report.raw, f"Well, {USER_TURN_INCOMPLETE_SHORT_MARKER} go on please")
        self.assertEqual((report.marker, report.kind), (USER_TURN_INCOMPLETE_SHORT_MARKER, "short"))
        self.assertEqual(
            report.markers,
            [
                USER_TURN_COMPLETE_MARKER,
                USER_TURN_INCOMPLETE_SHORT_MARKER,
                USER_TURN_INCOMPLETE_LONG_MARKER,
            ],
        )
        # The report precedes the end-of-response frame.
        frames = reader.frames
        self.assertLess(
            frames.index(report),
            next(i for i, f in enumerate(frames) if isinstance(f, LLMFullResponseEndFrame)),
        )

    async def test_a_response_without_a_marker_is_reported_as_such(self):
        reader = _Reader()
        await reader.push(LLMFullResponseStartFrame())
        await reader.text("Hello there")
        await reader.push(LLMFullResponseEndFrame())

        (report,) = reader.of(LLMMarkerResponseFrame)
        self.assertEqual((report.raw, report.marker, report.kind), ("Hello there", None, None))

    async def test_a_response_the_extension_never_saw_is_not_reported(self):
        reader = _Reader()
        await reader.push(LLMFullResponseStartFrame())
        await reader.push(LLMFullResponseEndFrame())
        self.assertEqual(reader.of(LLMMarkerResponseFrame), [])

    async def test_text_after_a_complete_marker_streams_straight_through(self):
        reader = _Reader()

        await reader.text(f"{USER_TURN_COMPLETE_MARKER} Hello")
        await reader.text(" there!")

        self.assertEqual([f.text for f in reader.of(LLMTextFrame)], ["Hello", " there!"])

    async def test_unmarked_text_is_pushed_when_the_response_ends(self):
        """Text the model wrote without a marker is not lost: it goes out at the end."""
        reader = _Reader()
        await reader.push(LLMFullResponseStartFrame())
        await reader.text("Hello there")
        self.assertEqual(reader.of(LLMTextFrame), [])

        await reader.push(LLMFullResponseEndFrame())

        self.assertEqual([f.text for f in reader.of(LLMTextFrame)], ["Hello there"])


class MockLLMService(LLMService):
    """Minimal LLM service for testing system_instruction composition."""

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


async def _enable_turn_completion(
    service: LLMService, config: UserTurnCompletionConfig | None = None
) -> UserTurnCompletionLLMExtension:
    """Install the extension the way the stop strategy does: over a settings update."""
    await service._update_settings(
        LLMSettings(extensions=[partial(UserTurnCompletionLLMExtension, config)])
    )
    (extension,) = [e for e in service._extensions if isinstance(e, UserTurnCompletionLLMExtension)]
    return extension


class TestSystemInstructionComposition(unittest.IsolatedAsyncioTestCase):
    """Tests for turn completion system_instruction composition in LLMService."""

    async def test_enable_turn_completion_sets_system_instruction(self):
        """Enabling turn completion should set system_instruction to completion instructions."""
        service = MockLLMService()
        self.assertIsNone(service._settings.system_instruction)

        await _enable_turn_completion(service)

        self.assertEqual(service._settings.system_instruction, USER_TURN_COMPLETION_INSTRUCTIONS)
        self.assertIsNone(service._base_system_instruction)

    async def test_enable_turn_completion_appends_to_existing_system_instruction(self):
        """Enabling turn completion should append instructions to existing system_instruction."""
        service = MockLLMService(system_instruction="You are a helpful assistant.")

        await _enable_turn_completion(service)

        expected = f"You are a helpful assistant.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)
        self.assertEqual(service._base_system_instruction, "You are a helpful assistant.")

    async def test_update_system_instruction_while_turn_completion_active(self):
        """Changing system_instruction while turn completion is active should recompose."""
        service = MockLLMService(system_instruction="Original prompt.")

        await _enable_turn_completion(service)
        expected = f"Original prompt.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)

        # Now update system_instruction
        await service._update_settings(LLMSettings(system_instruction="New prompt."))
        expected = f"New prompt.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)
        self.assertEqual(service._base_system_instruction, "New prompt.")

    async def test_custom_instructions_compose_in_place_of_the_default(self):
        service = MockLLMService(system_instruction="Base prompt.")

        await _enable_turn_completion(
            service, UserTurnCompletionConfig(instructions="Custom turn instructions.")
        )

        self.assertEqual(
            service._settings.system_instruction, "Base prompt.\n\nCustom turn instructions."
        )

    async def test_simultaneous_enable_and_system_instruction_change(self):
        """Enabling turn completion and changing system_instruction in the same delta
        should use the new system_instruction as the base."""
        service = MockLLMService(system_instruction="Original prompt.")

        await service._update_settings(
            LLMSettings(
                extensions=[UserTurnCompletionLLMExtension],
                system_instruction="New prompt.",
            )
        )

        # apply_update sets system_instruction to "New prompt." before _update_settings
        # runs, so the base should be the new value the user explicitly set.
        self.assertEqual(service._base_system_instruction, "New prompt.")
        expected = f"New prompt.\n\n{USER_TURN_COMPLETION_INSTRUCTIONS}"
        self.assertEqual(service._settings.system_instruction, expected)


class TestDeprecatedTurnCompletionSettings(unittest.IsolatedAsyncioTestCase):
    """The deprecated settings fields still switch the extension on and off, with a warning."""

    async def test_the_flag_attaches_and_detaches_the_extension(self):
        service = MockLLMService(system_instruction="You are a helpful assistant.")

        with self.assertWarns(DeprecationWarning):
            await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))
        self.assertIsInstance(service._extensions[0], UserTurnCompletionLLMExtension)
        self.assertIn(USER_TURN_COMPLETION_INSTRUCTIONS, service._settings.system_instruction)

        with self.assertWarns(DeprecationWarning):
            await service._update_settings(LLMSettings(filter_incomplete_user_turns=False))
        self.assertFalse(
            any(isinstance(e, UserTurnCompletionLLMExtension) for e in service._extensions)
        )
        self.assertEqual(service._settings.system_instruction, "You are a helpful assistant.")
        # The base prompt is retained — it's the single source of truth that
        # composition rebuilds from; disabling just recomposes without the
        # turn-completion addon.
        self.assertEqual(service._base_system_instruction, "You are a helpful assistant.")

    async def test_disable_turn_completion_restores_none(self):
        service = MockLLMService()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await service._update_settings(LLMSettings(filter_incomplete_user_turns=True))
            self.assertEqual(
                service._settings.system_instruction, USER_TURN_COMPLETION_INSTRUCTIONS
            )
            await service._update_settings(LLMSettings(filter_incomplete_user_turns=False))
        self.assertIsNone(service._settings.system_instruction)

    async def test_the_config_field_reconfigures_the_extension(self):
        service = MockLLMService(system_instruction="Base prompt.")
        extension = await _enable_turn_completion(service)

        custom_config = UserTurnCompletionConfig(instructions="Custom turn instructions.")
        with self.assertWarns(DeprecationWarning):
            await service._update_settings(LLMSettings(user_turn_completion_config=custom_config))

        self.assertIs(extension.config, custom_config)
        self.assertEqual(
            service._settings.system_instruction, "Base prompt.\n\nCustom turn instructions."
        )


class TestConfigurableMarkers(unittest.IsolatedAsyncioTestCase):
    """Markers are configurable, and the prompts follow whatever is configured."""

    def test_default_markers_are_the_fill_gradient(self):
        """Full is a finished turn, half is cut off mid-thought, empty is not started."""
        self.assertEqual(USER_TURN_COMPLETE_MARKER, "●")
        self.assertEqual(USER_TURN_INCOMPLETE_SHORT_MARKER, "◐")
        self.assertEqual(USER_TURN_INCOMPLETE_LONG_MARKER, "○")

    def test_prompts_follow_configured_markers(self):
        """Custom markers reach the instructions and both re-prompts."""
        config = UserTurnCompletionConfig(
            complete_marker="Y", incomplete_short_marker="N", incomplete_long_marker="W"
        )

        for prompt in (config.completion_instructions, config.short_prompt, config.long_prompt):
            for default in (
                USER_TURN_COMPLETE_MARKER,
                USER_TURN_INCOMPLETE_SHORT_MARKER,
                USER_TURN_INCOMPLETE_LONG_MARKER,
            ):
                self.assertNotIn(default, prompt)

        self.assertIn("Write Y, a space, then your full reply", config.completion_instructions)
        self.assertIn("Respond with only N", config.completion_instructions)
        self.assertIn("Respond with only W", config.completion_instructions)
        self.assertIn("respond with Y", config.short_prompt)
        self.assertIn("respond with Y", config.long_prompt)

    def test_custom_instructions_still_win(self):
        """An explicit instructions string overrides the rendered default."""
        config = UserTurnCompletionConfig(instructions="do it my way")
        self.assertEqual(config.completion_instructions, "do it my way")

    async def test_parser_detects_configured_markers(self):
        """The detector matches the configured markers, not the defaults."""
        reader = _Reader(
            UserTurnCompletionConfig(
                complete_marker="Y", incomplete_short_marker="N", incomplete_long_marker="W"
            )
        )

        await reader.text("Y Hello there!")

        self.assertEqual([f.text for f in reader.of(LLMTextFrame)], ["Hello there!"])
        self.assertEqual([f.marker for f in reader.of(LLMMarkerFrame)], ["Y"])


class TestUserTurnCompletionReports(unittest.IsolatedAsyncioTestCase):
    """The turn-completion judgment pushed upstream as UserTurnCompletionFrame."""

    def _reports(self, reader: _Reader) -> list[UserTurnCompletionFrame]:
        return reader.of(UserTurnCompletionFrame)

    async def test_each_marker_is_reported_upstream(self):
        for marker, kind in [
            (USER_TURN_COMPLETE_MARKER, "complete"),
            (USER_TURN_INCOMPLETE_SHORT_MARKER, "short"),
            (USER_TURN_INCOMPLETE_LONG_MARKER, "long"),
        ]:
            with self.subTest(kind=kind):
                reader = _Reader()

                await reader.text(f"{marker} Hello")

                reports = [
                    (f, d) for f, d in reader.pushed if isinstance(f, UserTurnCompletionFrame)
                ]
                self.assertEqual(len(reports), 1)
                report, direction = reports[0]
                self.assertEqual((report.kind, report.marker), (kind, marker))
                self.assertEqual(direction, FrameDirection.UPSTREAM)

    async def test_reports_carry_the_context_frame_id(self):
        reader = _Reader()
        context_frame = LLMContextFrame(context=LLMContext())
        await reader.process(context_frame)

        await reader.text(f"{USER_TURN_COMPLETE_MARKER} Hello")

        (report,) = self._reports(reader)
        self.assertEqual(report.context_frame_id, context_frame.id)

    async def test_a_tool_call_is_reported_as_complete(self):
        reader = _Reader()
        await reader.push(FunctionCallsStartedFrame(function_calls=[]))

        (report,) = self._reports(reader)
        self.assertEqual((report.kind, report.marker), ("complete", None))
        self.assertEqual(len(reader.of(UserTurnInferenceCompletedFrame)), 2)

    async def test_a_response_without_a_marker_is_reported_with_no_kind(self):
        reader = _Reader()
        await reader.push(LLMFullResponseStartFrame())
        await reader.text("Hello there")
        await reader.push(LLMFullResponseEndFrame())

        (report,) = self._reports(reader)
        self.assertEqual((report.kind, report.marker), (None, None))

    async def test_each_inference_is_reported_once(self):
        reader = _Reader()
        await reader.process(LLMContextFrame(context=LLMContext()))
        await reader.push(LLMFullResponseStartFrame())
        await reader.text(f"{USER_TURN_COMPLETE_MARKER} Let me check")
        await reader.push(FunctionCallsStartedFrame(function_calls=[]))
        await reader.push(LLMFullResponseEndFrame())

        # The next inference is reported afresh.
        await reader.process(LLMContextFrame(context=LLMContext()))
        await reader.push(LLMFullResponseStartFrame())
        await reader.text(USER_TURN_INCOMPLETE_SHORT_MARKER)
        await reader.push(LLMFullResponseEndFrame())

        self.assertEqual([r.kind for r in self._reports(reader)], ["complete", "short"])


if __name__ == "__main__":
    unittest.main()
