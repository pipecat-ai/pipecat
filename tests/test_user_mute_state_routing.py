#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn state and input filtering across mute boundaries."""

from dataclasses import dataclass

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMRunFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SystemFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnCompletionFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_mute import BaseUserMuteStrategy, FirstSpeechUserMuteStrategy
from pipecat.turns.user_start import (
    ExternalUserTurnStartStrategy,
    VADUserTurnStartStrategy,
    WakePhraseUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    LLMTurnCompletionUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

SPEECH_SIGNALS = [
    pytest.param(VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame, id="vad"),
    pytest.param(UserStartedSpeakingFrame, UserStoppedSpeakingFrame, id="external"),
    pytest.param(ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame, id="proposed"),
]


@dataclass
class CheckpointFrame(SystemFrame):
    """Capture the user and assistant state at a named checkpoint.

    Parameters:
        label: Name of the checkpoint.
    """

    label: str


@dataclass
class MuteControlFrame(SystemFrame):
    """Set a custom mute policy independently of bot playback.

    Parameters:
        mute: Whether the custom policy should mute input.
    """

    mute: bool


class TranscriptMuteStrategy(BaseUserMuteStrategy):
    """Observe suppressed transcripts and unmute on a spoken command."""

    def __init__(self):
        """Initialize the policy and the observed transcript list."""
        super().__init__()
        self.muted = False
        self.transcripts = []

    async def process_frame(self, frame: Frame) -> bool:
        """Apply control frames and observe the resume command."""
        if isinstance(frame, MuteControlFrame):
            self.muted = frame.mute
        elif isinstance(frame, TranscriptionFrame):
            self.transcripts.append(frame.text)
            if frame.text == "resume":
                self.muted = False
        return self.muted


class StateProbe(FrameProcessor):
    """Observe lifecycle state and inference without an external model."""

    def __init__(
        self, pair: LLMContextAggregatorPair, completion: UserTurnCompletionFrame | None = None
    ):
        """Initialize the probe for a context aggregator pair.

        Args:
            pair: User and assistant aggregators being observed.
            completion: Optional response to bind to the observed inference.
        """
        super().__init__(enable_direct_mode=True)
        self.pair = pair
        self.snapshots = {}
        self.requests = []
        self.frames = []
        self.completion = completion

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Record checkpoints and consume inference requests."""
        await super().process_frame(frame, direction)
        self.frames.append(type(frame))
        if isinstance(frame, LLMContextFrame):
            if self.completion is not None:
                self.completion.context_frame_id = frame.id
            self.requests.append(
                (self.pair.user()._user_is_muted, [dict(m) for m in frame.context.messages])
            )
        elif isinstance(frame, CheckpointFrame):
            controller = self.pair.user()._user_turn_controller
            self.snapshots[frame.label] = (
                controller._user_speaking,
                controller._user_turn,
                self.pair.assistant()._user_speaking,
                len(self.requests),
            )
        else:
            await self.push_frame(frame, direction)


def make_pair(
    start_frame: type[Frame] = VADUserStartedSpeakingFrame,
    mute_strategies: list[BaseUserMuteStrategy] | None = None,
) -> LLMContextAggregatorPair:
    """Build a pair with short deterministic turn timers."""
    if start_frame is VADUserStartedSpeakingFrame:
        strategies = UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.03)]
        )
    else:
        strategies = UserTurnStrategies(
            start=[ExternalUserTurnStartStrategy()],
            stop=[ExternalUserTurnStopStrategy(timeout=0.03)],
        )
    return LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=strategies,
            user_mute_strategies=(
                mute_strategies if mute_strategies is not None else [FirstSpeechUserMuteStrategy()]
            ),
            user_turn_stop_timeout=0.12,
            empty_user_turn=None,
        ),
    )


def transcript(text: str) -> TranscriptionFrame:
    """Create a finalized transcript."""
    return TranscriptionFrame(text=text, user_id="", timestamp="now", finalized=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(("start_frame", "stop_frame"), SPEECH_SIGNALS)
@pytest.mark.parametrize("has_text", [False, True], ids=["watchdog", "transcript"])
async def test_muted_stop_closes_existing_turn(
    start_frame: type[Frame], stop_frame: type[Frame], has_text: bool
):
    """A matching stop closes accepted speech without accepting muted text."""
    pair = make_pair(start_frame)
    probe = StateProbe(pair)
    turns = []
    stopped_strategies = []

    @pair.user().event_handler("on_user_turn_stopped")
    async def stopped(aggregator, strategy, message):
        turns.append(message.content)
        stopped_strategies.append(strategy)

    frames = [start_frame(), SleepFrame(sleep=0.02)]
    if has_text:
        frames += [transcript("Accepted"), SleepFrame(sleep=0.02)]
    frames += [
        BotStartedSpeakingFrame(),
        stop_frame(),
        transcript("Suppressed"),
        SleepFrame(sleep=0.25),
        CheckpointFrame("muted"),
        start_frame(),
        stop_frame(),
        SleepFrame(sleep=0.05),
        BotStoppedSpeakingFrame(),
        start_frame(),
        SleepFrame(sleep=0.02),
        transcript("Next turn"),
        stop_frame(),
        SleepFrame(sleep=0.2),
    ]
    await run_test(Pipeline([pair.user(), probe, pair.assistant()]), frames_to_send=frames)

    assert probe.snapshots["muted"] == (False, False, False, int(has_text))
    assert turns == ["Accepted" if has_text else "", "Next turn"]
    stop_strategy = pair.user()._user_turn_controller.user_turn_strategies.stop[0]
    assert stopped_strategies == [stop_strategy if has_text else None, stop_strategy]
    expected = ([{"role": "user", "content": "Accepted"}] if has_text else []) + [
        {"role": "user", "content": "Next turn"}
    ]
    assert pair.user().context.messages == expected
    assert len(probe.requests) == len(expected)
    if has_text:
        assert probe.requests[0] == (True, [{"role": "user", "content": "Accepted"}])
    assert probe.frames.count(UserStartedSpeakingFrame) == 2
    assert probe.frames.count(UserStoppedSpeakingFrame) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("start_frame", "stop_frame"), SPEECH_SIGNALS)
async def test_muted_speech_tracks_activity_without_opening_turn(
    start_frame: type[Frame], stop_frame: type[Frame]
):
    """Muted starts update speech state without opening or announcing a turn."""
    pair = make_pair(start_frame)
    probe = StateProbe(pair)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[
            BotStartedSpeakingFrame(),
            start_frame(),
            transcript("Suppressed"),
            CheckpointFrame("speaking"),
            stop_frame(),
            SleepFrame(sleep=0.2),
            CheckpointFrame("stopped"),
            BotStoppedSpeakingFrame(),
        ],
    )
    assert probe.snapshots["speaking"] == (True, False, False, 0)
    assert probe.snapshots["stopped"] == (False, False, False, 0)
    assert not pair.user().context.messages
    assert UserStartedSpeakingFrame not in probe.frames
    assert UserStoppedSpeakingFrame not in probe.frames


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(FrameDirection))
async def test_muted_stop_pairs_with_external_start_blocked_by_gate(direction: FrameDirection):
    """An announced speaking segment can end even when a gate blocks its turn."""
    pair = make_pair(UserStartedSpeakingFrame, mute_strategies=[TranscriptMuteStrategy()])
    pair.user()._user_turn_controller.user_turn_strategies.start.insert(
        0, WakePhraseUserTurnStartStrategy(phrases=["hey bot"])
    )
    probe = StateProbe(pair)
    processors = [pair.user(), probe, pair.assistant()]
    if direction == FrameDirection.UPSTREAM:
        processors.reverse()
    await run_test(
        Pipeline(processors),
        frames_to_send_direction=direction,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            CheckpointFrame("gated"),
            MuteControlFrame(True),
            UserStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            CheckpointFrame("stopped"),
            MuteControlFrame(False),
            UserStartedSpeakingFrame(),
            MuteControlFrame(True),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            CheckpointFrame("next_segment"),
            MuteControlFrame(False),
        ],
    )
    assert probe.snapshots["gated"] == (True, False, True, 0)
    assert probe.snapshots["stopped"] == (False, False, False, 0)
    assert probe.snapshots["next_segment"] == (False, False, False, 0)
    assert probe.frames.count(UserStartedSpeakingFrame) == 2
    assert probe.frames.count(UserStoppedSpeakingFrame) == 2
    assert not pair.user().context.messages
    assert not pair.user()._user_idle_controller._user_turn_in_progress


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(FrameDirection))
async def test_muted_external_stop_deduplicates_each_direction(direction: FrameDirection):
    """A forwarded stop closes its direction without hiding the other stop."""
    pair = make_pair()
    down, up = await run_test(
        pair.user(),
        frames_to_send_direction=direction,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.2),
            BotStoppedSpeakingFrame(),
        ],
    )
    for frames in (down, up):
        assert sum(isinstance(f, UserStartedSpeakingFrame) for f in frames) == 1
        assert sum(isinstance(f, UserStoppedSpeakingFrame) for f in frames) == 1
        assert not any(isinstance(f, LLMContextFrame) for f in frames)


@pytest.mark.asyncio
async def test_muted_broadcast_stop_does_not_generate_another_sibling():
    """An external broadcast already closes speaking state in both directions."""
    turn_processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            stop=[
                SpeechTimeoutUserTurnStopStrategy(
                    user_speech_timeout=0.03, wait_for_transcript=False
                )
            ]
        )
    )
    pair = make_pair(UserStartedSpeakingFrame)
    down, up = await run_test(
        Pipeline([turn_processor, pair.user(), pair.assistant()]),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            BotStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.25),
            BotStoppedSpeakingFrame(),
        ],
    )
    for frames in (down, up):
        stops = [f for f in frames if isinstance(f, UserStoppedSpeakingFrame)]
        assert len(stops) == 1
        assert stops[0].broadcast_sibling_id is not None
    assert not pair.user()._user_turn_controller._user_turn
    assert not pair.assistant()._user_speaking


@pytest.mark.asyncio
@pytest.mark.parametrize(("start_frame", "stop_frame"), SPEECH_SIGNALS)
async def test_unmute_does_not_force_an_active_speaker_to_stop(
    start_frame: type[Frame], stop_frame: type[Frame]
):
    """Mute transitions preserve ongoing speech until its matching stop."""
    pair = make_pair(start_frame)
    probe = StateProbe(pair)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[
            start_frame(),
            SleepFrame(sleep=0.02),
            transcript("Accepted"),
            SleepFrame(sleep=0.02),
            BotStartedSpeakingFrame(),
            SleepFrame(sleep=0.25),
            CheckpointFrame("muted"),
            BotStoppedSpeakingFrame(),
            CheckpointFrame("unmuted"),
            stop_frame(),
            SleepFrame(sleep=0.2),
        ],
    )
    assert probe.snapshots["muted"] == (True, True, True, 0)
    assert probe.snapshots["unmuted"] == (True, True, True, 0)
    assert len(probe.requests) == 1
    assert probe.requests[0][0] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("start_frame", "stop_frame"), SPEECH_SIGNALS)
@pytest.mark.parametrize("enable_interruptions", [False, True])
async def test_muted_speech_preserves_pending_turn_inference(
    start_frame: type[Frame], stop_frame: type[Frame], enable_interruptions: bool
):
    """Muted speech leaves an accepted turn's response and completion intact."""
    pair = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=enable_interruptions)],
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.03),
                    LLMTurnCompletionUserTurnStopStrategy(),
                ],
            ),
            user_mute_strategies=[TranscriptMuteStrategy()],
            user_turn_stop_timeout=1.0,
            empty_user_turn=None,
        ),
    )
    completion = UserTurnCompletionFrame(kind="complete")
    probe = StateProbe(pair, completion=completion)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            transcript("Accepted"),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.08),
            CheckpointFrame("pending"),
            MuteControlFrame(True),
            start_frame(),
            SleepFrame(sleep=0.02),
            completion,
            stop_frame(),
            SleepFrame(sleep=0.08),
            CheckpointFrame("closed"),
            MuteControlFrame(False),
        ],
    )
    assert probe.snapshots["pending"] == (False, True, True, 1)
    assert probe.frames.count(InterruptionFrame) == int(enable_interruptions)
    assert UserTurnResponseCancelFrame not in probe.frames
    assert probe.snapshots["closed"] == (False, False, False, 1)
    assert len(probe.requests) == 1
    assert pair.user().context.messages == [{"role": "user", "content": "Accepted"}]


@pytest.mark.asyncio
async def test_custom_strategy_can_unmute_from_suppressed_transcript():
    """A resume command reaches the mute policy without entering the context."""
    strategy = TranscriptMuteStrategy()
    pair = make_pair(mute_strategies=[strategy])
    probe = StateProbe(pair)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[
            MuteControlFrame(True),
            VADUserStartedSpeakingFrame(),
            transcript("Suppressed"),
            SleepFrame(sleep=0.02),
            transcript("resume"),
            SleepFrame(sleep=0.02),
            CheckpointFrame("resumed"),
            transcript("Accepted"),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.2),
        ],
    )
    assert strategy.transcripts == ["Suppressed", "resume", "Accepted"]
    assert probe.snapshots["resumed"] == (True, False, False, 0)
    assert pair.user().context.messages == [{"role": "user", "content": "Accepted"}]
    assert len(probe.requests) == 1


@pytest.mark.asyncio
async def test_mute_policies_remain_combined_until_all_release():
    """Releasing one policy does not override another policy's mute."""
    strategy = TranscriptMuteStrategy()
    pair = make_pair(mute_strategies=[strategy, FirstSpeechUserMuteStrategy()])
    probe = StateProbe(pair)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[
            MuteControlFrame(True),
            BotStartedSpeakingFrame(),
            MuteControlFrame(False),
            transcript("Suppressed"),
            SleepFrame(sleep=0.02),
            BotStoppedSpeakingFrame(),
            transcript("Accepted"),
            SleepFrame(sleep=0.2),
        ],
    )
    assert strategy.transcripts == ["Suppressed", "Accepted"]
    assert pair.user().context.messages == [{"role": "user", "content": "Accepted"}]
    assert len(probe.requests) == 1


@pytest.mark.asyncio
async def test_explicit_inference_request_remains_available_while_muted():
    """Muting user input does not disable application-requested inference."""
    strategy = TranscriptMuteStrategy()
    pair = make_pair(mute_strategies=[strategy])
    pair.user().context.add_message({"role": "developer", "content": "Start the greeting"})
    probe = StateProbe(pair)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[MuteControlFrame(True), LLMRunFrame(), SleepFrame(sleep=0.05)],
    )
    assert probe.requests == [(True, [{"role": "developer", "content": "Start the greeting"}])]


@pytest.mark.asyncio
async def test_muted_audio_and_interruption_do_not_reach_downstream():
    """Input filtering remains active while speech state is maintained."""
    pair = make_pair()
    probe = StateProbe(pair)
    await run_test(
        Pipeline([pair.user(), probe, pair.assistant()]),
        frames_to_send=[
            BotStartedSpeakingFrame(),
            InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1),
            InterruptionFrame(),
            transcript("Suppressed"),
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            BotStoppedSpeakingFrame(),
        ],
    )
    assert InputAudioRawFrame not in probe.frames
    assert InterruptionFrame not in probe.frames
    assert TranscriptionFrame not in probe.frames
    assert not probe.requests
