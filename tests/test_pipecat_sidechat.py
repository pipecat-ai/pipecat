#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for PipecatSidechat, the processor, and how it reads who said what."""

import pytest

from pipecat.classifiers.sidechat.conversation import SidechatAgent, SidechatLine, SidechatTurn
from pipecat.classifiers.sidechat.model import (
    PipecatSidechatModel,
    SidechatAnswer,
    SidechatThresholds,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
    LLMContextUserTurnFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.sidechat import PipecatSidechat, SidechatSpeakers, SidechatVerdict
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils import models as shared_models


def _answer(respond: float, **intent: float) -> SidechatAnswer:
    shares = {"direct": 0.0, "indirect": 0.0, "open": 0.0, "about": 0.0, "other": 0.0, **intent}
    return SidechatAnswer(intent=shares, respond=respond, unclear=0.1)


RESPOND = _answer(0.95, direct=0.85, indirect=0.05, other=0.1)
HOLD = _answer(0.5, open=0.7, other=0.3)
IGNORE = _answer(0.02, about=0.1, other=0.9)


class FakeSidechatModel:
    """Answers each line by its text, and remembers the lines it was given."""

    thresholds = SidechatThresholds(respond=0.6, hold=0.4, open=0.3)
    prepares = True

    def __init__(
        self,
        answers: dict[str, SidechatAnswer] | None = None,
        *,
        error: Exception | None = None,
    ):
        self.answers = answers or {}
        self.lines: list[SidechatLine] = []
        self.prepared: list[SidechatLine] = []
        self._error = error

    def prepare(self, agent: SidechatAgent, line: SidechatLine):
        self.prepared.append(line)

    def answer(self, agent: SidechatAgent, line: SidechatLine) -> SidechatAnswer:
        self.lines.append(line)
        if self._error:
            raise self._error
        return self.answers.get(line.text, IGNORE)


@pytest.fixture(autouse=True)
def _unshared_models(monkeypatch):
    """Each test loads its own model."""
    monkeypatch.setattr(shared_models, "_models", {})


# Who said what


def test_a_tagged_turn_is_split_by_speaker():
    speakers = SidechatSpeakers()

    assert speakers.split("[Speaker 1] Shall we ask? [Speaker 2] Go on.") == [
        ("Speaker 1", "Shall we ask?"),
        ("Speaker 2", "Go on."),
    ]


def test_a_speakers_transcripts_in_a_row_are_one_line():
    speakers = SidechatSpeakers()

    assert speakers.split("[Speaker 1] Atlas, how far [Speaker 1] is Leeds? [Speaker 2] Hm.") == [
        ("Speaker 1", "Atlas, how far is Leeds?"),
        ("Speaker 2", "Hm."),
    ]


def test_words_without_a_tag_have_no_speaker():
    speakers = SidechatSpeakers()

    assert speakers.split("Who's this?") == [(None, "Who's this?")]
    assert speakers.split("Hello [Speaker 2] hi") == [(None, "Hello"), ("Speaker 2", "hi")]
    assert speakers.split("[Speaker 1]  ") == []


@pytest.mark.parametrize(
    "speaker_format, text",
    [
        ("Speaker {speaker}: {text}", "Speaker A: Shall we ask? Speaker B: Go on."),
        ("<{speaker}>{text}</{speaker}>", "<A>Shall we ask?</A> <B>Go on.</B>"),
        (
            "{speaker_number} ({speaker}) said: {text}",
            "1 (A) said: Shall we ask? 2 (B) said: Go on.",
        ),
    ],
)
def test_other_formats_are_read_by_their_own_tags(speaker_format, text):
    first, second = SidechatSpeakers(speaker_format).split(text)

    assert (first[1], second[1]) == ("Shall we ask?", "Go on.")
    assert first[0] != second[0]


@pytest.mark.parametrize("speaker_format", ["{text}", "{text} ({speaker})", "[{speaker}]"])
def test_a_format_has_to_name_the_speaker_before_the_text(speaker_format):
    with pytest.raises(ValueError):
        SidechatSpeakers(speaker_format)


# The processor


def _sidechat(monkeypatch, model: FakeSidechatModel, **kwargs) -> tuple[PipecatSidechat, list]:
    """A processor whose model downloads as ``model``, and the verdicts it reports."""
    monkeypatch.setattr(PipecatSidechatModel, "download", lambda **kwargs: model)
    sidechat = PipecatSidechat(agent_name="Atlas", **kwargs)
    verdicts: list[SidechatVerdict] = []

    @sidechat.event_handler("on_sidechat_decision")
    async def on_sidechat_decision(sidechat, verdict):
        verdicts.append(verdict)

    return sidechat, verdicts


def _turn(text: str, *, speculation: bool = False) -> list:
    """A user turn as the user aggregator ends it: the turn, then the context that answers it."""
    return [
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        LLMContextUserTurnFrame(text=text, timestamp="now", speculation=speculation),
        LLMContextFrame(context=LLMContext(), speculation=speculation),
        SleepFrame(sleep=0.05),
    ]


def _answered(frames) -> int:
    return sum(isinstance(f, LLMContextFrame) for f in frames)


@pytest.mark.asyncio
async def test_a_line_for_the_bot_is_answered(monkeypatch):
    sidechat, verdicts = _sidechat(
        monkeypatch, FakeSidechatModel({"Atlas, what time is it?": RESPOND})
    )

    down, _ = await run_test(sidechat, frames_to_send=_turn("Atlas, what time is it?"))

    assert _answered(down) == 1
    assert [(v.decision, v.reason, v.respond) for v in verdicts] == [("respond", "model", 0.95)]


@pytest.mark.asyncio
async def test_a_line_for_someone_else_isnt_answered(monkeypatch):
    sidechat, verdicts = _sidechat(monkeypatch, FakeSidechatModel())

    down, _ = await run_test(sidechat, frames_to_send=_turn("Sam, did you see the game?"))

    assert _answered(down) == 0
    assert [(v.decision, v.reason) for v in verdicts] == [("ignore", "model")]


@pytest.mark.asyncio
async def test_a_held_turn_is_answered_when_nobody_else_speaks(monkeypatch):
    sidechat, verdicts = _sidechat(
        monkeypatch, FakeSidechatModel({"Does anyone know?": HOLD}), hold_secs=0.2
    )

    down, _ = await run_test(
        sidechat, frames_to_send=[*_turn("Does anyone know?"), SleepFrame(sleep=0.4)]
    )

    assert _answered(down) == 1
    assert [(v.decision, v.reason) for v in verdicts] == [("hold", "model"), ("respond", "silence")]


@pytest.mark.asyncio
async def test_a_held_turn_is_let_go_when_someone_speaks(monkeypatch):
    sidechat, verdicts = _sidechat(
        monkeypatch, FakeSidechatModel({"Does anyone know?": HOLD}), hold_secs=0.3
    )

    down, _ = await run_test(
        sidechat,
        frames_to_send=[
            *_turn("Does anyone know?"),
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.4),
        ],
    )

    assert _answered(down) == 0
    assert [(v.decision, v.reason) for v in verdicts] == [
        ("hold", "model"),
        ("ignore", "floor_taken"),
    ]


@pytest.mark.asyncio
async def test_a_held_speculation_is_let_go(monkeypatch):
    sidechat, verdicts = _sidechat(monkeypatch, FakeSidechatModel({"Does anyone know?": HOLD}))

    down, _ = await run_test(sidechat, frames_to_send=_turn("Does anyone know?", speculation=True))

    assert _answered(down) == 0
    assert [(v.decision, v.reason) for v in verdicts] == [("ignore", "speculation")]


@pytest.mark.asyncio
async def test_a_context_that_follows_no_user_turn_passes_unjudged(monkeypatch):
    model = FakeSidechatModel()
    sidechat, _ = _sidechat(monkeypatch, model)

    down, _ = await run_test(sidechat, frames_to_send=[LLMContextFrame(context=LLMContext())])

    assert _answered(down) == 1
    assert model.lines == []


@pytest.mark.asyncio
async def test_the_model_reads_who_said_each_turn_and_what_the_bot_said(monkeypatch):
    model = FakeSidechatModel({"Atlas, how far is Leeds?": RESPOND})
    sidechat, _ = _sidechat(monkeypatch, model)

    await run_test(
        sidechat,
        frames_to_send=[
            *_turn("[Speaker 1] Atlas, how far [Speaker 1] is Leeds?"),
            LLMContextAssistantTurnFrame(text="About forty miles.", timestamp="now"),
            *_turn("[Speaker 2] And by train?"),
        ],
    )

    first, second = model.lines
    assert (first.text, first.speaker, first.history) == (
        "Atlas, how far is Leeds?",
        "Speaker 1",
        (),
    )
    assert (second.text, second.speaker) == ("And by train?", "Speaker 2")
    assert second.history == (
        SidechatTurn(text="Atlas, how far is Leeds?", speaker="Speaker 1"),
        SidechatTurn(text="About forty miles.", agent=True),
    )


@pytest.mark.asyncio
async def test_a_turn_of_several_speakers_is_answered_when_a_line_is_for_the_bot(monkeypatch):
    model = FakeSidechatModel({"Atlas, what time is it?": RESPOND})
    sidechat, verdicts = _sidechat(monkeypatch, model)

    down, _ = await run_test(
        sidechat,
        frames_to_send=_turn("[Speaker 1] Atlas, what time is it? [Speaker 2] I think it's late."),
    )

    assert _answered(down) == 1
    assert [(v.decision, v.line.text) for v in verdicts] == [("respond", "Atlas, what time is it?")]
    # The last line was read first, with the line before it as its history.
    last, first = model.lines
    assert last.history == (SidechatTurn(text="Atlas, what time is it?", speaker="Speaker 1"),)
    assert first.history == ()


@pytest.mark.asyncio
async def test_a_line_to_hold_counts_only_when_its_the_last(monkeypatch):
    model = FakeSidechatModel({"Does anyone know?": HOLD})
    sidechat, verdicts = _sidechat(monkeypatch, model, hold_secs=0.1)

    down, _ = await run_test(
        sidechat,
        frames_to_send=[
            *_turn("[Speaker 1] Does anyone know? [Speaker 2] Saturday, I think."),
            SleepFrame(sleep=0.3),
        ],
    )

    assert _answered(down) == 0
    assert [(v.decision, v.line.text) for v in verdicts] == [("ignore", "Saturday, I think.")]


@pytest.mark.asyncio
async def test_every_line_of_a_turn_joins_the_turns_the_model_reads(monkeypatch):
    model = FakeSidechatModel()
    sidechat, _ = _sidechat(monkeypatch, model)

    await run_test(
        sidechat,
        frames_to_send=[
            *_turn("[Speaker 1] Shall we ask? [Speaker 2] Go on."),
            *_turn("[Speaker 1] Atlas?"),
        ],
    )

    assert model.lines[-1].history == (
        SidechatTurn(text="Shall we ask?", speaker="Speaker 1"),
        SidechatTurn(text="Go on.", speaker="Speaker 2"),
    )


@pytest.mark.asyncio
async def test_without_tags_every_speaker_is_unknown(monkeypatch):
    model = FakeSidechatModel()
    sidechat, _ = _sidechat(monkeypatch, model, speaker_format=None)

    await run_test(sidechat, frames_to_send=_turn("[Speaker 1] Hi."))

    assert [(line.text, line.speaker) for line in model.lines] == [("[Speaker 1] Hi.", None)]


@pytest.mark.asyncio
async def test_the_turns_so_far_are_read_ahead_of_the_next_line(monkeypatch):
    model = FakeSidechatModel()
    sidechat, _ = _sidechat(monkeypatch, model)

    await run_test(
        sidechat,
        frames_to_send=[
            *_turn("[Speaker 1] Morning."),
            LLMContextAssistantTurnFrame(text="Morning!", timestamp="now"),
            SleepFrame(sleep=0.05),
        ],
    )

    assert [line.history for line in model.prepared] == [
        (SidechatTurn(text="Morning.", speaker="Speaker 1"),),
        (
            SidechatTurn(text="Morning.", speaker="Speaker 1"),
            SidechatTurn(text="Morning!", agent=True),
        ),
    ]


@pytest.mark.asyncio
async def test_a_speculation_isnt_one_of_the_turns(monkeypatch):
    model = FakeSidechatModel()
    sidechat, _ = _sidechat(monkeypatch, model)

    await run_test(
        sidechat,
        frames_to_send=[
            *_turn("[Speaker 1] Atlas, how far", speculation=True),
            UserTurnResponseCancelFrame(),
            *_turn("[Speaker 1] Atlas, how far is Leeds?"),
        ],
    )

    assert [line.history for line in model.lines] == [(), ()]


@pytest.mark.asyncio
async def test_a_turn_that_confirms_a_speculation_isnt_decided_again(monkeypatch):
    model = FakeSidechatModel({"Atlas, how far is Leeds?": RESPOND})
    sidechat, verdicts = _sidechat(monkeypatch, model)

    down, _ = await run_test(
        sidechat,
        frames_to_send=[
            *_turn("Atlas, how far is Leeds?", speculation=True),
            # The turn ends as speculated: it is written, and nothing runs the LLM again.
            LLMContextUserTurnFrame(text="Atlas, how far is Leeds?", timestamp="now"),
            SleepFrame(sleep=0.05),
            # A context that follows no user turn.
            LLMContextFrame(context=LLMContext()),
        ],
    )

    assert _answered(down) == 2
    assert len(verdicts) == 1
    assert len(model.lines) == 1


@pytest.mark.asyncio
async def test_a_turn_after_a_withdrawn_speculation_is_decided(monkeypatch):
    model = FakeSidechatModel({"Atlas, how far is Leeds?": RESPOND})
    sidechat, verdicts = _sidechat(monkeypatch, model)

    down, _ = await run_test(
        sidechat,
        frames_to_send=[
            *_turn("Atlas, how far", speculation=True),
            UserTurnResponseCancelFrame(),
            *_turn("Atlas, how far is Leeds?"),
        ],
    )

    assert [(v.decision, v.line.text) for v in verdicts] == [
        ("ignore", "Atlas, how far"),
        ("respond", "Atlas, how far is Leeds?"),
    ]
    assert _answered(down) == 1


@pytest.mark.asyncio
async def test_a_turn_begun_over_the_bot_is_marked(monkeypatch):
    model = FakeSidechatModel()
    sidechat, _ = _sidechat(monkeypatch, model)

    await run_test(
        sidechat,
        frames_to_send=[BotStartedSpeakingFrame(), *_turn("Stop."), *_turn("Sorry, go on.")],
    )

    assert [line.over_agent for line in model.lines] == [True, True]


@pytest.mark.asyncio
async def test_without_the_model_only_a_turn_naming_the_bot_is_answered(monkeypatch):
    sidechat, verdicts = _sidechat(monkeypatch, FakeSidechatModel(error=RuntimeError("boom")))

    down, _ = await run_test(
        sidechat,
        frames_to_send=[
            *_turn("[Speaker 1] Atlas, what time is it? [Speaker 2] Is it late?"),
            *_turn("Is it late?"),
        ],
    )

    assert _answered(down) == 1
    assert [(v.decision, v.reason, v.line.text) for v in verdicts] == [
        ("respond", "error", "Atlas, what time is it?"),
        ("ignore", "error", "Is it late?"),
    ]


@pytest.mark.asyncio
async def test_a_line_with_one_of_the_bots_commands_is_answered(monkeypatch):
    model = FakeSidechatModel({"Atlas, what time is it?": RESPOND, "Shall we eat first?": HOLD})
    sidechat, verdicts = _sidechat(monkeypatch, model, commands=["play chess", "go to sleep"])

    down, _ = await run_test(
        sidechat,
        frames_to_send=[
            *_turn("Let's PLAY CHESS."),  # the model would ignore it
            *_turn("Did you see the chess players in the park?"),  # the words, not the phrase
            *_turn("Atlas, what time is it?"),
            *_turn("Shall we eat first? Then go to sleep."),  # it holds this text otherwise
        ],
    )

    assert _answered(down) == 3
    assert [(v.decision, v.reason) for v in verdicts] == [
        ("respond", "command"),
        ("ignore", "model"),
        ("respond", "model"),
        ("respond", "command"),
    ]
    # The model still read the line: its numbers come with the verdict.
    assert verdicts[0].respond == IGNORE.respond


@pytest.mark.asyncio
async def test_a_command_from_any_speaker_of_a_turn_answers_it(monkeypatch):
    sidechat, verdicts = _sidechat(monkeypatch, FakeSidechatModel(), commands=["play chess"])

    down, _ = await run_test(
        sidechat,
        frames_to_send=[*_turn("[Speaker 1] Can we play chess? [Speaker 2] After lunch, maybe.")],
    )

    assert _answered(down) == 1
    assert [(v.decision, v.reason, v.line.text) for v in verdicts] == [
        ("respond", "command", "Can we play chess?")
    ]


@pytest.mark.asyncio
async def test_without_the_model_a_command_is_answered_as_the_bots_name_is(monkeypatch):
    model = FakeSidechatModel(error=RuntimeError("boom"))
    sidechat, verdicts = _sidechat(monkeypatch, model, commands=["go to sleep"])

    down, _ = await run_test(
        sidechat, frames_to_send=[*_turn("Okay, go to sleep."), *_turn("Is it late?")]
    )

    assert _answered(down) == 1
    assert [(v.decision, v.reason) for v in verdicts] == [("respond", "error"), ("ignore", "error")]


def test_a_command_matches_whole_and_without_case():
    sidechat = PipecatSidechat(agent_name="Atlas", commands=["play chess", " ", "Chess board"])

    assert sidechat._commanded("play chess")
    assert sidechat._commanded("Okay, PLAY CHESS!")
    assert sidechat._commanded("Put the chess board up.")
    assert not sidechat._commanded("He was displaying chessboards.")
    assert not sidechat._commanded("Play some chess?")
    assert not PipecatSidechat(agent_name="Atlas")._commanded("play chess")


def test_a_command_is_read_as_speech_recognition_writes_it():
    sidechat = PipecatSidechat(agent_name="Atlas", commands=["what's on the table"])

    assert sidechat._commanded("So what’s on  the\ntable right now?")
    assert sidechat._commanded("WHAT'S ON THE TABLE")
    assert not sidechat._commanded("What is on the table?")


def test_one_command_given_as_a_string_is_one_command():
    sidechat = PipecatSidechat(agent_name="Atlas", commands="play chess")

    assert sidechat._commanded("Let's play chess.")
    assert not sidechat._commanded("A pleasant day.")  # every letter of the phrase is in here


def test_the_bots_role_reaches_the_model():
    role = "voice assistant in the meeting room's screen; it shows the agenda"
    assert PipecatSidechat(agent_name="Atlas", role=role)._classifier._agent.role == role
    assert PipecatSidechat(agent_name="Atlas")._classifier._agent.role == "AI assistant on the call"


@pytest.mark.asyncio
async def test_a_model_that_cant_load_leaves_only_the_bots_name(monkeypatch):
    def download(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(PipecatSidechatModel, "download", download)
    sidechat = PipecatSidechat(agent_name="Atlas")

    down, _ = await run_test(
        sidechat,
        frames_to_send=[*_turn("Atlas, what time is it?"), *_turn("Is it late?")],
        ignore_start=True,
    )

    assert _answered(down) == 1


# After the user aggregator


def _said(speaker: int, text: str) -> TranscriptionFrame:
    """A transcript as a diarizing STT service tags it with the default format."""
    return TranscriptionFrame(
        text=f"[Speaker {speaker}] {text}",
        user_id="",
        timestamp="now",
        speaker=TranscriptionSpeaker(label=str(speaker - 1), number=speaker),
    )


def _spoken(*transcripts: TranscriptionFrame) -> list:
    """A user turn as it reaches the user aggregator."""
    return [
        ProposedUserStartedSpeakingFrame(),
        *transcripts,
        ProposedUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.3),
    ]


def _after_the_user_aggregator(sidechat: PipecatSidechat) -> tuple[Pipeline, LLMContext]:
    context = LLMContext()
    user_aggregator = LLMUserAggregator(
        context, params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies())
    )
    return Pipeline([user_aggregator, sidechat]), context


@pytest.mark.asyncio
async def test_the_user_aggregators_turns_are_read_and_stay_in_the_context(monkeypatch):
    model = FakeSidechatModel({"Atlas, what time is it?": RESPOND})
    sidechat, verdicts = _sidechat(monkeypatch, model)
    pipeline, context = _after_the_user_aggregator(sidechat)

    down, _ = await run_test(
        pipeline,
        frames_to_send=[
            *_spoken(_said(1, "Did you see the game?"), _said(2, "What a finish.")),
            *_spoken(_said(2, "Atlas, what time"), _said(2, "is it?")),
        ],
    )

    assert _answered(down) == 1
    assert [(v.decision, v.line.speaker, v.line.text) for v in verdicts] == [
        ("ignore", "Speaker 2", "What a finish."),
        ("respond", "Speaker 2", "Atlas, what time is it?"),
    ]
    assert verdicts[-1].line.history == (
        SidechatTurn(text="Did you see the game?", speaker="Speaker 1"),
        SidechatTurn(text="What a finish.", speaker="Speaker 2"),
    )
    # The bot heard both turns, tags and all.
    assert [message["content"] for message in context.messages] == [
        "[Speaker 1] Did you see the game? [Speaker 2] What a finish.",
        "[Speaker 2] Atlas, what time [Speaker 2] is it?",
    ]


@pytest.mark.asyncio
async def test_a_turn_held_after_the_user_aggregator_is_answered_in_the_quiet(monkeypatch):
    model = FakeSidechatModel({"Does anyone know when the next match is?": HOLD})
    sidechat, verdicts = _sidechat(monkeypatch, model, hold_secs=0.2)
    pipeline, _ = _after_the_user_aggregator(sidechat)

    down, _ = await run_test(
        pipeline,
        frames_to_send=[
            *_spoken(_said(1, "Does anyone know when the next match is?")),
            SleepFrame(sleep=0.4),
        ],
    )

    assert _answered(down) == 1
    assert [(v.decision, v.reason) for v in verdicts] == [("hold", "model"), ("respond", "silence")]
