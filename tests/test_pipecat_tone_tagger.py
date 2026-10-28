#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that tags an LLM's replies with Pipecat Tone."""

import time

import pytest

from pipecat.adapters.tts.services.cartesia_adapter import CartesiaExpressionAdapter
from pipecat.adapters.tts.services.elevenlabs_adapter import ElevenLabsAudioTagExpressionAdapter
from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    ExpressionMetadataFrame,
    Frame,
    LLMContextAssistantTurnFrame,
    LLMContextUserTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSMetadataFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame
from pipecat.processors.tone import _models
from pipecat.processors.tone.tagger import PipecatToneTagger
from pipecat.processors.tone.tagger._input import Layout, Message, TurnContext, encode
from pipecat.processors.tone.tagger._labels import LabelSpace, SentenceLabels, Span
from pipecat.processors.tone.tagger.model import PipecatToneTaggerModel
from pipecat.tests.utils import SleepFrame, run_test

EXPRESSIVE = PipelineParams(enable_expressive_conversation=True)

CARTESIA = TTSMetadataFrame(
    service_name="tts", expression=CartesiaExpressionAdapter().capabilities()
)
ELEVENLABS = TTSMetadataFrame(
    service_name="tts", expression=ElevenLabsAudioTagExpressionAdapter().capabilities()
)


class FakeTaggerModel:
    """Labels each sentence from a script, keyed by the sentence."""

    def __init__(
        self,
        labels: dict[str, SentenceLabels] | None = None,
        breaths_per_reply: int | None = None,
        error: Exception | None = None,
    ):
        self.labels = labels or {}
        self.breaths_per_reply = breaths_per_reply
        self.calls: list[tuple[TurnContext, list[str], str]] = []
        self._error = error

    def tag(self, turn: TurnContext, earlier: list[str], sentence: str) -> SentenceLabels:
        self.calls.append((turn, earlier, sentence))
        if self._error:
            raise self._error
        return self.labels.get(sentence, SentenceLabels())


@pytest.fixture(autouse=True)
def _unshared_models(monkeypatch):
    """Each test loads its own model."""
    monkeypatch.setattr(_models, "_models", {})


def _tagger(monkeypatch, model: FakeTaggerModel) -> PipecatToneTagger:
    """A tagger whose model downloads as ``model``."""
    monkeypatch.setattr(PipecatToneTaggerModel, "download", lambda **kwargs: model)
    return PipecatToneTagger()


def _reply(*chunks: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(chunk) for chunk in chunks),
        LLMFullResponseEndFrame(),
    ]


def _expressive(policy: ExpressionPolicy) -> PipelineParams:
    return PipelineParams(enable_expressive_conversation=True, expression_policy=policy)


async def _tagged(
    monkeypatch, model: FakeTaggerModel, *frames: Frame, params=EXPRESSIVE
) -> list[str]:
    down, _ = await run_test(
        _tagger(monkeypatch, model), frames_to_send=[*frames], pipeline_params=params
    )
    return [f.text for f in down if isinstance(f, AggregatedTextFrame)]


@pytest.mark.asyncio
async def test_the_tagger_announces_itself_both_ways_at_start(monkeypatch):
    down, up = await run_test(
        _tagger(monkeypatch, FakeTaggerModel()),
        frames_to_send=[],
        pipeline_params=EXPRESSIVE,
    )

    assert [f for f in down if isinstance(f, ExpressionMetadataFrame)]
    assert [f for f in up if isinstance(f, ExpressionMetadataFrame)]


@pytest.mark.asyncio
async def test_labels_are_written_in_the_names_the_tts_performs(monkeypatch):
    model = FakeTaggerModel(
        {
            "Congratulations!": SentenceLabels(delivery="warm", sound_after="laugh"),
            "Your code is A B 7.": SentenceLabels(spans=[Span(13, 18, "slow")]),
        }
    )

    tagged = await _tagged(
        monkeypatch, model, CARTESIA, *_reply("Congratulations! Your code ", "is A B 7.")
    )

    assert tagged == [
        "<<content>> Congratulations! [[laughter]]",
        "<<neutral>> Your code is <<slow: A B 7>>.",
    ]


@pytest.mark.asyncio
async def test_a_delivery_is_written_only_where_it_changes(monkeypatch):
    calm = SentenceLabels(delivery="calm")
    model = FakeTaggerModel({"I'm sorry.": calm, "I'll fix it.": calm})

    tagged = await _tagged(monkeypatch, model, ELEVENLABS, *_reply("I'm sorry. I'll fix it."))

    assert tagged == ["<<calm>> I'm sorry.", "I'll fix it."]


@pytest.mark.asyncio
async def test_each_sentence_is_tagged_after_the_ones_before_it(monkeypatch):
    model = FakeTaggerModel()

    await _tagged(monkeypatch, model, ELEVENLABS, *_reply("One. Two. Three."))

    assert [(earlier, sentence) for _, earlier, sentence in model.calls] == [
        ([], "One."),
        (["One."], "Two."),
        (["One.", "Two."], "Three."),
    ]


@pytest.mark.asyncio
async def test_the_policy_narrows_the_names_written(monkeypatch):
    model = FakeTaggerModel({"Great!": SentenceLabels(delivery="warm", sound_before="laugh")})
    policy = ExpressionPolicy(delivery=["calm"], sounds=[])

    tagged = await _tagged(
        monkeypatch, model, CARTESIA, *_reply("Great!"), params=_expressive(policy)
    )

    assert tagged == ["Great!"]


@pytest.mark.asyncio
async def test_a_policy_that_leaves_out_a_name_gets_the_next_best(monkeypatch):
    model = FakeTaggerModel({"Don't worry.": SentenceLabels(delivery="calm")})
    policy = ExpressionPolicy(delivery=["reassuring", "warmly"])

    tagged = await _tagged(
        monkeypatch, model, ELEVENLABS, *_reply("Don't worry."), params=_expressive(policy)
    )

    assert tagged == ["<<reassuring>> Don't worry."]


@pytest.mark.asyncio
async def test_a_sentence_that_cant_be_tagged_is_spoken_untagged_with_an_error(monkeypatch):
    model = FakeTaggerModel(error=RuntimeError("boom"))

    down, up = await run_test(
        _tagger(monkeypatch, model),
        frames_to_send=[ELEVENLABS, *_reply("Hello there.")],
        pipeline_params=EXPRESSIVE,
    )

    assert [f.text for f in down if isinstance(f, AggregatedTextFrame)] == ["Hello there."]
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert errors and not errors[0].fatal


@pytest.mark.asyncio
async def test_a_breath_is_offered_only_with_the_breathe_preset(monkeypatch):
    model = FakeTaggerModel()

    await _tagged(monkeypatch, model, ELEVENLABS, *_reply("Hi."))
    await _tagged(
        monkeypatch,
        model,
        ELEVENLABS,
        *_reply("Hi."),
        params=_expressive(ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE])),
    )

    without, with_breathe = (turn.space.sounds for turn, _, _ in model.calls)
    assert "breath" not in without
    assert with_breathe["breath"] == "short breath"


@pytest.mark.asyncio
async def test_the_call_is_kept_from_the_turn_frames(monkeypatch):
    model = FakeTaggerModel()
    angry = [PerceptionSignal("emotion", labels=[("angry", 0.62), ("disgusted", 0.2)])]

    await _tagged(
        monkeypatch,
        model,
        ELEVENLABS,
        LLMContextAssistantTurnFrame(text="How can I help?", timestamp="t0"),
        LLMContextUserTurnFrame(text="My order is late.", timestamp="t1", perception=angry),
        *_reply("Sorry."),
    )

    turn = model.calls[0][0]
    assert [(m.role, m.text) for m in turn.messages] == [
        ("assistant", "How can I help?"),
        ("user", "My order is late."),
    ]
    # The reader's emotions, folded into the tones the model reads.
    assert turn.heard == [PerceptionSignal("tone", labels=[("frustrated", 0.82)])]


@pytest.mark.asyncio
async def test_each_reply_is_tagged_knowing_the_labels_of_the_one_before(monkeypatch):
    calm = SentenceLabels(delivery="calm")
    model = FakeTaggerModel({"I'm sorry.": calm})

    await _tagged(
        monkeypatch, model, ELEVENLABS, *_reply("I'm sorry. Let me look."), *_reply("Done.")
    )

    first, _, second = (turn for turn, _, _ in model.calls)
    assert first.previous is None
    assert second.previous == [calm, SentenceLabels()]


@pytest.mark.asyncio
async def test_breaths_past_the_models_limit_for_a_reply_are_dropped(monkeypatch):
    breath = SentenceLabels(sound_before="breath")
    model = FakeTaggerModel({"One.": breath, "Two.": breath}, breaths_per_reply=1)
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE])

    tagged = await _tagged(
        monkeypatch, model, ELEVENLABS, *_reply("One. Two."), params=_expressive(policy)
    )

    assert tagged == ["[[short breath]] One.", "Two."]


@pytest.mark.asyncio
async def test_a_breath_after_a_sentence_counts_toward_the_limit(monkeypatch):
    model = FakeTaggerModel(
        {
            "One.": SentenceLabels(sound_after="breath"),
            "Two.": SentenceLabels(sound_before="breath"),
        },
        breaths_per_reply=1,
    )
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE])

    tagged = await _tagged(
        monkeypatch, model, ELEVENLABS, *_reply("One. Two."), params=_expressive(policy)
    )

    assert tagged == ["One. [[short breath]]", "Two."]


def test_a_layout_that_reads_no_turns_reads_none_of_the_call():
    turn = TurnContext(
        [Message("user", "My order is late.")], LabelSpace({}, {}, False), ExpressionPolicy(), None
    )

    call, _, _ = encode(turn, [], "Sorry.", Layout(context_turns=0))

    assert call == ""


@pytest.mark.asyncio
async def test_text_the_user_typed_is_part_of_the_call(monkeypatch):
    model = FakeTaggerModel()

    await _tagged(
        monkeypatch,
        model,
        ELEVENLABS,
        LLMContextAssistantTurnFrame(text="How can I help?", timestamp="t0"),
        # RTVI drains the pipeline before it appends text that runs at once.
        SleepFrame(sleep=0.05),
        RTVISendTextFrame(msg_id="1", text="The spots have resolved!"),
        *_reply("That's encouraging news."),
    )

    turn = model.calls[0][0]
    assert [(m.role, m.text) for m in turn.messages] == [
        ("assistant", "How can I help?"),
        ("user", "The spots have resolved!"),
    ]
    assert turn.heard is None


@pytest.mark.asyncio
async def test_a_confirmed_turn_replaces_its_speculation_where_it_stands(monkeypatch):
    model = FakeTaggerModel()

    await _tagged(
        monkeypatch,
        model,
        ELEVENLABS,
        LLMContextUserTurnFrame(text="My order", timestamp="t1", speculation=True),
        *_reply("Sorry."),
        LLMContextAssistantTurnFrame(text="Sorry.", timestamp="t2"),
        LLMContextUserTurnFrame(text="My order is late.", timestamp="t1"),
        *_reply("Next."),
    )

    early, later = (turn for turn, _, _ in model.calls)
    assert [(m.role, m.text) for m in early.messages] == [("user", "My order")]
    assert [(m.role, m.text) for m in later.messages] == [
        ("user", "My order is late."),
        ("assistant", "Sorry."),
    ]


@pytest.mark.asyncio
async def test_a_withdrawn_speculation_leaves_the_call(monkeypatch):
    model = FakeTaggerModel()

    await _tagged(
        monkeypatch,
        model,
        ELEVENLABS,
        LLMContextAssistantTurnFrame(text="How can I help?", timestamp="t0"),
        LLMContextUserTurnFrame(text="My order", timestamp="t1", speculation=True),
        SleepFrame(sleep=0.05),
        UserTurnResponseCancelFrame(),
        LLMContextAssistantTurnFrame(text="Take your time.", timestamp="t2"),
        LLMContextUserTurnFrame(text="My order is late.", timestamp="t3"),
        *_reply("Sorry."),
    )

    turn = model.calls[0][0]
    assert [(m.role, m.text) for m in turn.messages] == [
        ("assistant", "How can I help?"),
        ("assistant", "Take your time."),
        ("user", "My order is late."),
    ]


@pytest.mark.asyncio
async def test_replies_pass_through_when_the_tts_performs_nothing(monkeypatch):
    model = FakeTaggerModel()

    tagged = await _tagged(
        monkeypatch,
        model,
        TTSMetadataFrame(service_name="tts", expression=None),
        *_reply("Hello there."),
    )

    assert tagged == ["Hello there."]
    assert not model.calls


@pytest.mark.asyncio
async def test_without_expressive_conversation_nothing_is_tagged_or_announced(monkeypatch):
    model = FakeTaggerModel()

    down, up = await run_test(
        _tagger(monkeypatch, model), frames_to_send=[ELEVENLABS, *_reply("Hello there.")]
    )

    assert [f.text for f in down if isinstance(f, AggregatedTextFrame)] == ["Hello there."]
    assert not [f for f in down + up if isinstance(f, ExpressionMetadataFrame)]
    assert not model.calls


@pytest.mark.asyncio
async def test_the_tts_metadata_goes_on_unchanged(monkeypatch):
    _, up = await run_test(
        _tagger(monkeypatch, FakeTaggerModel()),
        frames_to_send=[ELEVENLABS],
        frames_to_send_direction=FrameDirection.UPSTREAM,
        pipeline_params=EXPRESSIVE,
    )

    assert [f.expression for f in up if isinstance(f, TTSMetadataFrame)] == [ELEVENLABS.expression]


@pytest.mark.asyncio
async def test_a_model_that_cant_load_leaves_the_tagger_unusable(monkeypatch):
    def fail(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(PipecatToneTaggerModel, "download", fail)
    tagger = PipecatToneTagger()

    down, up = await run_test(
        tagger,
        frames_to_send=[ELEVENLABS, *_reply("<<calm>> Hello there.")],
        pipeline_params=EXPRESSIVE,
    )

    assert not tagger.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    # The LLM is still taught the tags, and its reply is spoken as it wrote it.
    assert not [f for f in down + up if isinstance(f, ExpressionMetadataFrame)]
    assert [f.text for f in down if isinstance(f, AggregatedTextFrame)] == ["<<calm>> Hello there."]


@pytest.mark.asyncio
async def test_the_pipeline_starts_once_the_model_has_loaded(monkeypatch):
    def slow(**kwargs):
        time.sleep(0.2)
        return FakeTaggerModel({"Hello there.": SentenceLabels(delivery="calm")})

    monkeypatch.setattr(PipecatToneTaggerModel, "download", slow)

    down, _ = await run_test(
        PipecatToneTagger(),
        frames_to_send=[ELEVENLABS, *_reply("Hello there.")],
        pipeline_params=EXPRESSIVE,
    )

    announced = next(i for i, f in enumerate(down) if isinstance(f, ExpressionMetadataFrame))
    reply = next(i for i, f in enumerate(down) if isinstance(f, AggregatedTextFrame))
    assert announced < reply
    assert down[reply].text == "<<calm>> Hello there."


@pytest.mark.asyncio
async def test_every_tagger_shares_one_model(monkeypatch):
    model = FakeTaggerModel()
    loads = []

    def download(**kwargs):
        loads.append(kwargs)
        return model

    monkeypatch.setattr(PipecatToneTaggerModel, "download", download)

    for _ in range(2):
        await run_test(PipecatToneTagger(), frames_to_send=[], pipeline_params=EXPRESSIVE)

    assert len(loads) == 1
