#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that tags an LLM's replies with Pipecat Tone."""

import time

import numpy as np
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
from pipecat.processors.tone.tagger import model as tagger_model
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
        breath_kinds: dict[str, float] | None = None,
    ):
        self.labels = labels or {}
        self.breaths_per_reply = breaths_per_reply
        self.breath_kinds = breath_kinds or {}
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
async def test_labels_are_written_by_their_universal_names(monkeypatch):
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
        "<<warm>> Congratulations! [[laugh]]",
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

    without, with_breathe = (turn.space for turn, _, _ in model.calls)
    assert "breath" not in without.sounds and without.breaths == {}
    assert with_breathe.sounds["breath"] == "breath"
    assert with_breathe.breaths == {
        "light breath": "light breath",
        "deep breath": "deep breath",
        "exhale": "exhale",
    }


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

    assert tagged == ["[[breath]] One.", "Two."]


BREATHE = _expressive(ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE]))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["light breath", "deep breath", "exhale"])
async def test_each_breath_is_drawn_as_a_kind_at_the_models_shares(monkeypatch, kind):
    model = FakeTaggerModel(
        {"One.": SentenceLabels(sound_before="breath")}, breath_kinds={kind: 1.0}
    )

    tagged = await _tagged(monkeypatch, model, ELEVENLABS, *_reply("One."), params=BREATHE)

    assert tagged == [f"[[{kind}]] One."]


@pytest.mark.asyncio
async def test_a_call_draws_its_breaths_alike_each_time(monkeypatch):
    breath = SentenceLabels(sound_before="breath")
    kinds = {"light breath": 0.59, "deep breath": 0.36, "exhale": 0.05}
    replies = [frame for n in range(8) for frame in _reply(f"Reply {n}.")]
    labels = {f"Reply {n}.": breath for n in range(8)}

    first, second = [
        await _tagged(
            monkeypatch,
            FakeTaggerModel(labels, breath_kinds=kinds),
            ELEVENLABS,
            *replies,
            params=BREATHE,
        )
        for _ in range(2)
    ]

    assert first == second
    assert {tagged.split("]]")[0] for tagged in first} <= {
        "[[light breath",
        "[[deep breath",
        "[[exhale",
    }


@pytest.mark.asyncio
async def test_the_next_reply_reads_a_drawn_breath_as_a_breath(monkeypatch):
    breath = SentenceLabels(sound_before="breath")
    model = FakeTaggerModel({"One.": breath}, breath_kinds={"exhale": 1.0})

    await _tagged(monkeypatch, model, ELEVENLABS, *_reply("One."), *_reply("Two."), params=BREATHE)

    second = model.calls[1][0]
    assert second.previous == [breath]


@pytest.mark.asyncio
async def test_a_policy_that_names_one_breath_draws_only_that_kind(monkeypatch):
    breath = SentenceLabels(sound_before="breath")
    model = FakeTaggerModel(
        {f"Reply {n}.": breath for n in range(6)}, breath_kinds={"light breath": 0.5, "exhale": 0.5}
    )
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE], sounds=["light breath"])
    replies = [frame for n in range(6) for frame in _reply(f"Reply {n}.")]

    tagged = await _tagged(monkeypatch, model, ELEVENLABS, *replies, params=_expressive(policy))

    assert tagged == [f"[[light breath]] Reply {n}." for n in range(6)]


@pytest.mark.asyncio
async def test_a_policy_that_names_only_a_kind_still_breathes(monkeypatch):
    breath = SentenceLabels(sound_before="breath")
    model = FakeTaggerModel(
        {"One.": breath}, breath_kinds={"light breath": 0.5, "deep breath": 0.5}
    )
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.BREATHE], sounds=["deep breath"])

    tagged = await _tagged(
        monkeypatch, model, ELEVENLABS, *_reply("One."), params=_expressive(policy)
    )

    assert tagged == ["[[deep breath]] One."]


@pytest.mark.asyncio
async def test_a_name_for_breath_is_written_for_every_breath(monkeypatch):
    model = FakeTaggerModel(
        {"One.": SentenceLabels(sound_before="breath")}, breath_kinds={"exhale": 1.0}
    )
    monkeypatch.setattr(PipecatToneTaggerModel, "download", lambda **kwargs: model)

    down, _ = await run_test(
        PipecatToneTagger(vocabulary={"breath": ["inhales deeply"]}),
        frames_to_send=[ELEVENLABS, *_reply("One.")],
        pipeline_params=BREATHE,
    )

    assert [f.text for f in down if isinstance(f, AggregatedTextFrame)] == [
        "[[inhales deeply]] One."
    ]


def test_a_models_breath_kinds_come_from_its_config():
    model = object.__new__(PipecatToneTaggerModel)

    model.config = {"breath_kinds": {"light breath": 0.6, "exhale": 0.4}}
    assert model.breath_kinds == {"light breath": 0.6, "exhale": 0.4}
    # A model without them writes a plain breath.
    model.config = {}
    assert model.breath_kinds == {}


def _decoded(before: list[float], breath: float, sentence: str = "Hi.") -> SentenceLabels:
    """A sentence's labels from a model's six heads: the sound before's logits and the breath head's."""
    model = object.__new__(PipecatToneTaggerModel)
    model.config = {
        "deliveries": ["none", "calm"],
        "sounds": ["none", "laugh", "chuckle", "sigh", "breath"],
        "sound_bias": {"breath": [[5, -1.0], [None, 0.0]]},
    }
    space = LabelSpace({"calm": "calm"}, {"laugh": "laughs", "breath": "quick breath"}, False)
    quiet = np.array([9.0, 0.0, 0.0, 0.0, 0.0])
    tokens = np.array([[9.0, 0.0, 0.0]])
    heads = [
        np.array([9.0, 0.0]),
        np.array(before),
        quiet,
        tokens,
        tokens,
        np.array([breath]),
    ]
    return model._decode(heads, [(0, 2)], sentence, space)


def test_a_breath_comes_from_its_own_head_shifted_for_the_sentences_length():
    quiet = [9.0, 0.0, 0.0, 0.0, 0.0]

    # Up to five words, the breath head's logit is shifted down by one.
    assert _decoded(quiet, 0.5).sound_before is None
    assert _decoded(quiet, 0.5, "Hi " * 6).sound_before == "breath"


def test_the_sound_heads_never_choose_a_breath():
    breath_class_wins = [0.0, 0.0, 0.0, 0.0, 9.0]

    assert _decoded(breath_class_wins, -1.0, "Hi " * 6).sound_before is None


def test_a_breath_goes_only_where_no_other_sound_starts_the_sentence():
    laugh = [0.0, 9.0, 0.0, 0.0, 0.0]

    assert _decoded(laugh, 9.0, "Hi " * 6).sound_before == "laugh"


def test_a_sentence_is_read_as_the_text_the_model_was_trained_on():
    policy = ExpressionPolicy(
        guidance=[
            ExpressionGuidance.MATCH_STYLE,
            ExpressionGuidance.SLOW_FOR_DETAILS,
            ExpressionGuidance.BREATHE,
        ]
    )
    heard = [PerceptionSignal("tone", labels=[("frustrated", 0.62), ("neutral", 0.3)])]
    space = LabelSpace.resolve(
        ElevenLabsAudioTagExpressionAdapter().capabilities(), policy, "ElevenLabsTTSService#0"
    )
    turn = TurnContext(
        [
            Message("assistant", "Thanks for calling. How can I help?"),
            Message("user", "I need my new booking code.", heard),
        ],
        space,
        policy,
        [SentenceLabels(delivery="calm"), SentenceLabels(sound_before="breath")],
    )
    # tagger-rc1's layout.
    layout = Layout(earlier=1, max_tokens=160, previous_tags=True)

    call, sentence, start = encode(turn, ["Sure, one moment."], "Your new code is AB12CD.", layout)

    # What training wrote for this turn.
    assert call == "caller (tone frustrated likely, neutral guess): I need my new booking code."
    assert sentence == (
        "tts elevenlabs | can calm sympathetic apologetic warm excited surprised sad grateful "
        "curious confident slow fast | sounds laugh chuckle sigh breath | stress yes | policy "
        "match style, slow for details, breathe | caller sounded tone frustrated likely, neutral "
        "guess | last reply calm breath | state new detail | reply so far: Sure, one moment. | "
        "sentence: Your new code is AB12CD."
    )
    assert start == 354


def test_a_service_the_model_wasnt_trained_with_is_read_as_other():
    capabilities = ElevenLabsAudioTagExpressionAdapter().capabilities()

    assert LabelSpace.resolve(capabilities, ExpressionPolicy(), "ElevenLabsTTSService#0").tts == (
        "elevenlabs"
    )
    assert LabelSpace.resolve(capabilities, ExpressionPolicy(), "MyTTSService#0").tts == "other"


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


@pytest.mark.asyncio
async def test_the_model_downloads_at_the_size_and_release_asked_for(monkeypatch):
    asked = []

    def download(**kwargs):
        asked.append(kwargs)
        return FakeTaggerModel()

    monkeypatch.setattr(PipecatToneTaggerModel, "download", download)
    for tagger in (PipecatToneTagger(), PipecatToneTagger(size="32m", revision="tagger-rc9")):
        await run_test(tagger, frames_to_send=[], pipeline_params=EXPRESSIVE)

    assert asked == [
        {"size": "17m", "revision": "tagger-rc1", "threads": 1},
        {"size": "32m", "revision": "tagger-rc9", "threads": 1},
    ]


def test_each_size_downloads_its_own_folder(monkeypatch, tmp_path):
    seen = {}

    def snapshot_download(repo, revision, allow_patterns):
        seen.update(repo=repo, revision=revision, patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(tagger_model, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        PipecatToneTaggerModel,
        "__init__",
        lambda self, model_dir, threads: seen.update(dir=model_dir),
    )

    PipecatToneTaggerModel.download(size="32m")

    assert seen == {
        "repo": "pipecat-ai/tone-models",
        "revision": "tagger-rc1",
        "patterns": ["tagger/32m/*"],
        "dir": tmp_path / "tagger" / "32m",
    }


@pytest.mark.asyncio
async def test_a_folder_of_your_own_loads_in_place_of_a_download(monkeypatch, tmp_path):
    loaded = []

    def fail(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(PipecatToneTaggerModel, "download", fail)
    monkeypatch.setattr(
        PipecatToneTaggerModel,
        "__init__",
        lambda self, model_dir, threads: loaded.append((model_dir, threads)),
    )

    for tagger in [PipecatToneTagger(model_dir=str(tmp_path), threads=2) for _ in range(2)]:
        down, _ = await run_test(tagger, frames_to_send=[], pipeline_params=EXPRESSIVE)

    # Loaded once a process, and shared.
    assert loaded == [(tmp_path, 2)]
    assert [f for f in down if isinstance(f, ExpressionMetadataFrame)]


@pytest.mark.asyncio
async def test_a_vocabulary_names_a_label_first_where_the_service_performs_it(monkeypatch):
    model = FakeTaggerModel()
    monkeypatch.setattr(PipecatToneTaggerModel, "download", lambda **kwargs: model)
    vocabulary = {"warm": ["cheerful"], "laugh": ["giggles"]}

    for tts in (ELEVENLABS, CARTESIA):
        monkeypatch.setattr(_models, "_models", {})
        await run_test(
            PipecatToneTagger(vocabulary=vocabulary),
            frames_to_send=[tts, *_reply("Hi.")],
            pipeline_params=EXPRESSIVE,
        )

    elevenlabs, cartesia = (turn.space for turn, _, _ in model.calls)
    # ElevenLabs takes free text, so it performs both names.
    assert elevenlabs.deliveries["warm"] == "cheerful"
    assert elevenlabs.sounds["laugh"] == "giggles"
    # Cartesia performs neither, so the universal names stand.
    assert cartesia.deliveries["warm"] == "warm"
    assert cartesia.sounds["laugh"] == "laugh"


def test_a_vocabulary_for_a_label_the_tagger_doesnt_write_is_refused():
    with pytest.raises(ValueError, match="cheerful"):
        PipecatToneTagger(vocabulary={"cheerful": ["happy"]})
