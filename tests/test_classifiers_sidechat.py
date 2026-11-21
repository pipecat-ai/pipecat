#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Pipecat Sidechat's inputs, its model and its classifier."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from pipecat.classifiers.base_classifier import (
    ChoiceResult,
    ClassifierError,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.sidechat import model as sidechat_model
from pipecat.classifiers.sidechat.classifier import PipecatSidechatClassifier
from pipecat.classifiers.sidechat.conversation import (
    INPUT_VERSION,
    TWO_TOWER_INPUT_VERSION,
    SidechatAgent,
    SidechatLine,
    SidechatTurn,
    encoder_text,
    line_text,
    mentions,
    numbers,
    turns_text,
)
from pipecat.classifiers.sidechat.model import (
    SIDECHAT_REVISION,
    PipecatSidechatModel,
    SidechatAnswer,
    SidechatThresholds,
    decide,
)
from pipecat.utils import models as shared_models
from pipecat.utils.asyncio.task_manager import TaskManager

ATLAS = SidechatAgent(name="Atlas", aliases=("at last",), role="AI assistant on the call")

THRESHOLDS = SidechatThresholds(respond=0.6, hold=0.4, open=0.3)


def _answer(respond: float, **intent: float) -> SidechatAnswer:
    shares = {"direct": 0.0, "indirect": 0.0, "open": 0.0, "about": 0.0, "other": 0.0, **intent}
    return SidechatAnswer(intent=shares, respond=respond, unclear=0.1)


RESPOND = _answer(0.95, direct=0.85, indirect=0.05, other=0.1)
HOLD = _answer(0.5, open=0.7, other=0.3)
IGNORE = _answer(0.02, about=0.1, other=0.9)


class FakeSidechatModel:
    """Answers each line by its text, and remembers the lines it was given."""

    thresholds = THRESHOLDS

    def __init__(
        self,
        answers: dict[str, SidechatAnswer] | None = None,
        *,
        error: Exception | None = None,
    ):
        self.answers = answers or {}
        self.lines: list[SidechatLine] = []
        self.prepared: list[SidechatLine] = []
        self.prepares = True
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


# Inputs


def test_the_encoder_reads_the_agents_card_then_the_turns_by_role():
    line = SidechatLine(
        text="And by train?",
        speaker="Speaker 1",
        history=[
            SidechatTurn(text="Morning.", speaker="Speaker 2"),
            SidechatTurn(text="Atlas, how far is Leeds from York?", speaker="Speaker 1"),
            SidechatTurn(text="About twenty-five miles.", agent=True),
        ],
    )

    text, line_start = encoder_text(ATLAS, line)

    assert text == (
        "agent: Atlas | also: at last | AI assistant on the call\n"
        "[other1] Morning.\n"
        "[speaker] Atlas, how far is Leeds from York?\n"
        "[agent] About twenty-five miles.\n"
        ">> [speaker] And by train?"
    )
    assert text[line_start:] == ">> [speaker] And by train?"


def test_turns_are_unknown_when_the_lines_speaker_is():
    line = SidechatLine(
        text="Can you look it up?",
        history=[SidechatTurn(text="Hi.", speaker="Speaker 2")],
        over_agent=True,
    )

    text, _ = encoder_text(SidechatAgent(name="Atlas"), line)

    assert text == "agent: Atlas\n[unknown] Hi.\n>> [unknown] (over the agent) Can you look it up?"


HISTORY = [
    SidechatTurn(text="Morning.", speaker="Speaker 2"),
    SidechatTurn(text="Atlas, how far is Leeds from York?", speaker="Speaker 1"),
    SidechatTurn(text="About twenty-five miles.", agent=True),
]


def test_the_small_model_reads_the_turns_apart_from_the_line():
    line = SidechatLine(text="And by train?", speaker="Speaker 1", history=HISTORY)

    assert turns_text(ATLAS, line, 320, lambda text: len(text.split())) == (
        "agent: Atlas | also: at last | AI assistant on the call\n"
        "[p1] Morning.\n"
        "[p2] Atlas, how far is Leeds from York?\n"
        "[agent] About twenty-five miles."
    )
    assert line_text(ATLAS, line) == (
        "agent: Atlas | also: at last | AI assistant on the call\n>> [p2] And by train?"
    )


def test_the_turns_dont_depend_on_the_line():
    said = SidechatLine(text="And by train?", speaker="Speaker 1", history=HISTORY)
    to_come = SidechatLine(text="", history=HISTORY)

    count = lambda text: len(text.split())  # noqa: E731
    assert turns_text(ATLAS, to_come, 320, count) == turns_text(ATLAS, said, 320, count)


def test_the_oldest_turns_are_left_out_until_the_turns_fit():
    line = SidechatLine(text="And by train?", history=HISTORY)

    text = turns_text(SidechatAgent(name="Atlas"), line, 8, lambda text: len(text.split()))

    # The person left is numbered from the turns that are read.
    assert text == "agent: Atlas\n[agent] About twenty-five miles."


def test_a_line_before_any_turn_and_a_speaker_new_to_the_turns():
    agent = SidechatAgent(name="Atlas")

    first = SidechatLine(text="Hello?", speaker="Speaker 3")
    assert turns_text(agent, first, 320, len) == "agent: Atlas\n(start of the conversation)"
    assert line_text(agent, first) == "agent: Atlas\n>> [new] Hello?"
    unknown = SidechatLine(text="Hello?", history=HISTORY, over_agent=True)
    assert line_text(agent, unknown) == "agent: Atlas\n>> [unknown] (over the agent) Hello?"


def test_the_numbers_read_the_rhythm_of_the_turns_and_the_lines_words():
    line = SidechatLine(
        text="Atlas, and by train?",
        speaker="Speaker 1",
        history=[
            SidechatTurn(text="How far is Leeds?", speaker="Speaker 1"),
            SidechatTurn(text="About forty miles. Anything else?", agent=True),
        ],
    )

    values = numbers(ATLAS, line)

    assert len(values) == 31
    assert values[:5] == [1.0, 0.0, 0.0, 0.0, 0.0]  # the agent spoke just before
    assert values[5] == 1.0  # and asked a question
    assert values[6:9] == [1.0, 0.0, 0.0]  # this speaker is the one it answered
    assert values[14:17] == [0.0, 0.0, 1.0]  # whether it was said over the agent isn't known
    assert values[17:19] == [1.0, 0.0]  # speaker known, no people listed
    assert values[19:22] == [1.0, 1.0, 0.0]  # the agent is named, first


def test_only_the_last_eight_turns_are_read():
    history = [SidechatTurn(text=f"Line {i}.", speaker="Speaker 2") for i in range(12)]

    text, _ = encoder_text(ATLAS, SidechatLine(text="Hm.", speaker="Speaker 1", history=history))

    assert "Line 3." not in text
    assert text.count("[other1]") == 8


def test_the_agent_is_named_by_its_name_or_an_alias():
    assert mentions(ATLAS, "Atlas, what time is it?")
    assert mentions(ATLAS, "at last a number that makes sense")
    assert not mentions(ATLAS, "atlases are heavy")


# The model


def _tokenizer(folder):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.save(str(folder / "tokenizer.json"))


CONFIG = {
    "input_version": INPUT_VERSION,
    "max_len": 320,
    "temperatures": {"intent": 2.0, "respond": 2.0, "unclear": 1.0},
    "thresholds": {"respond": 0.6, "wait": 0.4, "open": 0.3},
    "thresholds_int8": {"respond": 0.55, "wait": 0.4, "open": 0.3},
}


class FakeSession:
    """Stands in for the ONNX session: gives fixed logits, and remembers its inputs."""

    def __init__(self, path, options, providers):
        self.path = path
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return [
            np.array([[2.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        ]


def _model(
    monkeypatch, tmp_path, precision="fp32", **config
) -> tuple[PipecatSidechatModel, FakeSession]:
    """A model loaded from a folder with CONFIG, changed by ``config``, and its session."""
    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, **config}), encoding="utf-8")
    _tokenizer(tmp_path)
    monkeypatch.setattr(sidechat_model.ort, "InferenceSession", FakeSession)
    model = PipecatSidechatModel(tmp_path, precision=precision)
    return model, model._session


LINE = SidechatLine(text="Atlas, what time is it?", speaker="Speaker 1")


def test_the_model_reads_the_line_apart_from_the_rest(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path)

    model.answer(ATLAS, LINE)

    inputs = session.inputs[0]
    assert session.path == str(tmp_path / "model.onnx")
    # [CLS], the card's 12 words, the line's 7 and [SEP].
    assert inputs["ids"].shape == (1, 21)
    assert inputs["mask"].tolist() == [[1] * 21]
    assert inputs["line_mask"].tolist() == [[0] * 13 + [1] * 7 + [0]]
    assert inputs["num"].dtype == np.float32
    assert inputs["num"][0] == pytest.approx(numbers(ATLAS, LINE))


SMALL = {"input_version": TWO_TOWER_INPUT_VERSION, "ctx_len": 320, "line_len": 96}


class FakeTurnsSession(FakeSession):
    """Stands in for the small model's session over the turns: one state per token."""

    def run(self, output_names, inputs):
        self.inputs.append(inputs)
        return [np.ones((1, inputs["ids"].shape[1], 4), dtype=np.float32)]


def _small_model(monkeypatch, tmp_path, precision="fp32", **config):
    """A small model, with its sessions over the turns and over the line."""
    (tmp_path / "config.json").write_text(
        json.dumps({**CONFIG, **SMALL, **config}), encoding="utf-8"
    )
    _tokenizer(tmp_path)
    monkeypatch.setattr(
        sidechat_model.ort,
        "InferenceSession",
        lambda path, options, providers: (FakeTurnsSession if "context" in path else FakeSession)(
            path, options, providers
        ),
    )
    model = PipecatSidechatModel(tmp_path, precision=precision)
    return model, model._turns_session, model._session


def test_the_small_model_reads_the_line_with_the_turns_it_encoded(monkeypatch, tmp_path):
    model, turns, line = _small_model(monkeypatch, tmp_path, precision="int8")
    said = SidechatLine(text="And by train?", speaker="Speaker 1", history=HISTORY)

    model.answer(ATLAS, said)

    assert turns.path == str(tmp_path / "context-int8.onnx")
    assert line.path == str(tmp_path / "line-int8.onnx")
    # [CLS], the card's 12 words, the turns' 14 and [SEP].
    assert turns.inputs[0]["ids"].shape == (1, 28)
    inputs = line.inputs[0]
    # [CLS], the card's 12 words, the line's 5 and [SEP].
    assert inputs["ids"].shape == (1, 19)
    assert inputs["ctx"].shape == (1, 28, 4)
    assert inputs["ctx_mask"].tolist() == [[1] * 28]
    assert inputs["num"][0] == pytest.approx(numbers(ATLAS, said))
    assert "line_mask" not in inputs


def test_turns_prepared_ahead_arent_encoded_again(monkeypatch, tmp_path):
    model, turns, _ = _small_model(monkeypatch, tmp_path)

    assert model.prepares
    model.prepare(ATLAS, SidechatLine(text="", history=HISTORY))
    model.answer(ATLAS, SidechatLine(text="And by train?", speaker="Speaker 1", history=HISTORY))
    model.answer(ATLAS, SidechatLine(text="Or by bus?", speaker="Speaker 2", history=HISTORY))

    assert len(turns.inputs) == 1


def test_the_small_models_texts_are_cut_to_their_lengths(monkeypatch, tmp_path):
    model, turns, line = _small_model(monkeypatch, tmp_path, ctx_len=20, line_len=6)

    model.answer(ATLAS, SidechatLine(text="And by train?", speaker="Speaker 1", history=HISTORY))

    # The oldest turns are left out whole, and the line is cut.
    assert turns.inputs[0]["ids"].shape == (1, 18)
    assert line.inputs[0]["ids"].shape == (1, 6)


def test_the_base_model_has_nothing_to_prepare(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path)

    model.prepare(ATLAS, SidechatLine(text="", history=HISTORY))

    assert not model.prepares
    assert session.inputs == []


def test_a_line_cut_off_by_the_token_limit_is_read_from_the_last_tokens(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path, max_len=6)

    model.answer(ATLAS, LINE)

    assert session.inputs[0]["line_mask"].tolist() == [[1] * 6]


def test_the_models_outputs_are_calibrated_by_its_temperatures(monkeypatch, tmp_path):
    model, _ = _model(monkeypatch, tmp_path)

    answer = model.answer(ATLAS, LINE)

    e = np.exp(1.0)
    assert answer.intent["direct"] == pytest.approx(e / (e + 4))
    assert sum(answer.intent.values()) == pytest.approx(1.0)
    assert answer.respond == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert answer.unclear == pytest.approx(0.5)
    assert answer.addressed == pytest.approx(answer.intent["direct"] + answer.intent["indirect"])


def test_int8_weights_come_with_thresholds_of_their_own(monkeypatch, tmp_path):
    model, session = _model(monkeypatch, tmp_path, precision="int8")

    assert session.path == str(tmp_path / "model-int8.onnx")
    assert model.thresholds == SidechatThresholds(respond=0.55, hold=0.4, open=0.3)


def test_int8_weights_without_thresholds_of_their_own_use_the_full_ones(monkeypatch, tmp_path):
    model, _ = _model(monkeypatch, tmp_path, precision="int8", thresholds_int8=None)

    assert model.thresholds == THRESHOLDS


@pytest.mark.parametrize(
    "config",
    [
        {"input_version": "sidechat-input-0.1"},
        {"text_turns": 4},
    ],
)
def test_a_model_this_code_doesnt_read_isnt_loaded(monkeypatch, tmp_path, config):
    with pytest.raises(ValueError):
        _model(monkeypatch, tmp_path, **config)


def test_the_model_downloads_at_the_release_asked_for(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    _tokenizer(tmp_path)
    monkeypatch.setattr(sidechat_model.ort, "InferenceSession", FakeSession)
    asked = {}

    def snapshot_download(repo, *, revision, allow_patterns):
        asked.update(repo=repo, revision=revision, allow_patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(sidechat_model, "snapshot_download", snapshot_download)

    PipecatSidechatModel.download(revision="rc9", variant="base", precision="int8")

    assert asked == {
        "repo": "pipecat-ai/sidechat",
        "revision": "rc9",
        "allow_patterns": ["config.json", "tokenizer.json", "model-int8.onnx"],
    }


def test_the_small_model_downloads_from_its_folder_of_the_release(monkeypatch, tmp_path):
    folder = tmp_path / "agent-1x"
    folder.mkdir()
    _small_model(monkeypatch, folder)
    asked = {}

    def snapshot_download(repo, *, revision, allow_patterns):
        asked.update(revision=revision, allow_patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(sidechat_model, "snapshot_download", snapshot_download)

    model = PipecatSidechatModel.download(precision="fp32")

    assert model.prepares
    assert asked == {
        "revision": SIDECHAT_REVISION,
        "allow_patterns": [
            "agent-1x/config.json",
            "agent-1x/tokenizer.json",
            "agent-1x/context.onnx",
            "agent-1x/line.onnx",
        ],
    }


@pytest.mark.parametrize(
    "respond, open_, decision",
    [(0.6, 0.0, "respond"), (0.4, 0.0, "hold"), (0.1, 0.3, "hold"), (0.39, 0.29, "ignore")],
)
def test_a_decision_comes_from_respond_then_open(respond, open_, decision):
    assert decide(respond, open_, THRESHOLDS) == decision


# The classifier

STATE = {
    "text": "And by train?",
    "speaker": "Speaker 1",
    "history": [
        {"speaker": "Speaker 1", "text": "Atlas, how far is Leeds?"},
        {"text": "About forty miles.", "agent": True},
    ],
}


async def _classifier(monkeypatch, model: FakeSidechatModel, **kwargs) -> PipecatSidechatClassifier:
    """A classifier set up with ``model`` as its download."""
    monkeypatch.setattr(PipecatSidechatModel, "download", lambda **kwargs: model)
    classifier = PipecatSidechatClassifier(agent_name="Atlas", **kwargs)
    await classifier.setup(TaskManager())
    return classifier


def test_the_questions_are_known_before_the_model_loads():
    classifier = PipecatSidechatClassifier(agent_name="Atlas")

    assert list(classifier.questions) == ["respond", "addressed", "open", "intent", "unclear"]
    assert classifier.model == f"pipecat-ai/sidechat@{SIDECHAT_REVISION}/small"
    with pytest.raises(ClassifierError):
        _ = classifier.thresholds


@pytest.mark.asyncio
async def test_nothing_is_answered_before_the_model_loads():
    classifier = PipecatSidechatClassifier(agent_name="Atlas")

    with pytest.raises(ClassifierError):
        await classifier.ask(STATE, classifier.questions)


@pytest.mark.asyncio
async def test_each_question_is_answered_from_one_read_of_the_line(monkeypatch):
    model = FakeSidechatModel({"And by train?": RESPOND})
    classifier = await _classifier(monkeypatch, model)

    results = await classifier.ask(STATE, classifier.questions)

    assert len(model.lines) == 1
    assert results["respond"] == YesNoResult(probability=0.95)
    assert results["addressed"].probability == pytest.approx(0.9)
    assert results["open"] == YesNoResult(probability=0.0)
    assert results["unclear"] == YesNoResult(probability=0.1)
    intent = results["intent"]
    assert isinstance(intent, ChoiceResult)
    assert (intent.choice, intent.confidence) == ("direct", 0.85)
    assert classifier.thresholds == THRESHOLDS


@pytest.mark.asyncio
async def test_the_state_is_read_as_the_line_and_the_turns_before_it(monkeypatch):
    model = FakeSidechatModel()
    classifier = await _classifier(monkeypatch, model, aliases=["at last"])

    await classifier.ask(
        {**STATE, "people": ["Sam"], "over_agent": True}, {"r": classifier.questions["respond"]}
    )
    await classifier.ask({**STATE, "over_agent": False}, {"r": classifier.questions["respond"]})

    first, second = model.lines
    assert first == SidechatLine(
        text="And by train?",
        speaker="Speaker 1",
        history=(
            SidechatTurn(text="Atlas, how far is Leeds?", speaker="Speaker 1"),
            SidechatTurn(text="About forty miles.", agent=True),
        ),
        people=("Sam",),
        over_agent=True,
    )
    # The model was trained knowing only speech over the agent, never its absence.
    assert second.over_agent is None
    assert classifier.agent == SidechatAgent(
        name="Atlas", aliases=("at last",), role="AI assistant on the call"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["And by train?", {"speaker": "Speaker 1"}, {"text": "Hm.", "history": [{"a": 1}]}]
)
async def test_a_state_without_a_line_is_refused(monkeypatch, state):
    model = FakeSidechatModel()
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError):
        await classifier.ask(state, classifier.questions)
    assert model.lines == []


@pytest.mark.asyncio
async def test_a_question_of_its_own_making_is_refused_before_reading(monkeypatch):
    model = FakeSidechatModel()
    classifier = await _classifier(monkeypatch, model)

    with pytest.raises(ClassifierError):
        await classifier.ask(STATE, {"for_bot": YesNoQuestion(instructions="Is it for the bot?")})
    assert model.lines == []


@pytest.mark.asyncio
async def test_a_failed_read_raises_a_classifier_error(monkeypatch):
    classifier = await _classifier(monkeypatch, FakeSidechatModel(error=RuntimeError("boom")))

    with pytest.raises(ClassifierError):
        await classifier.ask(STATE, classifier.questions)


@pytest.mark.asyncio
async def test_a_model_that_cant_load_fails_setup(monkeypatch):
    def download(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(PipecatSidechatModel, "download", download)

    with pytest.raises(ClassifierError):
        await PipecatSidechatClassifier(agent_name="Atlas").setup(TaskManager())


@pytest.mark.asyncio
async def test_classifiers_share_the_loaded_model_of_their_weights(monkeypatch):
    loads = []

    def download(**kwargs):
        loads.append(kwargs)
        return FakeSidechatModel()

    monkeypatch.setattr(PipecatSidechatModel, "download", download)
    for precision in ("fp32", "fp32", "int8"):
        await PipecatSidechatClassifier(agent_name="Atlas", precision=precision).setup(
            TaskManager()
        )

    assert [kwargs["precision"] for kwargs in loads] == ["fp32", "int8"]


@pytest.mark.asyncio
async def test_each_variant_is_its_own_model(monkeypatch):
    loads = []

    def download(**kwargs):
        loads.append(kwargs)
        return FakeSidechatModel()

    monkeypatch.setattr(PipecatSidechatModel, "download", download)
    for variant in ("small", "base", "small"):
        await PipecatSidechatClassifier(agent_name="Atlas", variant=variant).setup(TaskManager())

    assert [kwargs["variant"] for kwargs in loads] == ["small", "base"]


@pytest.mark.asyncio
async def test_the_turns_before_a_line_are_read_ahead(monkeypatch):
    model = FakeSidechatModel()
    classifier = await _classifier(monkeypatch, model)

    await classifier.prepare({"history": [{"speaker": "Speaker 1", "text": "Morning."}]})

    assert model.lines == []
    assert [line.history for line in model.prepared] == [
        (SidechatTurn(text="Morning.", speaker="Speaker 1"),)
    ]


@pytest.mark.asyncio
async def test_a_model_that_reads_a_line_whole_isnt_prepared(monkeypatch):
    model = FakeSidechatModel()
    model.prepares = False
    classifier = await _classifier(monkeypatch, model)

    await classifier.prepare({"history": [{"text": "Morning."}]})

    assert model.prepared == []
