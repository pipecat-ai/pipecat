#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Sidechat's model, on the CPU."""

import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from loguru import logger

from pipecat.classifiers.sidechat.conversation import (
    INPUT_VERSION,
    TWO_TOWER_INPUT_VERSION,
    SidechatAgent,
    SidechatLine,
    encoder_text,
    line_text,
    numbers,
    turns_text,
)

try:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Pipecat Sidechat, you need to `uv add "pipecat-ai[sidechat]"`.')
    raise ImportError(f"Missing module: {e}") from e

SIDECHAT_REPO = "pipecat-ai/sidechat"
SIDECHAT_REVISION = "rc0.3.1"

#: Why a line was said, as the model tells them apart.
INTENTS = ("direct", "indirect", "open", "about", "other")

#: What the agent does with a line: speak now, speak if nobody else does
#: after a pause, or stay quiet.
SidechatDecision = Literal["respond", "hold", "ignore"]

#: Each variant's folder in a release.
_VARIANT_FOLDERS = {"base": "", "small": "agent-1x"}
_FILES = ("config.json", "tokenizer.json")
#: The graphs of a model that reads the turns and the line as one text, and
#: of one that reads them apart.
_GRAPHS = {INPUT_VERSION: ("model",), TWO_TOWER_INPUT_VERSION: ("context", "line")}
_SUFFIXES = {"fp32": ".onnx", "int8": "-int8.onnx"}

#: How many of the line's last tokens stand for it when truncation cut it off.
_LINE_FALLBACK_TOKENS = 16
#: How many encodings of earlier turns the small model keeps.
_KEPT_TURNS = 16


@dataclass(frozen=True)
class SidechatThresholds:
    """Where the model's probabilities turn into a decision.

    Parameters:
        respond: The ``respond`` probability at or above which the agent
            speaks now.
        hold: The ``respond`` probability at or above which the agent speaks
            if nobody else does after a pause.
        open: The ``open`` probability at or above which the agent speaks if
            nobody else does after a pause.
    """

    respond: float
    hold: float
    open: float


@dataclass(frozen=True)
class SidechatAnswer:
    """The model's reading of a line. Every number is a probability.

    Parameters:
        intent: Why the line was said, over :data:`INTENTS`, summing to 1:
            ``direct`` (said to the agent), ``indirect`` (wants the agent to
            act without speaking to it), ``open`` (a question or request to
            nobody in particular), ``about`` (mentions the agent and wants
            nothing from it) and ``other`` (for a person, the people, another
            device or nobody).
        respond: That the agent should speak now. A line the agent could take
            or leave, such as a question to the room that nothing earlier
            settles, reads about 0.5.
        unclear: That the line fits the agent and someone else equally.
    """

    intent: dict[str, float]
    respond: float
    unclear: float

    @property
    def addressed(self) -> float:
        """That the line was for the agent: its ``direct`` and ``indirect`` intents together."""
        return self.intent["direct"] + self.intent["indirect"]

    @property
    def open(self) -> float:
        """That the line is a question or request open to anyone."""
        return self.intent["open"]


class PipecatSidechatModel:
    """Pipecat Sidechat's model, on the CPU.

    The model reads one line of a group conversation with the turns before it
    and says whether the line was for the agent and whether the agent should
    speak. Its release's ``config.json`` holds the temperatures that calibrate
    its outputs and the thresholds its decisions were chosen with.

    A release has two variants. ``base`` reads the turns and the line
    together. ``small`` reads them apart: :meth:`prepare` encodes the turns
    before a line while it is still being said, and :meth:`answer` then reads
    only the line.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
    ):
        """Load the model from a folder.

        Args:
            model_dir: A folder with the model's ``config.json``,
                ``tokenizer.json`` and ONNX files.
            precision: ``"int8"`` for the quantized weights, which decide the
                same on nearly every line at about half the size, or
                ``"fp32"``.
            threads: The CPU threads for one read.

        Raises:
            ValueError: If the folder holds a model this class doesn't read.
        """
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        version = self.config.get("input_version")
        if version not in _GRAPHS:
            raise ValueError(
                f"{model_dir}: the model reads {version}, and this code builds "
                f"{' and '.join(_GRAPHS)}"
            )
        if self.config.get("text_turns"):
            raise ValueError(f"{model_dir}: the model reads a shortened history, not built here")
        self._two_tower = version == TWO_TOWER_INPUT_VERSION
        # The int8 weights come with thresholds of their own when the release chose them.
        thresholds = (precision == "int8" and self.config.get("thresholds_int8")) or self.config[
            "thresholds"
        ]
        self._thresholds = SidechatThresholds(
            respond=float(thresholds["respond"]),
            hold=float(thresholds["wait"]),
            open=float(thresholds["open"]),
        )
        self._temperatures = {name: float(t) for name, t in self.config["temperatures"].items()}
        self._max_tokens = int(self.config["max_len"])
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # The inputs are cut here by length, whatever the tokenizer's file says.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        sessions = [
            ort.InferenceSession(
                str(model_dir / f"{graph}{_SUFFIXES[precision]}"),
                options,
                providers=["CPUExecutionProvider"],
            )
            for graph in _GRAPHS[version]
        ]
        self._session = sessions[-1]
        self._turns_session = sessions[0] if self._two_tower else None
        if self._two_tower:
            self._turns_tokens = int(self.config["ctx_len"])
            self._line_tokens = int(self.config["line_len"])
        # The small model's encodings of earlier turns, by their text. A
        # prepare() can run on another thread than the answer() it is for.
        self._turns: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._turns_lock = threading.Lock()

    @classmethod
    def download(
        cls,
        *,
        revision: str = SIDECHAT_REVISION,
        variant: Literal["base", "small"] = "small",
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
    ) -> "PipecatSidechatModel":
        """Load the model from Hugging Face, downloading its files the first time.

        Args:
            revision: The model's release on Hugging Face, or ``"main"`` for
                the latest.
            variant: The model's size, ``"base"`` or ``"small"``.
            precision: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.

        Returns:
            The model.
        """
        folder = _VARIANT_FOLDERS[variant]
        version = INPUT_VERSION if variant == "base" else TWO_TOWER_INPUT_VERSION
        names = [*_FILES, *(f"{graph}{_SUFFIXES[precision]}" for graph in _GRAPHS[version])]
        root = snapshot_download(
            SIDECHAT_REPO,
            revision=revision,
            allow_patterns=[f"{folder}/{name}" if folder else name for name in names],
        )
        return cls(Path(root) / folder, precision=precision, threads=threads)

    @property
    def thresholds(self) -> SidechatThresholds:
        """The thresholds the model's release chose its decisions with, for these weights."""
        return self._thresholds

    @property
    def prepares(self) -> bool:
        """Whether :meth:`prepare` saves :meth:`answer` work: true of the small model."""
        return self._two_tower

    def prepare(self, agent: SidechatAgent, line: SidechatLine):
        """Encode the turns before a line ahead of the line itself.

        Only the small model reads the turns apart; for the base model this
        does nothing. The line's text and speaker aren't read, so it can be
        called as soon as the turn before the line ends.

        Args:
            agent: The agent.
            line: The line to come, of which only the turns before it are read.
        """
        if self._two_tower:
            self._turns_states(agent, line)

    def answer(self, agent: SidechatAgent, line: SidechatLine) -> SidechatAnswer:
        """Read one line.

        Args:
            agent: The agent.
            line: The line, with the turns before it.

        Returns:
            The model's reading of the line.
        """
        intent, respond, unclear = self._session.run(None, self._inputs(agent, line))
        t = self._temperatures
        logits = [float(x) / t["intent"] for x in cast(np.ndarray, intent)[0]]
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        total = sum(exps)
        return SidechatAnswer(
            intent={name: e / total for name, e in zip(INTENTS, exps)},
            respond=_sigmoid(float(cast(np.ndarray, respond)[0]) / t["respond"]),
            unclear=_sigmoid(float(cast(np.ndarray, unclear)[0]) / t["unclear"]),
        )

    def decide(
        self, answer: SidechatAnswer, thresholds: SidechatThresholds | None = None
    ) -> SidechatDecision:
        """What the agent does with a line, from the model's reading of it.

        Args:
            answer: The model's reading of the line.
            thresholds: Where to decide, by default the release's.

        Returns:
            ``"respond"`` to speak now, ``"hold"`` to speak if nobody else
            does after a pause, or ``"ignore"``.
        """
        return decide(answer.respond, answer.open, thresholds or self._thresholds)

    def _inputs(self, agent: SidechatAgent, line: SidechatLine) -> dict[str, np.ndarray]:
        """The model's inputs for one line."""
        num = np.array([numbers(agent, line)], dtype=np.float32)
        if self._two_tower:
            states, states_mask = self._turns_states(agent, line)
            ids, mask = self._ids(line_text(agent, line), self._line_tokens)
            return {"ids": ids, "mask": mask, "ctx": states, "ctx_mask": states_mask, "num": num}
        text, line_start = encoder_text(agent, line)
        encoding = self._tokenizer.encode(text)
        count = min(self._max_tokens, len(encoding.ids))
        ids = np.array([encoding.ids[:count]], dtype=np.int64)
        mask = np.ones_like(ids)
        # The tokens of the line itself, which the model pools apart from the
        # rest. Truncation can cut the line off, and then the last tokens stand
        # for it.
        line_mask = np.array(
            [[int(start >= line_start) for start, _ in encoding.offsets[:count]]], dtype=np.int64
        )
        if not line_mask.any():
            line_mask[0, max(0, count - _LINE_FALLBACK_TOKENS) : count] = 1
        return {"ids": ids, "mask": mask, "line_mask": line_mask, "num": num}

    def _ids(self, text: str, max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self._tokenizer.encode(text).ids[:max_tokens]], dtype=np.int64)
        return ids, np.ones_like(ids)

    def _turns_states(
        self, agent: SidechatAgent, line: SidechatLine
    ) -> tuple[np.ndarray, np.ndarray]:
        """The small model's encoding of the turns before a line, and its mask."""
        assert self._turns_session
        text = turns_text(
            agent, line, self._turns_tokens, lambda text: len(self._tokenizer.encode(text).ids)
        )
        with self._turns_lock:
            kept = self._turns.get(text)
            if kept is not None:
                self._turns.move_to_end(text)
                return kept
        ids, mask = self._ids(text, self._turns_tokens)
        states = cast(np.ndarray, self._turns_session.run(None, {"ids": ids, "mask": mask})[0])
        with self._turns_lock:
            self._turns[text] = (states, mask)
            while len(self._turns) > _KEPT_TURNS:
                self._turns.popitem(last=False)
        return states, mask


def decide(respond: float, open: float, thresholds: SidechatThresholds) -> SidechatDecision:
    """What the agent does with a line, from the model's probabilities for it.

    Args:
        respond: The probability that the agent should speak now.
        open: The probability that the line is a question or request open to anyone.
        thresholds: Where to decide.

    Returns:
        ``"respond"`` to speak now, ``"hold"`` to speak if nobody else does
        after a pause, or ``"ignore"``.
    """
    if respond >= thresholds.respond:
        return "respond"
    if respond >= thresholds.hold or open >= thresholds.open:
        return "hold"
    return "ignore"


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
