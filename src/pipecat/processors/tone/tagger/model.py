#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's tagger model, on the CPU."""

import json
import threading
from pathlib import Path
from typing import Literal, cast

import numpy as np
from loguru import logger

from pipecat.processors.tone.tagger._input import Layout, TurnContext, encode
from pipecat.processors.tone.tagger._labels import LabelSpace, SentenceLabels, Span

try:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Pipecat Tone, you need to `uv add "pipecat-ai[tone]"`.')
    raise ImportError(f"Missing module: {e}") from e

TAGGER_REPO = "pipecat-ai/tone-models"
TAGGER_REVISION = "tagger-rc1"

_INPUTS = ("input_ids", "attention_mask", "sentence_mask")
# A span or stressed word's tokens, as the model classes them.
_OUTSIDE, _BEGIN, _INSIDE = 0, 1, 2


class PipecatToneTaggerModel:
    """Pipecat Tone's tagger model, on the CPU. One model can serve every call."""

    def __init__(self, model_dir: str | Path, *, threads: int = 1):
        """Load the model from a folder.

        Args:
            model_dir: A folder with the model's ``config.json``,
                ``tokenizer.json`` and ONNX file.
            threads: The CPU threads for one sentence.
        """
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())
        self._layout = Layout(**self.config["layout"])
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_dir / self.config["onnx"]), options, providers=["CPUExecutionProvider"]
        )
        # Encoding sets the tokenizer's truncation, so calls from other threads take turns.
        self._encoding = threading.Lock()

    @property
    def breaths_per_reply(self) -> int | None:
        """The most breaths a reply takes, or None for no limit."""
        return self.config.get("breaths_per_reply")

    @property
    def breath_kinds(self) -> dict[str, float]:
        """The kinds a breath is drawn as, with their shares; empty for a plain breath."""
        return self.config.get("breath_kinds") or {}

    @classmethod
    def download(
        cls,
        *,
        size: Literal["17m", "32m"] = "17m",
        revision: str = TAGGER_REVISION,
        threads: int = 1,
    ) -> "PipecatToneTaggerModel":
        """Load the model from Hugging Face, downloading its files the first time.

        Args:
            size: The model's size, ``"17m"`` or ``"32m"``.
            revision: The model's release, such as ``"tagger-rc1"``.
            threads: The CPU threads for one sentence.

        Returns:
            The model.
        """
        folder = f"tagger/{size}"
        root = snapshot_download(TAGGER_REPO, revision=revision, allow_patterns=[f"{folder}/*"])
        return cls(Path(root) / folder, threads=threads)

    def tag(self, turn: TurnContext, earlier: list[str], sentence: str) -> SentenceLabels:
        """Label one sentence of a reply.

        Args:
            turn: The call so far, the labels it can use and the policy.
            earlier: The reply's earlier sentences, without tags.
            sentence: The sentence, without tags.

        Returns:
            The sentence's labels, from those the turn's label space holds.
        """
        with self._encoding:
            ids, attention, sentence_mask, offsets = self._encode(turn, earlier, sentence)
        feeds = {
            name: np.array([values], dtype=np.int64)
            for name, values in zip(_INPUTS, (ids, attention, sentence_mask))
        }
        heads = [cast(np.ndarray, head)[0] for head in self._session.run(None, feeds)]
        return self._decode(heads, offsets, sentence, turn.space)

    def _decode(
        self,
        heads: list[np.ndarray],
        offsets: list[tuple[int, int] | None],
        sentence: str,
        space: LabelSpace,
    ) -> SentenceLabels:
        """The labels from one sentence's heads, from those the space holds."""
        delivery, before, after, spans, emphasis, breath = heads
        sounds = self.config["sounds"]
        # Calibration shifts a sound's logit by the sentence's words:
        # [[most words, shift], ..., [None, shift]].
        bias = self.config.get("sound_bias") or {}
        words = sum(any(c.isalnum() for c in token) for token in sentence.split())
        shift = {sound: _shift(steps, words) for sound, steps in bias.items()}
        sound_shift = np.array([shift.get(sound, 0.0) for sound in sounds])
        # A breath has a head of its own, so the sound heads never choose one.
        others = [sound for sound in space.sounds if sound != "breath"]

        labels = SentenceLabels(
            delivery=_choose(delivery, self.config["deliveries"], space.deliveries),
            sound_before=_choose(before + sound_shift, sounds, others),
            sound_after=_choose(after + sound_shift, sounds, others),
        )
        # A breath goes before the sentence, where its logit, shifted, is above zero
        # and no other sound starts the sentence.
        if (
            not labels.sound_before
            and "breath" in space.sounds
            and breath[0] + shift.get("breath", 0.0) > 0
        ):
            labels.sound_before = "breath"
        n = len(offsets)
        if "slow" in space.deliveries:
            slow = _ranges(spans[:n].argmax(-1).tolist(), offsets, sentence)
            labels.spans = [Span(a, b, "slow") for a, b in slow]
        if space.emphasis:
            labels.emphasis = _ranges(emphasis[:n].argmax(-1).tolist(), offsets, sentence)
        return labels

    def _encode(self, turn: TurnContext, earlier: list[str], sentence: str):
        first, second, start = encode(turn, earlier, sentence, self._layout)
        encoding = self._encode_pair(first, second)
        offsets: list[tuple[int, int] | None] = []
        sentence_mask = []
        for seq, special, (a, b) in zip(
            encoding.sequence_ids, encoding.special_tokens_mask, encoding.offsets
        ):
            inside = seq == 1 and not special and a >= start and b > a
            sentence_mask.append(int(inside))
            offsets.append((a - start, b - start) if inside else None)
        return encoding.ids, encoding.attention_mask, sentence_mask, offsets

    def _encode_pair(self, first: str, second: str):
        layout = self._layout
        if layout.cut_call_first:
            self._tokenizer.enable_truncation(
                layout.max_tokens, strategy="only_first", direction="left"
            )
            # The tokenizer won't cut a text to nothing, so a call too long to
            # keep any of is left out.
            for call in (first, ""):
                try:
                    return self._tokenizer.encode(call, second)
                except Exception:
                    pass
        self._tokenizer.enable_truncation(
            layout.max_tokens, strategy="longest_first", direction="left"
        )
        return self._tokenizer.encode(first, second)


def _shift(steps: list, words: int) -> float:
    return next(shift for most, shift in steps if most is None or words <= most)


def _choose(logits: np.ndarray, classes: list[str], allowed) -> str | None:
    """The likeliest class the space allows, or None for no label."""
    mask = [c == "none" or c in allowed for c in classes]
    best = classes[int(np.where(mask, logits, -np.inf).argmax())]
    return None if best == "none" else best


def _ranges(
    classes: list[int], offsets: list[tuple[int, int] | None], sentence: str
) -> list[tuple[int, int]]:
    """Character ranges from begin, inside and outside token classes, widened to whole words."""
    ranges: list[list[int]] = []
    current: list[int] | None = None
    for cls, offset in zip(classes, offsets):
        if offset is None:
            continue
        if cls == _BEGIN or (cls == _INSIDE and current is None):
            if current:
                ranges.append(current)
            current = [offset[0], offset[1]]
        elif cls == _INSIDE and current:
            current[1] = offset[1]
        elif current:
            ranges.append(current)
            current = None
    if current:
        ranges.append(current)
    words = []
    for a, b in ranges:
        # A subword token's offsets can include the space before it.
        while a < b and sentence[a].isspace():
            a += 1
        while b > a and sentence[b - 1].isspace():
            b -= 1
        if a == b:
            continue
        while a > 0 and sentence[a - 1].isalnum():
            a -= 1
        while b < len(sentence) and sentence[b].isalnum():
            b += 1
        words.append((a, b))
    return words
