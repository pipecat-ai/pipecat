#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's reader model, on the CPU."""

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from loguru import logger

try:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Pipecat Tone, you need to `uv add "pipecat-ai[tone]"`.')
    raise ImportError(f"Missing module: {e}") from e

TONE_REPO = "pipecat-ai/tone-models"
TONE_REVISION = "reader-rc2"

_FOLDER = "reader"

_MODEL_FILES = {"fp16": "model_fp16.onnx", "fp32": "model.opt.onnx"}


class PipecatToneReaderModel:
    """Pipecat Tone's reader model, on the CPU."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        precision: Literal["fp16", "fp32"] = "fp16",
        threads: int = 4,
    ):
        """Load the model from a folder.

        Args:
            model_dir: A folder with the model's ``config.json``,
                ``tokenizer.json`` and ONNX file.
            precision: ``"fp16"`` for the weights in 16 bits, which give the
                same tags at half the size, or ``"fp32"``.
            threads: The CPU threads for one read.
        """
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(self.config["max_len"])
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_dir / _MODEL_FILES[precision]), options, providers=["CPUExecutionProvider"]
        )
        calibration = self.config.get("calibration") or {}
        self._emotion_temperature = float(calibration.get("emotion_temperature", 1.0))
        self._move_temperature = float(calibration.get("move_temperature", 1.0))
        self._sarcasm_curve = calibration.get("sarcasm")
        self._sarcasm_threshold = float(self.config.get("sarcasm_tau_raw", 0.5))

    @classmethod
    def download(
        cls,
        *,
        revision: str = TONE_REVISION,
        precision: Literal["fp16", "fp32"] = "fp16",
        threads: int = 4,
    ) -> "PipecatToneReaderModel":
        """Load the model from Hugging Face, downloading its files the first time.

        Args:
            revision: The model's revision.
            precision: ``"fp16"`` or ``"fp32"``.
            threads: The CPU threads for one read.

        Returns:
            The model.
        """
        files = ("config.json", "tokenizer.json", _MODEL_FILES[precision])
        root = snapshot_download(
            TONE_REPO,
            revision=revision,
            allow_patterns=[f"{_FOLDER}/{name}" for name in files],
        )
        return cls(Path(root) / _FOLDER, precision=precision, threads=threads)

    @property
    def emotions(self) -> list[str]:
        """The emotions the model reads."""
        return self.config["emotions"]

    def tag(self, text: str, previous: str | None = None) -> dict[str, Any]:
        """Read one turn.

        Args:
            text: The turn.
            previous: The turn before it, if there is one.

        Returns:
            The model's tags, in its own form: ``emotion``, ``move``,
            ``sarcasm`` and ``backchannel``.
        """
        rendered = f'"{text.strip()}"'
        if previous and previous.strip():
            rendered += f' Before: "{previous.strip()}"'
        encoding = self._tokenizer.encode(rendered)
        inputs = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
        results = self._session.run(None, inputs)
        outputs = dict(zip(self.config["outputs"], (cast(np.ndarray, r)[0] for r in results)))

        emotions = _softmax(outputs["emotion"], self._emotion_temperature)
        moves = _softmax(outputs["move"], self._move_temperature)
        emotion = self.config["emotions"][int(emotions.argmax())]
        intensity = self.config["intensities"][int(_softmax(outputs["emotion_intensity"]).argmax())]
        raw_sarcasm = float(_softmax(outputs["sarcasm"])[1])
        backchannel = float(_softmax(outputs["backchannel"])[1])
        return {
            "emotion": {
                "label": emotion,
                "intensity": "low" if emotion == "neutral" else intensity,
                "confidence": round(float(emotions.max()), 4),
                "probabilities": _probabilities(self.config["emotions"], emotions),
            },
            "move": {
                "label": self.config["moves"][int(moves.argmax())],
                "confidence": round(float(moves.max()), 4),
                "probabilities": _probabilities(self.config["moves"], moves),
            },
            "sarcasm": {
                "sarcastic": raw_sarcasm >= self._sarcasm_threshold,
                "probability": round(_platt(raw_sarcasm, self._sarcasm_curve), 4),
            },
            "backchannel": {
                "is_backchannel": backchannel >= 0.5,
                "probability": round(backchannel, 4),
            },
        }


def _softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def _probabilities(labels: list[str], probabilities: np.ndarray) -> dict[str, float]:
    return {label: round(float(p), 4) for label, p in zip(labels, probabilities)}


def _platt(p: float, curve: dict | None) -> float:
    """The calibrated chance, ``sigmoid(a * logit(p) + b)``, or ``p`` without a curve."""
    if not curve:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = curve["a"] * math.log(p / (1 - p)) + curve["b"]
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
