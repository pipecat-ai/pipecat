#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone: how each user turn was meant, read from its words.

Part of expressive conversation (alpha). Pipecat Tone
(https://huggingface.co/pipecat-ai/tone) reads one turn of text, with the turn
before it, and returns the speaker's emotion, whether the turn is sarcastic,
the move it makes and whether it's only a listening sound, each with a
calibrated confidence. It runs on the CPU in a few milliseconds. The inference
here follows the model's reference runtime (Apache-2.0).

**Alpha.** These types are likely to change between releases.
"""

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from loguru import logger

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.formatter import ExpressionFormatter
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TextType,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Tone, you need to `uv add "pipecat-ai[tone]"`.')
    raise ImportError(f"Missing module: {e}") from e

TONE_REPO = "pipecat-ai/tone"
TONE_REVISION = "rc1"

_MODEL_FILES = {"fp16": "model_fp16.onnx", "fp32": "model.opt.onnx"}

_EMOTION = (
    "The feeling the words show. Read from the words alone, so neutral means they show no "
    "strong feeling, not that the user feels none: go by what they say."
)
_SARCASM = (
    "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: "
    "a sarcastic turn can read happy."
)


class ToneModel:
    """Pipecat Tone's model, on the CPU."""

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
    ) -> "ToneModel":
        """Load the model from Hugging Face, downloading its files the first time.

        Args:
            revision: The model's revision.
            precision: ``"fp16"`` or ``"fp32"``.
            threads: The CPU threads for one read.

        Returns:
            The model.
        """
        model_dir = snapshot_download(
            TONE_REPO,
            revision=revision,
            allow_patterns=["config.json", "tokenizer.json", _MODEL_FILES[precision]],
        )
        return cls(model_dir, precision=precision, threads=threads)

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


class ToneProcessor(FrameProcessor):
    """Reads how each user turn was meant with Pipecat Tone.

    Goes between the STT service and the user aggregator. Each final
    transcription is read, with the bot's reply before it, and the reading is
    set as its ``perception``: the turn's ``emotion``, its two likeliest labels
    with their confidences, and its ``sarcasm``, the chance it's sarcastic.
    The read runs off the event loop, and the transcription waits for it, a
    few milliseconds.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        pipeline = Pipeline([transport.input(), stt, ToneProcessor(), user_aggregator, llm, ...])
    """

    def __init__(
        self,
        *,
        model: ToneModel | None = None,
        precision: Literal["fp16", "fp32"] = "fp16",
        threads: int = 4,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            model: The model to read with, or None to download Pipecat Tone
                from Hugging Face at start.
            precision: The weights to download: ``"fp16"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(**kwargs)
        self._model = model
        self._precision: Literal["fp16", "fp32"] = precision
        self._threads = threads
        self._previous: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Read final transcriptions, and keep the bot's replies to read them against.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
            await self.push_frame(
                PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
            )
            return
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            frame.perception = await self._read(frame.text)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._previous = _words(frame.text)
        await self.push_frame(frame, direction)

    async def _start(self):
        if self._model is None:
            self._model = await asyncio.to_thread(
                ToneModel.download, precision=self._precision, threads=self._threads
            )

    def _capabilities(self) -> PerceptionCapabilities:
        assert self._model
        return PerceptionCapabilities(
            signals=[
                PerceptionTerm("emotion", _EMOTION, labels=list(self._model.emotions)),
                PerceptionTerm("sarcasm", _SARCASM),
            ]
        )

    async def _read(self, text: str) -> list[PerceptionSignal] | None:
        assert self._model
        try:
            tags = await asyncio.to_thread(self._model.tag, text, self._previous)
        except Exception as e:
            await self.push_error(f"Tone couldn't read the turn: {e}", e)
            return None
        likeliest = sorted(tags["emotion"]["probabilities"].items(), key=lambda p: -p[1])[:2]
        return [
            PerceptionSignal("emotion", labels=likeliest),
            PerceptionSignal("sarcasm", value=round(tags["sarcasm"]["probability"], 2)),
        ]


def _words(text: str) -> str:
    """The words of the bot's reply, without its expression tags."""
    segments = ExpressionFormatter(BaseExpressionAdapter, owner="Tone").parse(text)
    return " ".join(ExpressionFormatter.shown_text(segments, TextType.SENTENCE).split())


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
