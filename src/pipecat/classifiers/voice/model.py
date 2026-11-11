#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Voice Reader's model, on the CPU."""

import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from loguru import logger

from pipecat.audio.turn.smart_turn._whisper_features import compute_whisper_log_mel_features

try:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download, snapshot_download
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Pipecat Voice Reader, you need to `uv add "pipecat-ai[voice-reader]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

VOICE_REPO = "pipecat-ai/voice-reader"
VOICE_REVISION = "rc6"

#: The audio the model hears: 16-bit mono PCM at this rate, and the last of it
#: up to this long.
VOICE_SAMPLE_RATE = 16000
VOICE_WINDOW_SECS = 8.0

_MODEL_FILES = {"int8": "model.int8.onnx", "fp32": "model.onnx"}

#: Samples of audio in one frame of the model's features, and frames a second.
_HOP = 160
_FRAMES_PER_SEC = VOICE_SAMPLE_RATE // _HOP


class PipecatVoiceReaderModel:
    """Pipecat Voice Reader's model, on the CPU.

    The model hears a clip of speech and gives one value for each of its
    outputs, which its ``config.json`` names. Each value leaves the model
    final, with its calibration applied.
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
            model_dir: A folder with the model's ``config.json`` and ONNX file.
            precision: ``"int8"`` for the quantized weights, which read the
                same at a quarter of the size, or ``"fp32"``.
            threads: The CPU threads for one read.

        Raises:
            ValueError: If the folder holds a model this class doesn't read.
        """
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if not isinstance(self.config["outputs"], dict):
            raise ValueError(f"{model_dir}: a release before rc4, whose values need calibrating")
        window_secs = float(self.config.get("window_secs", VOICE_WINDOW_SECS))
        if window_secs != VOICE_WINDOW_SECS:
            raise ValueError(
                f"{model_dir}: the model reads {window_secs:g}s windows, "
                f"and its features are computed over {VOICE_WINDOW_SECS:g}s"
            )
        self._outputs = list(self.config["outputs"])
        self._tones = list(self.config["tones"])
        self._sarcasm_threshold = float(self.config["outputs"]["sarcasm"].get("threshold", 0.5))
        # The lengths the model reads, in frames of its features.
        read_secs = self.config.get("read_secs") or [VOICE_WINDOW_SECS]
        self._read_frames = sorted(round(float(secs) * _FRAMES_PER_SEC) for secs in read_secs)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_dir / _MODEL_FILES[precision]), options, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def download(
        cls,
        *,
        revision: str = VOICE_REVISION,
        variant: Literal["base", "tiny"] = "base",
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
    ) -> "PipecatVoiceReaderModel":
        """Load the model from Hugging Face, downloading its files the first time.

        Args:
            revision: The model's revision.
            variant: The model's size, ``"base"`` or ``"tiny"``.
            precision: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.

        Returns:
            The model.

        Raises:
            ValueError: If the release has no such variant.
        """
        # The release's index names each variant's folder. Hugging Face counts
        # a download for each request of this file, so every load asks for it.
        index = json.loads(
            Path(hf_hub_download(VOICE_REPO, "config.json", revision=revision)).read_text(
                encoding="utf-8"
            )
        )
        variants = index["variants"]
        if variant not in variants:
            raise ValueError(
                f"{VOICE_REPO}@{revision} has no {variant!r} variant: one of {', '.join(variants)}"
            )
        folder = variants[variant]["folder"]
        files = ("config.json", _MODEL_FILES[precision])
        root = snapshot_download(
            VOICE_REPO,
            revision=revision,
            allow_patterns=[f"{folder}/{name}" for name in files],
        )
        return cls(Path(root) / folder, precision=precision, threads=threads)

    @property
    def tones(self) -> list[str]:
        """The tones of voice the model tells apart."""
        return list(self._tones)

    @property
    def outputs(self) -> list[str]:
        """The names of the values the model gives for a clip."""
        return list(self._outputs)

    @property
    def sarcasm_threshold(self) -> float:
        """The ``sarcasm`` value from which the model's release counts a turn as sarcastic."""
        return self._sarcasm_threshold

    def read(self, audio: bytes) -> dict[str, Any]:
        """Read one clip of speech.

        Args:
            audio: The clip, as 16-bit mono PCM at 16 kHz. Only its last 8
                seconds are read.

        Returns:
            One value per output: ``tone`` as the probability of each tone,
            and every other output as a number.
        """
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        window = np.zeros(int(VOICE_WINDOW_SECS * VOICE_SAMPLE_RATE), dtype=np.float32)
        samples = samples[-len(window) :]
        if len(samples):
            # The model was trained on speech followed by silence, so the clip
            # goes first. It goes without its own mean, so a microphone's DC
            # offset doesn't lift the silence after it off zero.
            window[: len(samples)] = samples - samples.mean()
        features = compute_whisper_log_mel_features(window, do_normalize=True)
        # The model reads the shortest of its lengths that holds the clip, so a
        # short clip costs a short read.
        clip_frames = max(-(-len(samples) // _HOP), 1)
        frames = next((n for n in self._read_frames if clip_frames <= n), self._read_frames[-1])
        features = np.ascontiguousarray(features[:, :frames])
        results = self._session.run(None, {"input_features": features[None]})
        values = {name: cast(np.ndarray, r)[0] for name, r in zip(self._outputs, results)}
        tone = dict(zip(self._tones, map(float, values.pop("tone"))))
        return {"tone": tone, **{name: float(value) for name, value in values.items()}}
