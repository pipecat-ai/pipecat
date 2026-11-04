#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was meant with Pipecat Tone."""

from pathlib import Path
from typing import Literal

from pipecat.classifiers.tone.classifier import PipecatToneClassifier
from pipecat.classifiers.tone.model import TONE_REVISION
from pipecat.expressive.processors.perception_reader import PerceptionQuestion, PerceptionReader

_EMOTION = (
    "The feeling the words show. Read from the words alone, so neutral means they show no "
    "strong feeling, not that the user feels none: go by what they say."
)
_SARCASM = (
    "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: "
    "a sarcastic turn can read happy."
)


class PipecatToneReader(PerceptionReader):
    """Reads how each user turn was meant with Pipecat Tone.

    Goes between the STT service and the user aggregator. Each final
    transcription is read, with the bot's reply before it, and the reading is
    set as its ``perception``: the turn's ``emotion``, its two likeliest labels
    with their confidences, and its ``sarcasm``, the chance it's sarcastic.
    The read runs off the event loop, and the transcription waits for it, a
    few milliseconds. It's a :class:`PerceptionReader` that asks a
    :class:`PipecatToneClassifier` its emotion and sarcasm questions.

    The model loads while the pipeline is set up, once a process, and every
    reader shares it. The first load downloads it from Hugging Face, which holds
    the pipeline's start, so download it ahead, such as when building the
    image. A model that can't be loaded is reported as an error that leaves the
    reader unusable, and transcriptions pass through unread.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        pipeline = Pipeline([transport.input(), stt, PipecatToneReader(), user_aggregator, llm, ...])
    """

    def __init__(
        self,
        *,
        revision: str = TONE_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["fp16", "fp32"] = "fp16",
        threads: int = 4,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            revision: The model's release on Hugging Face, such as ``"reader-rc2"``.
            model_dir: A folder to load the model from instead of downloading one,
                laid out as a release's ``reader``. ``revision`` is then unused.
            precision: The weights to load: ``"fp16"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            **kwargs: Additional arguments passed to the parent.
        """
        tone = PipecatToneClassifier(
            revision=revision, model_dir=model_dir, precision=precision, threads=threads
        )
        # Tone's questions come from its model once it's loaded: see `questions`.
        super().__init__(classifier=tone, questions={}, **kwargs)
        self._tone = tone

    @property
    def questions(self) -> dict[str, PerceptionQuestion]:
        """Tone's emotion and sarcasm questions, with the loaded model's options.

        Raises:
            ClassifierError: If the model isn't loaded yet.
        """
        tone = self._tone.questions
        return {
            "emotion": PerceptionQuestion(question=tone["emotion"], description=_EMOTION),
            "sarcasm": PerceptionQuestion(question=tone["sarcasm"], description=_SARCASM),
        }
