#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that hears how each user turn sounded with Pipecat Voice Reader, and its source."""

from pathlib import Path
from typing import Literal

from pipecat.classifiers.voice.classifier import PipecatVoiceClassifier
from pipecat.classifiers.voice.model import VOICE_REVISION
from pipecat.expressive.processors.perception_reader import (
    PerceptionQuestion,
    PerceptionReader,
    PerceptionSource,
)

_TONE = (
    "How the user's voice sounds, from the audio alone, not the words: positive (warm, bright, "
    "pleased), neutral, sad (low, deflated), frustrated (tense, irritated, clipped). Read it "
    "with the emotion the words show."
)
_AROUSAL = "How activated the voice sounds, from 0 (flat, low energy) to 1 (highly animated)."
_LAUGHTER = "The chance the turn has laughter in it, from 0 to 1."
_SPEECH_RATE = (
    "Speaking rate in syllables per second; relaxed conversation is about 4, slow deliberate "
    "speech under 3."
)
_SARCASM_VOICE = (
    "The chance the delivery alone sounds sarcastic, from 0 to 1: flat or drawn-out while the "
    "words are bright, or a mock-cheerful lift. Weak on its own: weigh it with sarcasm from the "
    "words, and act when both point the same way or the words are plainly positive while the "
    "voice is flat or frustrated."
)
_SAD = (
    "How many listeners would hear sadness in the delivery, from 0 to 1, from the audio alone: "
    "0.5 means about half of them. Most turns read near 0. A graded cue for one turn, not a "
    "diagnosis: quiet speech can raise it, so read it with the words and the turns before."
)


class PipecatVoiceSource(PerceptionSource):
    """Pipecat Voice Reader as one of a :class:`PerceptionReader`'s sources.

    Hears the user's speech each time they stop and reports how it sounded:
    their ``tone`` of voice, its two likeliest labels with their confidences,
    the voice's ``arousal``, the chance of ``laughter``, their ``speech_rate``
    in syllables per second and ``sarcasm_voice``, the chance the delivery
    sounds sarcastic. The model's sarcasm is reported as ``sarcasm_voice`` so
    it can go beside a source that reads the words, such as
    :class:`~pipecat.expressive.processors.tone_reader.PipecatToneSource`,
    which reports a ``sarcasm`` of its own. Another source's signal needs a
    name other than these.

    The reader needs the user's audio and the VAD's frames, so the pipeline
    needs a VAD.

    Example::

        reader = PerceptionReader(sources=[PipecatToneSource(), PipecatVoiceSource()])
    """

    def __init__(
        self,
        *,
        variant: Literal["base", "tiny"] = "base",
        revision: str = VOICE_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
        sad: bool = False,
        timeout: float | None = None,
    ):
        """Initialize the source.

        Args:
            variant: The model's size: ``"base"`` reads tone and sadness best,
                and ``"tiny"`` costs about half the CPU.
            revision: The model's release on Hugging Face, such as ``"rc6"``.
            model_dir: A folder to load the model from instead of downloading one,
                laid out as one of a release's variants. ``variant`` and
                ``revision`` are then unused.
            precision: The weights to load: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            sad: Whether to also report ``sad``, the chance a listener would
                hear sadness in the turn.
            timeout: Seconds a transcription waits for its reading after the
                user stops speaking, 2 by default.
        """
        voice = PipecatVoiceClassifier(
            variant=variant,
            revision=revision,
            model_dir=model_dir,
            precision=precision,
            threads=threads,
        )
        asked = voice.questions
        questions = {
            "tone": PerceptionQuestion(question=asked["tone"], description=_TONE),
            "arousal": PerceptionQuestion(question=asked["arousal"], description=_AROUSAL),
            "laughter": PerceptionQuestion(question=asked["laughter"], description=_LAUGHTER),
            "speech_rate": PerceptionQuestion(
                question=asked["speech_rate"], description=_SPEECH_RATE
            ),
            "sarcasm_voice": PerceptionQuestion(
                question=asked["sarcasm"], description=_SARCASM_VOICE
            ),
        }
        if sad:
            questions["sad"] = PerceptionQuestion(question=asked["sad"], description=_SAD)
        super().__init__(classifier=voice, questions=questions, timeout=timeout)


class PipecatVoiceReader(PerceptionReader):
    """Hears how each user turn sounded with Pipecat Voice Reader.

    Goes between the STT service and the user aggregator, in a pipeline with a
    VAD. Each time the user stops speaking, their speech is heard, up to its
    last 8 seconds, and the reading goes in the ``perception`` of its
    transcriptions: the user's ``tone`` of voice, its two likeliest labels
    with their confidences, the voice's ``arousal``, the chance of
    ``laughter``, their ``speech_rate`` and ``sarcasm_voice``, the chance the
    delivery sounds sarcastic. The read runs off the event loop, and the
    transcriptions wait for it, about a tenth of a second. Speech shorter than
    0.8 seconds isn't read. It's a :class:`PerceptionReader` whose one source
    is a :class:`PipecatVoiceSource`.

    The model loads while the pipeline is set up, once a process, and every
    reader shares it. The first load downloads it from Hugging Face, which holds
    the pipeline's start, so download it ahead, such as when building the
    image. A model that can't be loaded is reported as an error that leaves the
    reader unusable, and transcriptions pass through unread.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        pipeline = Pipeline([transport.input(), stt, PipecatVoiceReader(), user_aggregator, llm, ...])
    """

    def __init__(
        self,
        *,
        variant: Literal["base", "tiny"] = "base",
        revision: str = VOICE_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
        sad: bool = False,
        timeout: float | None = None,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            variant: The model's size: ``"base"`` reads tone and sadness best,
                and ``"tiny"`` costs about half the CPU.
            revision: The model's release on Hugging Face, such as ``"rc6"``.
            model_dir: A folder to load the model from instead of downloading one,
                laid out as one of a release's variants. ``variant`` and
                ``revision`` are then unused.
            precision: The weights to load: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            sad: Whether to also report ``sad``, the chance a listener would
                hear sadness in the turn.
            timeout: Seconds a transcription waits for its reading after the
                user stops speaking, 2 by default.
            **kwargs: Additional arguments passed to the parent.
        """
        voice = PipecatVoiceSource(
            variant=variant,
            revision=revision,
            model_dir=model_dir,
            precision=precision,
            threads=threads,
            sad=sad,
            timeout=timeout,
        )
        super().__init__(sources=[voice], **kwargs)
