#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was meant with Pipecat Tone."""

import asyncio
from pathlib import Path
from typing import Literal

from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTurnFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.processors.tone._models import load_model
from pipecat.processors.tone.reader.model import TONE_REVISION, PipecatToneReaderModel

_EMOTION = (
    "The feeling the words show. Read from the words alone, so neutral means they show no "
    "strong feeling, not that the user feels none: go by what they say."
)
_SARCASM = (
    "The chance the turn is sarcastic, from 0 to 1. Read the emotion together with it: "
    "a sarcastic turn can read happy."
)


class PipecatToneReader(FrameProcessor):
    """Reads how each user turn was meant with Pipecat Tone.

    Goes between the STT service and the user aggregator. Each final
    transcription is read, with the bot's reply before it, and the reading is
    set as its ``perception``: the turn's ``emotion``, its two likeliest labels
    with their confidences, and its ``sarcasm``, the chance it's sarcastic.
    The read runs off the event loop, and the transcription waits for it, a
    few milliseconds.

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
        super().__init__(**kwargs)
        self._model: PipecatToneReaderModel | None = None
        self._revision = revision
        self._model_dir = Path(model_dir) if model_dir else None
        self._precision: Literal["fp16", "fp32"] = precision
        self._threads = threads
        self._previous: str | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and load its model.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._model = await load_model(
            self,
            ("reader", self._model_dir or self._revision, self._precision, self._threads),
            self._open_model,
        )

    def _open_model(self) -> PipecatToneReaderModel:
        if self._model_dir:
            return PipecatToneReaderModel(
                self._model_dir, precision=self._precision, threads=self._threads
            )
        return PipecatToneReaderModel.download(
            revision=self._revision, precision=self._precision, threads=self._threads
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Read final transcriptions, and keep the bot's replies to read them against.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._model:
                await self.push_frame(
                    PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
                )
            return
        if isinstance(frame, TranscriptionFrame) and frame.text.strip() and self._model:
            frame.perception = await self._read(frame.text)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._previous = frame.text
        await self.push_frame(frame, direction)

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
            await self.push_error(f"{self}: couldn't read the turn: {e}", e)
            return None
        likeliest = sorted(tags["emotion"]["probabilities"].items(), key=lambda p: -p[1])[:2]
        return [
            PerceptionSignal("emotion", labels=likeliest),
            PerceptionSignal("sarcasm", value=round(tags["sarcasm"]["probability"], 2)),
        ]
