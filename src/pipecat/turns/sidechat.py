#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that lets a bot in a group conversation answer only the lines meant for it.

**Experimental.** The model, its outputs and these types are likely to change
between releases.
"""

import asyncio
import re
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from loguru import logger

from pipecat.classifiers.base_classifier import BaseClassifier, ClassifierError, YesNoQuestion
from pipecat.classifiers.sidechat.classifier import DEFAULT_ROLE, PipecatSidechatClassifier
from pipecat.classifiers.sidechat.conversation import (
    MAX_HISTORY,
    SidechatLine,
    SidechatTurn,
    mentions,
)
from pipecat.classifiers.sidechat.model import SIDECHAT_REVISION, SidechatDecision, decide
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
    LLMContextUserTurnFrame,
    MetricsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup

#: The STT ``speaker_format`` whose tags the processor reads by default.
SIDECHAT_SPEAKER_FORMAT = "[Speaker {speaker_number}] {text}"

#: Why the processor decided as it did.
SidechatReason = Literal["model", "silence", "floor_taken", "speculation", "error"]

_SPEAKER_FIELD = re.compile(r"\{(speaker|speaker_number)\}")


@dataclass(frozen=True)
class SidechatVerdict:
    """What the processor did with a user turn.

    Parameters:
        line: The line the decision came from, with the turns the model read
            before it. A turn in which several people spoke has a line for
            each, and this is the one for the bot, or else the last.
        decision: ``"respond"`` when the LLM answers the turn, ``"hold"``
            while the processor waits to see whether anyone else speaks, and
            ``"ignore"``.
        reason: ``"model"`` for the model's decision, ``"silence"`` for a held
            turn answered once nobody spoke, ``"floor_taken"`` for a held turn
            let go because someone did, ``"speculation"`` for a held turn
            answered from an eager end of turn, which is let go, and
            ``"error"`` when the model gave no answer and the processor went
            by whether the line names the bot.
        respond: The model's probability that the bot should speak now, or
            None without an answer.
        open: The model's probability that the line is a question or request
            open to anyone, or None without an answer.
    """

    line: SidechatLine
    decision: SidechatDecision
    reason: SidechatReason
    respond: float | None = None
    open: float | None = None


class SidechatSpeakers:
    """Reads who said what in a transcript an STT service tagged with its ``speaker_format``.

    A diarizing STT service writes each speaker's words into the transcript
    with ``speaker_format``, so a user turn in which two people spoke reads
    ``"[Speaker 1] Shall we ask? [Speaker 2] Go on."``. This splits such a
    text back into each speaker's words, without the tags.
    """

    def __init__(self, speaker_format: str = SIDECHAT_SPEAKER_FORMAT):
        """Initialize the reader.

        Args:
            speaker_format: The STT service's ``speaker_format``, with
                ``{speaker}`` or ``{speaker_number}`` before ``{text}``.

        Raises:
            ValueError: If the format doesn't name the speaker before the text.
        """
        before, text, after = speaker_format.partition("{text}")
        if not text or not _SPEAKER_FIELD.search(before):
            raise ValueError(
                f"speaker_format {speaker_format!r} needs {{speaker}} or {{speaker_number}} "
                "before {text}"
            )
        self._tag = re.compile(self._pattern(before.rstrip(), named=True) + r"\s*")
        # What the format writes after the text, such as a closing tag.
        closing = self._pattern(after.strip(), named=False) if after.strip() else None
        self._closing = re.compile(closing) if closing else None
        self._closing_at_end = re.compile(closing + "$") if closing else None

    def split(self, text: str) -> list[tuple[str | None, str]]:
        """Split a transcript into each speaker's words, in the order they were said.

        Args:
            text: The transcript, as the STT service tagged it.

        Returns:
            Each run of one speaker's words with the speaker's label, such as
            ``"Speaker 1"``. Words before the first tag, or all of an untagged
            transcript, have no speaker.
        """
        tags = [
            tag
            for tag in self._tag.finditer(text)
            if not (self._closing and self._closing.match(text, tag.start()))
        ]
        runs: list[tuple[str | None, str]] = []
        lead = text[: tags[0].start()] if tags else text
        if lead.strip():
            runs.append((None, lead.strip()))
        for tag, following in zip(tags, [*tags[1:], None]):
            words = text[tag.end() : following.start() if following else len(text)].strip()
            if self._closing_at_end:
                words = self._closing_at_end.sub("", words).strip()
            if not words:
                continue
            fields = tag.groupdict()
            speaker = f"Speaker {fields.get('speaker_number') or fields.get('speaker')}"
            if runs and runs[-1][0] == speaker:
                runs[-1] = (speaker, f"{runs[-1][1]} {words}")
            else:
                runs.append((speaker, words))
        return runs

    @staticmethod
    def _pattern(template: str, *, named: bool) -> str:
        """A pattern matching what ``template`` writes, capturing its first speaker fields."""
        pattern, position, captured = "", 0, set()
        for field in _SPEAKER_FIELD.finditer(template):
            pattern += re.escape(template[position : field.start()])
            name = field.group(1)
            value = r"\d+" if name == "speaker_number" else r"\S+?"
            if named and name not in captured:
                captured.add(name)
                pattern += f"(?P<{name}>{value})"
            else:
                pattern += f"(?:{value})"
            position = field.end()
        return pattern + re.escape(template[position:])


class PipecatSidechat(FrameProcessor):
    """Lets the LLM answer only the lines of a group conversation meant for the bot.

    Goes between the user aggregator and the LLM. The bot hears every line:
    the user aggregator writes each user turn to the context. The processor
    holds the context frame that would run the LLM and asks Pipecat Sidechat
    whether the turn was for the bot:

    - ``respond``: the LLM answers it.
    - ``hold``: a line the bot could take or leave, such as a question to the
      room. If nobody speaks for ``hold_secs`` after it, the LLM answers it;
      if someone does, it is let go.
    - ``ignore``: the LLM doesn't run, and the bot stays quiet.

    The model reads each line with up to 8 turns before it, the bot's own
    included, as the aggregators write them to the context. It also reads who
    said each one, from the tags a diarizing STT service writes into the
    transcript with its ``speaker_format``, the same tags that tell the LLM
    who said what. Give the STT service and the processor the same format.
    Without tags every speaker is unknown, and the model goes by the words and
    the bot's turns alone.

    A user turn in which several people spoke is read as a line for each. The
    LLM answers it if any of them is for the bot, and a line the bot would
    hold counts only when it is the last.

    A context frame that doesn't follow a user turn, such as a greeting or a
    function call result, passes through. The processor decides whether the
    bot answers a turn, not whether the turn interrupts the bot: that stays
    with the user turn start strategies. It is for a pipeline with an STT
    service and an LLM, not a speech-to-speech one.

    The model loads while the pipeline is set up, once a process, and every
    processor of the same variant shares it. The first load downloads it from
    Hugging Face, which holds the pipeline's start, so download it ahead, such
    as when building the image. A model that can't be loaded is reported as an
    error that leaves the processor unusable, and it then lets the LLM answer
    only a line that names the bot. It does the same for a line the model
    can't read within ``timeout``.

    Event handlers available:

    - on_sidechat_decision: Called with a :class:`SidechatVerdict` for each
      user turn the processor decides, and again when a held turn is answered
      or let go.

    Example::

        sidechat = PipecatSidechat(agent_name="Atlas")

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_aggregator,
                sidechat,
                llm,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        )

        @sidechat.event_handler("on_sidechat_decision")
        async def on_sidechat_decision(sidechat, verdict: SidechatVerdict):
            logger.info(f"{verdict.decision} ({verdict.reason}): {verdict.line.text}")
    """

    def __init__(
        self,
        *,
        agent_name: str,
        aliases: Sequence[str] = (),
        role: str | None = DEFAULT_ROLE,
        speaker_format: str | None = SIDECHAT_SPEAKER_FORMAT,
        variant: Literal["base", "small"] = "small",
        revision: str = SIDECHAT_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
        hold_secs: float = 2.5,
        timeout: float = 2.0,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            agent_name: The bot's name, as people say it.
            aliases: Other ways speech recognition writes the bot's name, such
                as ``"at last"`` for ``"Atlas"``.
            role: One line on what the bot is and does, such as ``"voice
                assistant in the meeting room's screen; it shows the agenda"``.
                The model reads it with every line. By default the role of a
                bot on a call.
            speaker_format: The STT service's ``speaker_format``, whose tags
                say who said each part of a turn, or None when it has none.
            variant: The model's size. ``"small"`` reads a line in a few
                milliseconds on half a CPU. ``"base"`` decides better, most of
                all on requests that don't name the bot, and takes a few
                hundred milliseconds a line on one CPU.
            revision: The model's release on Hugging Face, such as
                ``"rc0.3.1"``. By default the one this version of Pipecat was
                tested with. ``"main"`` follows the latest.
            model_dir: A folder to load the model from instead of downloading
                one, laid out as one of a release's variants. ``variant`` and
                ``revision`` are then unused.
            precision: The weights to load: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            hold_secs: Seconds of nobody speaking after a held turn before the
                LLM answers it.
            timeout: Seconds to wait for the model before going by whether
                the line names the bot.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._classifier = PipecatSidechatClassifier(
            agent_name=agent_name,
            aliases=aliases,
            role=role,
            variant=variant,
            revision=revision,
            model_dir=model_dir,
            precision=precision,
            threads=threads,
        )
        self._speakers = SidechatSpeakers(speaker_format) if speaker_format else None
        self._hold_secs = hold_secs
        self._timeout = timeout

        self._history: deque[SidechatTurn] = deque(maxlen=MAX_HISTORY)
        # The lines of the user turn whose context frame is on its way, and of
        # a speculation's.
        self._lines: list[SidechatLine] = []
        self._speculative_lines: list[SidechatLine] = []
        # Whether a speculation is in flight: the turn that confirms it is
        # written to the context with no context frame of its own.
        self._speculating = False
        self._held: tuple[SidechatVerdict, LLMContextFrame, asyncio.Task] | None = None
        self._deciding = asyncio.Lock()
        self._user_speaking = False
        self._speech_ended_at = time.monotonic()
        self._bot_speaking = False
        self._over_agent = False

        self._register_event_handler("on_sidechat_decision")
        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and load the model.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        try:
            await self._classifier.setup(self.task_manager)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e, force_treat_as_permanent=True)

    async def cleanup(self):
        """Let go of a held turn and clean up the classifier."""
        await super().cleanup()
        held, self._held = self._held, None
        if held:
            await self.cancel_task(held[2])
        await self._classifier.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold each user turn's context frame until the model decides.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextUserTurnFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._user_turn(frame)
        elif isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            lines = self._speculative_lines if frame.speculation else self._lines
            if lines:
                if frame.speculation:
                    self._speculative_lines = []
                else:
                    self._lines = []
                self.create_task(self._decide(lines, frame), f"{self}::_decide")
                return
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            if frame.text.strip():
                self._history.append(SidechatTurn(text=frame.text.strip(), agent=True))
                self._prepare()
        elif isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._user_speaking = True
            self._over_agent = self._over_agent or self._bot_speaking
            await self._floor_taken()
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            self._user_speaking = False
            self._speech_ended_at = time.monotonic()
        elif isinstance(frame, (UserTurnResponseCancelFrame, InterruptionFrame)):
            self._speculating = False
            self._speculative_lines = []
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        await self.push_frame(frame, direction)

    async def _user_turn(self, frame: LLMContextUserTurnFrame):
        """Make lines of the turn the user aggregator wrote, to judge when its context comes."""
        runs = self._speakers.split(frame.text) if self._speakers else [(None, frame.text.strip())]
        history = list(self._history)
        lines = []
        for speaker, text in runs:
            lines.append(
                SidechatLine(
                    text=text,
                    speaker=speaker,
                    history=tuple(history[-MAX_HISTORY:]),
                    # The turn began over the bot, so its first line did.
                    over_agent=True if self._over_agent and not lines else None,
                )
            )
            history.append(SidechatTurn(text=text, speaker=speaker))
        if frame.speculation:
            self._speculative_lines = lines
            self._speculating = True
            return
        await self._floor_taken()
        # A turn that confirms a speculation was decided when it was speculated.
        self._lines = [] if self._speculating else lines
        self._speculating = False
        self._history.extend(history[len(self._history) :])
        self._over_agent = False
        self._prepare()

    def _prepare(self):
        """Have the model read the turns so far ahead of the next line."""
        state = self._state(SidechatLine(text="", history=tuple(self._history)))
        self.create_task(self._prepare_task(state), f"{self}::_prepare")

    async def _prepare_task(self, state: dict):
        try:
            await self._classifier.prepare(state)
        except ClassifierError as e:
            logger.debug(f"{self}: turns not read ahead: {e}")

    async def _decide(self, lines: list[SidechatLine], frame: LLMContextFrame):
        async with self._deciding:
            verdict = await self._verdict(lines)
            if verdict.decision == "hold" and frame.speculation:
                verdict = replace(verdict, decision="ignore", reason="speculation")
            if verdict.decision == "respond":
                await self._report(verdict)
                await self.push_frame(frame)
            elif verdict.decision == "hold":
                await self._hold(verdict, frame)
            else:
                await self._report(verdict)

    async def _verdict(self, lines: list[SidechatLine]) -> SidechatVerdict:
        """The decision on a turn's lines: a line for the bot's, or else the last line's."""
        try:
            async with asyncio.timeout(self._timeout):
                # The last line is read first: a turn is most often one line,
                # and a line for the bot settles the turn.
                last = await self._read(lines[-1])
                if last.decision == "respond":
                    return last
                for line in reversed(lines[:-1]):
                    earlier = await self._read(line)
                    if earlier.decision == "respond":
                        return earlier
                return last
        except (ClassifierError, TimeoutError) as e:
            texts = " / ".join(line.text for line in lines)
            logger.warning(f"{self}: no answer for [{texts}], going by the bot's name: {e}")
            named = [line for line in lines if mentions(self._classifier.agent, line.text)]
            return SidechatVerdict(
                named[-1] if named else lines[-1], "respond" if named else "ignore", "error"
            )

    async def _read(self, line: SidechatLine) -> SidechatVerdict:
        """The model's decision on one line."""
        asked = self._classifier.questions
        questions = {
            name: question
            for name in ("respond", "open")
            if isinstance(question := asked[name], YesNoQuestion)
        }
        results = await self._classifier.yes_no(self._state(line), questions)
        respond, open_ = results["respond"].probability, results["open"].probability
        decision = decide(respond, open_, self._classifier.thresholds)
        return SidechatVerdict(line, decision, "model", respond, open_)

    def _state(self, line: SidechatLine) -> dict:
        """A line as the classifier's state."""
        return {
            "text": line.text,
            "speaker": line.speaker,
            "history": [
                {"text": turn.text, "speaker": turn.speaker, "agent": turn.agent}
                for turn in line.history
            ],
            "over_agent": line.over_agent,
        }

    async def _hold(self, verdict: SidechatVerdict, frame: LLMContextFrame):
        """Answer the turn if nobody speaks for a while after it."""
        if self._user_speaking:
            await self._report(replace(verdict, decision="ignore", reason="floor_taken"))
            return
        await self._report(verdict)
        # The silence started when the room went quiet, not when the model answered.
        quiet = time.monotonic() - self._speech_ended_at
        delay = min(max(self._hold_secs - quiet, 0.0), self._hold_secs)
        task = self.create_task(self._answer_after(delay), f"{self}::_answer_after")
        self._held = (verdict, frame, task)

    async def _answer_after(self, delay: float):
        await asyncio.sleep(delay)
        held, self._held = self._held, None
        if held:
            verdict, frame, _ = held
            await self._report(replace(verdict, decision="respond", reason="silence"))
            await self.push_frame(frame)

    async def _floor_taken(self):
        """Let a held turn go: someone spoke after it."""
        held, self._held = self._held, None
        if held:
            verdict, _, task = held
            await self.cancel_task(task)
            await self._report(replace(verdict, decision="ignore", reason="floor_taken"))

    async def _report(self, verdict: SidechatVerdict):
        logger.debug(
            f"{self}: {verdict.decision} ({verdict.reason}) "
            f"<- {verdict.line.speaker or 'unknown'}: {verdict.line.text}"
        )
        await self._call_event_handler("on_sidechat_decision", verdict)

    async def _on_classifier_metrics(self, classifier: BaseClassifier, data: list[MetricsData]):
        await self.push_frame(MetricsFrame(data=data))
