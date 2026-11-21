#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifier backed by Pipecat Sidechat, a local model of who a line in a group conversation was for.

Pipecat Sidechat (https://huggingface.co/pipecat-ai/sidechat) reads one line
of a conversation between several people and an agent, with the turns before
it, and says whether the line was for the agent and whether the agent should
speak. It runs on the CPU.

**Experimental.** The model, its outputs and these types are likely to change
between releases.
"""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierInput,
    ClassifierQuestion,
    ClassifierResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.sidechat.conversation import SidechatAgent, SidechatLine, SidechatTurn
from pipecat.classifiers.sidechat.model import (
    INTENTS,
    SIDECHAT_REPO,
    SIDECHAT_REVISION,
    PipecatSidechatModel,
    SidechatAnswer,
    SidechatThresholds,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.models import load_shared_model

#: The role the model was trained with when an application gives none.
DEFAULT_ROLE = "AI assistant on the call"

#: The model's questions, each named after the output that answers it.
_QUESTIONS: dict[str, ClassifierQuestion] = {
    "respond": YesNoQuestion(instructions="Should the agent speak now?"),
    "addressed": YesNoQuestion(
        instructions="Was the line for the agent: said to it, or asking it to act?"
    ),
    "open": YesNoQuestion(
        instructions=(
            "Is the line a question or request to nobody in particular, which the agent "
            "could take if nobody else does?"
        )
    ),
    "intent": ChoiceQuestion(
        instructions="Why was the line said?",
        options={
            "direct": "said to the agent",
            "indirect": "wants the agent to act without speaking to it",
            "open": "a question or request to nobody in particular",
            "about": "mentions the agent and wants nothing from it",
            "other": "for a person, the people, another device or nobody",
        },
    ),
    "unclear": YesNoQuestion(instructions="Does the line fit the agent and someone else equally?"),
}


class PipecatSidechatClassifier(BaseClassifier):
    """Answers Sidechat's own questions about a line of a group conversation.

    It answers only the questions built into the model, the ones in
    :attr:`questions`: whether the agent should speak now (``respond``),
    whether the line was for the agent (``addressed``), whether it is a
    question or request open to anyone (``open``), why it was said
    (``intent``) and whether it fits the agent and someone else equally
    (``unclear``). Any other question raises a
    :class:`~pipecat.classifiers.base_classifier.ClassifierError`. A question
    is known by what it asks, so it can be asked under any name. One read of
    the line answers every question asked.

    ``respond`` is calibrated: a line the agent could take or leave, such as
    a question to the room that nothing earlier settles, reads about 0.5. The
    release's :attr:`thresholds` say where it chose to speak now and where to
    speak after a pause.

    The state is the line with the turns before it, oldest first. Speakers
    can be names or diarization labels such as ``"Speaker 1"``; the model
    only compares them, and reads the agent's own name as the sign that a
    line is for it::

        {
            "text": "And by train?",
            "speaker": "Speaker 1",
            "history": [
                {"speaker": "Speaker 1", "text": "Atlas, how far is Leeds?"},
                {"text": "About forty miles.", "agent": True},
            ],
        }

    ``speaker``, ``history``, ``people`` (the names of the people present)
    and ``over_agent`` (true when the line was said while the agent spoke)
    can be left out. The model reads the last 8 turns of ``history``.

    The model loads in :meth:`setup`, once a process, and every classifier
    of the same variant shares it. The first load downloads it from Hugging
    Face.

    The ``small`` variant reads the turns before a line apart from the line.
    :meth:`prepare` encodes them ahead, while the line is still being said, so
    the questions about it are answered in a few milliseconds.

    Example::

        classifier = PipecatSidechatClassifier(agent_name="Atlas")
        await classifier.setup(task_manager)

        results = await classifier.ask(state, classifier.questions)
        respond = results["respond"].probability
    """

    def __init__(
        self,
        *,
        agent_name: str,
        aliases: Sequence[str] = (),
        role: str | None = DEFAULT_ROLE,
        variant: Literal["base", "small"] = "small",
        revision: str = SIDECHAT_REVISION,
        model_dir: str | Path | None = None,
        precision: Literal["int8", "fp32"] = "int8",
        threads: int = 2,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            agent_name: The agent's name, as people say it.
            aliases: Other ways speech recognition writes the agent's name,
                such as ``"at last"`` for ``"Atlas"``.
            role: One line on what the agent is.
            variant: The model's size: ``"base"`` decides best, and
                ``"small"`` reads a line in a few milliseconds on half a CPU.
            revision: The model's release on Hugging Face, such as
                ``"rc0.3.1"``. By default the one this version of Pipecat was
                tested with. ``"main"`` follows the latest.
            model_dir: A folder to load the model from instead of downloading
                one, laid out as one of a release's variants. ``variant`` and
                ``revision`` are then unused.
            precision: The weights to load: ``"int8"`` or ``"fp32"``.
            threads: The CPU threads for one read.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._agent = SidechatAgent(name=agent_name, aliases=tuple(aliases), role=role)
        self._variant: Literal["base", "small"] = variant
        self._revision = revision
        self._model_dir = Path(model_dir) if model_dir else None
        self._precision: Literal["int8", "fp32"] = precision
        self._threads = threads
        self._model: PipecatSidechatModel | None = None

    @property
    def model(self) -> str:
        """The model's folder, or its release and variant on Hugging Face."""
        if self._model_dir:
            return str(self._model_dir)
        return f"{SIDECHAT_REPO}@{self._revision}/{self._variant}"

    @property
    def agent(self) -> SidechatAgent:
        """The agent the classifier judges lines for."""
        return self._agent

    @property
    def questions(self) -> dict[str, ClassifierQuestion]:
        """The questions the model answers, by name."""
        return dict(_QUESTIONS)

    @property
    def thresholds(self) -> SidechatThresholds:
        """Where the model's release chose to speak now, and to speak after a pause.

        Raises:
            ClassifierError: If the model isn't loaded yet.
        """
        if not self._model:
            raise ClassifierError(f"{self}: the thresholds come from the model, loaded in setup()")
        return self._model.thresholds

    async def setup(self, task_manager: BaseTaskManager):
        """Load the model, or share the one already loaded.

        Args:
            task_manager: The task manager of the owner.

        Raises:
            ClassifierError: If the model can't be loaded.
        """
        await super().setup(task_manager)
        key = (
            "sidechat",
            self._model_dir or (self._revision, self._variant),
            self._precision,
            self._threads,
        )
        try:
            self._model = await load_shared_model(key, self._open_model)
        except Exception as e:
            raise ClassifierError(f"couldn't load Pipecat Sidechat: {e!r}") from e

    async def prepare(self, state: dict[str, Any]):
        """Encode the turns before a line ahead of the line itself.

        Saves the ``small`` variant's next :meth:`ask` the work of reading
        the turns; with the ``base`` variant it does nothing.

        Args:
            state: The state of the line to come. Only its ``history`` is
                read, so its ``text`` can be left out.

        Raises:
            ClassifierError: If the model isn't loaded or can't read the turns.
        """
        if not self._model:
            raise ClassifierError(f"{self}: the model isn't loaded, see setup()")
        if not self._model.prepares:
            return
        line = self._line({**state, "text": state.get("text") or ""})
        try:
            await asyncio.to_thread(self._model.prepare, self._agent, line)
        except Exception as e:
            raise ClassifierError(f"couldn't read the turns: {e!r}") from e

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        inputs: Sequence[ClassifierInput] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions with one read of the line."""
        if not self._model:
            raise ClassifierError(f"{self}: the model isn't loaded, see setup()")
        heads = {name: self._head(question) for name, question in questions.items()}
        line = self._line(state)
        try:
            answer = await asyncio.to_thread(self._model.answer, self._agent, line)
        except Exception as e:
            raise ClassifierError(f"couldn't read the line: {e!r}") from e
        return {name: self._result(head, answer) for name, head in heads.items()}, None

    def _open_model(self) -> PipecatSidechatModel:
        """The model from its folder, or downloaded from Hugging Face."""
        if self._model_dir:
            return PipecatSidechatModel(
                self._model_dir, precision=self._precision, threads=self._threads
            )
        return PipecatSidechatModel.download(
            revision=self._revision,
            variant=self._variant,
            precision=self._precision,
            threads=self._threads,
        )

    def _head(self, question: ClassifierQuestion) -> str:
        """Which of the model's outputs answers the question, or a ClassifierError if none."""
        head = next((name for name, own in _QUESTIONS.items() if own == question), None)
        if head is None:
            raise ClassifierError(
                f"{self} answers only its own questions, from PipecatSidechatClassifier.questions"
            )
        return head

    def _line(self, state: str | dict[str, Any] | list[Any]) -> SidechatLine:
        """The line a state describes, or a ClassifierError if it describes none."""
        if not isinstance(state, Mapping) or not isinstance(state.get("text"), str):
            raise ClassifierError(f"{self} reads a state with the line's 'text'")
        try:
            history = tuple(
                SidechatTurn(
                    text=str(turn["text"]),
                    speaker=_label(turn.get("speaker")),
                    agent=bool(turn.get("agent", False)),
                )
                for turn in state.get("history") or []
            )
        except (KeyError, TypeError, AttributeError) as e:
            raise ClassifierError(f"{self}: each turn of 'history' needs its 'text'") from e
        people = state.get("people")
        return SidechatLine(
            text=state["text"],
            speaker=_label(state.get("speaker")),
            history=history,
            people=tuple(str(name) for name in people) if people else None,
            over_agent=True if state.get("over_agent") else None,
        )

    def _result(self, head: str, answer: SidechatAnswer) -> ClassifierResult:
        """The answer to one of the model's questions, from its reading of the line."""
        if head == "intent":
            choice = max(INTENTS, key=lambda name: answer.intent[name])
            return ChoiceResult(
                choice=choice, probabilities=dict(answer.intent), confidence=answer.intent[choice]
            )
        probability = {
            "respond": answer.respond,
            "addressed": answer.addressed,
            "open": answer.open,
            "unclear": answer.unclear,
        }[head]
        return YesNoResult(probability=min(max(probability, 0.0), 1.0))


def _label(speaker: Any) -> str | None:
    return None if speaker is None else str(speaker)
