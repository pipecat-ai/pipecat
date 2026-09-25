#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The ``user:`` and ``judge:`` blocks every scenario file carries, and their summary.

``user:`` decides how the user's turns reach the bot (text, or speech synthesized
by a TTS) and ``judge:`` how the bot's reply is judged (its LLM text, or a
transcription of its speech, by which judge LLM). :func:`describe_config`
renders both as the two-line summary printed before a run.
"""

from pathlib import Path
from typing import Any, Protocol

from pipecat.evals.services import (
    DEFAULT_OLLAMA_JUDGE_EXTRA,
    DEFAULT_OLLAMA_JUDGE_MODEL,
    DEFAULT_OPENAI_MODEL,
)

_DEFAULT_JUDGE = {
    "service": "ollama",
    "model": DEFAULT_OLLAMA_JUDGE_MODEL,
    "extra": dict(DEFAULT_OLLAMA_JUDGE_EXTRA),
}


def _parse_user_block(user: Any, path: Path) -> tuple[bool, dict | None]:
    """Parse the ``user:`` block into ``(user_audio, speech config)``."""
    if user is None:
        return False, None  # default: text modality
    if not isinstance(user, dict):
        raise ValueError(f"{path}: 'user:' must be a mapping")
    modality = user.get("modality", "text")
    if modality not in ("audio", "text"):
        raise ValueError(f"{path}: 'user.modality:' must be 'audio' or 'text', got {modality!r}")
    if modality == "text":
        return False, None
    speech = user.get("speech")
    if speech is not None and not isinstance(speech, dict):
        raise ValueError(f"{path}: 'user.speech:' must be a mapping")
    return True, speech


def _parse_judge_block(judge: Any, path: Path) -> tuple[bool, dict | None, dict]:
    """Parse the ``judge:`` block into (bot_audio, transcriber, eval-config)."""
    if judge is None:
        judge = {}
    if not isinstance(judge, dict):
        raise ValueError(f"{path}: 'judge:' must be a mapping")
    modality = judge.get("modality", "text")
    if modality not in ("audio", "text"):
        raise ValueError(f"{path}: 'judge.modality:' must be 'audio' or 'text', got {modality!r}")
    eval_cfg = judge.get("eval") or dict(_DEFAULT_JUDGE)
    if not isinstance(eval_cfg, dict):
        raise ValueError(f"{path}: 'judge.eval:' must be a mapping (the judge LLM service)")
    if modality == "text":
        return False, None, eval_cfg
    transcription = judge.get("transcription")
    if not isinstance(transcription, dict):
        raise ValueError(
            f"{path}: 'judge.modality: audio' requires a 'judge.transcription:' block "
            "(STT service to transcribe the bot's audio)"
        )
    return True, transcription, eval_cfg


# ANSI codes for the colored config summary (applied only when color=True). The
# section labels are bold, the separators dim, and each segment's keyword gets one
# hue per category so modality / service / judge LLM are easy to tell apart; the
# values are left uncolored.
_CFG_LABEL = "1"  # bold — the "user" / "judge" section labels
_CFG_SEP = "2"  # dim — the "->" arrow and "|" separators
_CFG_MODALITY = "33"  # yellow — modality keyword
_CFG_SERVICE = "32"  # green — speech (TTS) / transcription (STT) keywords
_CFG_EVAL = "35"  # magenta — LLM keywords: the judge's eval, a simulation's persona
_CFG_LIMIT = "36"  # cyan — a simulation's run caps

# A config line's segment: keyword, value, and the ANSI code the keyword is painted with.
_ConfigSegment = tuple[str, str, str]
# A labeled config line: its label, then its segments or plain text.
_ConfigLine = tuple[str, list[_ConfigSegment] | str]


class EvalConfigured(Protocol):
    """The modality and service config a scenario and a simulation share."""

    user_audio: bool
    user_speech: dict | None
    bot_audio: bool
    transcriber: dict | None
    judge: dict


# The model each LLM service builds with when the block names none.
_DEFAULT_LLM_MODELS = {"ollama": DEFAULT_OLLAMA_JUDGE_MODEL, "openai": DEFAULT_OPENAI_MODEL}


def _svc_model(
    cfg: dict, default_service: str, model_key: str, default_models: dict[str, str] | None = None
) -> str:
    """What a service config block builds, as the run summary names it.

    ``factory:<path>`` for a factory, else ``service/model``, the model being
    the block's or the one the service builds with by default, or the service
    alone when neither is known.
    """
    factory = cfg.get("factory")
    if factory:
        return f"factory:{factory}"
    service = str(cfg.get("service", default_service))
    model = cfg.get(model_key) or (default_models or {}).get(service.lower())
    return f"{service}/{model}" if model else service


def _llm_identity(cfg: dict) -> str:
    """What a ``judge.eval`` or ``simulator`` block builds, as the run summary names it."""
    return _svc_model(cfg, "ollama", "model", _DEFAULT_LLM_MODELS)


def _user_segments(scenario: EvalConfigured) -> list[_ConfigSegment]:
    """The ``user`` line's segments: modality, and the TTS the user's turns are spoken with."""
    segs = [("modality", "audio" if scenario.user_audio else "text", _CFG_MODALITY)]
    if scenario.user_speech:
        # The TTS "voice" is the speech config's model-equivalent.
        segs.append(("speech", _svc_model(scenario.user_speech, "?", "voice"), _CFG_SERVICE))
    return segs


def _judge_segments(scenario: EvalConfigured) -> list[_ConfigSegment]:
    """The ``judge`` line's segments: modality, the bot-speech STT, and the judge LLM."""
    segs = [("modality", "audio" if scenario.bot_audio else "text", _CFG_MODALITY)]
    if scenario.bot_audio:
        transcription = _svc_model(scenario.transcriber or {}, "whisper", "model")
        segs.append(("transcription", transcription, _CFG_SERVICE))
    segs.append(("eval", _llm_identity(scenario.judge), _CFG_EVAL))
    return segs


def _config_lines(lines: list[_ConfigLine], *, color: bool) -> str:
    """Render labeled config lines, ``label -> key: value | key: value``.

    With ``color``, the label is bold, the separators dim, and each keyword
    takes its segment's ANSI code.
    """

    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    arrow = paint(" -> ", _CFG_SEP)
    sep = paint(" | ", _CFG_SEP)

    def body(segs: list[_ConfigSegment] | str) -> str:
        if isinstance(segs, str):
            return segs
        return sep.join(f"{paint(key + ':', code)} {value}" for key, value, code in segs)

    return "\n".join(
        f"{paint(label.ljust(5), _CFG_LABEL)}{arrow}{body(segs)}" for label, segs in lines
    )


def describe_config(scenario: EvalConfigured, *, color: bool = False) -> str:
    """Two-line summary of a scenario's user + judge config, for pre-run logs.

    Args:
        scenario: The parsed scenario to summarize.
        color: When True, ANSI-color each segment's keyword by category (modality,
            service, judge LLM) so they're easy to tell apart.

    Returns:
        A ``user`` line and a ``judge`` line, each a set of ``key: value`` segments
        separated by ``|``, e.g.::

            user  -> modality: audio | speech: kokoro/af_heart
            judge -> modality: audio | transcription: moonshine/small-streaming | eval: ollama/gemma4:12b
    """
    lines: list[_ConfigLine] = [
        ("user", _user_segments(scenario)),
        ("judge", _judge_segments(scenario)),
    ]
    return _config_lines(lines, color=color)
