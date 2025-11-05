#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import math
from typing import Optional

from pipecat.frames.frames import TranscriptionFrame


def extract_whisper_probability(frame: TranscriptionFrame) -> Optional[float]:
    """Extract probability from Whisper-based TranscriptionFrame result.

    Works with Groq, OpenAI Whisper, or other Whisper-based services that use
    verbose_json format with segments containing avg_logprob.

    Converts avg_logprob to probability.

    Args:
        frame: TranscriptionFrame with result from GroqSTTService or OpenAISTTService
            (when include_prob_metrics=True and using Whisper models).

    Returns:
        Probability (0-1) if available, None otherwise.

    Example:
        >>> from pipecat.services.groq.stt import GroqSTTService
        >>> from pipecat.services.whisper.utils import extract_whisper_probability
        >>>
        >>> stt = GroqSTTService(include_prob_metrics=True)
        >>> # ... use stt in pipeline ...
        >>> # In your frame processor:
        >>> if isinstance(frame, TranscriptionFrame):
        >>>     prob = extract_whisper_probability(frame)
        >>>     if prob:
        >>>         print(f"Transcription confidence: {prob:.2%}")
    """
    if not frame.result:
        return None

    # Whisper verbose_json format: response.segments[0].avg_logprob
    if hasattr(frame.result, "segments") and frame.result.segments:
        segment = frame.result.segments[0]
        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is not None:
            return math.exp(avg_logprob)

    return None


def extract_openai_gpt4o_logprobs(frame: TranscriptionFrame) -> Optional[list]:
    """Extract logprobs from OpenAI GPT-4o-transcribe TranscriptionFrame result.

    Args:
        frame: TranscriptionFrame with result from OpenAISTTService
            using GPT-4o-transcribe model (when include_prob_metrics=True).

    Returns:
        List of logprobs if available, None otherwise.

    Example:
        >>> from pipecat.services.openai.stt import OpenAISTTService
        >>> from pipecat.services.whisper.utils import extract_openai_gpt4o_logprobs
        >>>
        >>> stt = OpenAISTTService(model="gpt-4o-transcribe", include_prob_metrics=True)
        >>> # ... use stt in pipeline ...
        >>> # In your frame processor:
        >>> if isinstance(frame, TranscriptionFrame):
        >>>     logprobs = extract_openai_gpt4o_logprobs(frame)
        >>>     if logprobs:
        >>>         # Calculate average logprob
        >>>         avg_logprob = sum(logprobs) / len(logprobs)
        >>>         prob = math.exp(avg_logprob)
        >>>         print(f"Transcription confidence: {prob:.2%}")
    """
    if not frame.result:
        return None

    # OpenAI GPT-4o-transcribe format: response.logprobs
    if hasattr(frame.result, "logprobs"):
        return frame.result.logprobs

    return None
