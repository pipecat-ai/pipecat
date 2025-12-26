#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility functions for extracting probability metrics from STT services."""

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

    Example::

        from pipecat.services.groq.stt import GroqSTTService
        from pipecat.services.whisper.utils import extract_whisper_probability

        stt = GroqSTTService(include_prob_metrics=True)
        # ... use stt in pipeline ...
        # In your frame processor:
        if isinstance(frame, TranscriptionFrame):
            prob = extract_whisper_probability(frame)
            if prob:
                print(f"Transcription confidence: {prob:.2%}")
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


def extract_openai_gpt4o_probability(frame: TranscriptionFrame) -> Optional[float]:
    """Extract probability from OpenAI GPT-4o-transcribe TranscriptionFrame result.

    Args:
        frame: TranscriptionFrame with result from OpenAISTTService
            using GPT-4o-transcribe model (when include_prob_metrics=True).

    Returns:
        Probability (0-1) if available, None otherwise.

    Example::

        from pipecat.services.openai.stt import OpenAISTTService
        from pipecat.services.whisper.utils import extract_openai_gpt4o_probability

        stt = OpenAISTTService(model="gpt-4o-transcribe", include_prob_metrics=True)
        # ... use stt in pipeline ...
        # In your frame processor:
        if isinstance(frame, TranscriptionFrame):
            prob = extract_openai_gpt4o_probability(frame)
            if prob:
                print(f"Transcription confidence: {prob:.2%}")
    """
    if not frame.result:
        return None

    # OpenAI GPT-4o-transcribe format: response.logprobs
    if hasattr(frame.result, "logprobs"):
        logprobs = frame.result.logprobs
        if logprobs:
            # Calculate average logprob and convert to probability
            avg_logprob = sum(logprobs) / len(logprobs)
            return math.exp(avg_logprob)

    return None


def extract_deepgram_probability(frame: TranscriptionFrame) -> Optional[float]:
    """Extract probability from Deepgram TranscriptionFrame result.

    Args:
        frame: TranscriptionFrame with result from DeepgramSTTService.

    Returns:
        Probability (0-1) if available, None otherwise.
        Returns alternative-level confidence if available, otherwise calculates
        average confidence from word-level confidences.

    Example::

        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.services.whisper.utils import extract_deepgram_probability

        stt = DeepgramSTTService()
        # ... use stt in pipeline ...
        # In your frame processor:
        if isinstance(frame, TranscriptionFrame):
            prob = extract_deepgram_probability(frame)
            if prob:
                print(f"Transcription confidence: {prob:.2%}")
    """
    if not frame.result:
        return None

    result = frame.result
    if hasattr(result, "channel") and result.channel:
        if hasattr(result.channel, "alternatives") and result.channel.alternatives:
            alt = result.channel.alternatives[0]
            conf = getattr(alt, "confidence", None)
            if conf is not None:
                return float(conf)

            words = getattr(alt, "words", None)
            if words:
                word_confs = [getattr(w, "confidence", None) for w in words]
                word_confs = [c for c in word_confs if c is not None]
                if word_confs:
                    return float(sum(word_confs) / len(word_confs))

    return None
