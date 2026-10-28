#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's tagger: expression tags for an LLM's plain replies.

Part of expressive conversation (alpha). The tagger model labels each sentence of a
reply as it streams in: a delivery, slow spans for details, stressed words, and a sound
before or after. It reads the call so far and how the user sounded, and runs on the CPU
in about 15 milliseconds a sentence. The labels are ``TAGGER_DELIVERY`` and
``TAGGER_SOUNDS``, written as tags in the names the TTS service performs them by.

**Alpha.** These types are likely to change between releases.
"""

from pipecat.processors.tone.tagger.processor import PipecatToneTagger

__all__ = ["PipecatToneTagger"]
