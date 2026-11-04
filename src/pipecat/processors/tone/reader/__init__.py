#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's reader: how each user turn was meant, read from its words.

Part of expressive conversation (alpha). Pipecat Tone
(https://huggingface.co/pipecat-ai/tone-models) reads one turn of text, with the turn
before it, and returns the speaker's emotion, whether the turn is sarcastic,
the move it makes and whether it's only a listening sound, each with a
calibrated confidence. It runs on the CPU in a few milliseconds.

**Alpha.** These types are likely to change between releases.
"""

from pipecat.processors.tone.reader.processor import PipecatToneReader

__all__ = ["PipecatToneReader"]
