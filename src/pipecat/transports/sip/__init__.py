#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SIP transport for Pipecat.

A native SIP endpoint built on the baresip-python binding: the bot itself
registers to a SIP server (or operates registration-less against a
trunk), answers and places calls, and exchanges audio, video, and DTMF
with the pipeline. Requires the ``sip`` extra::

    uv add "pipecat-ai[sip]"
"""
