#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frames carrying OpenClaw Gateway traffic through a pipeline.

The command frames mirror the Gateway's three methods, and the run frames
mirror what its ``chat`` event stream reports.
:class:`~pipecat.services.openclaw.processor.OpenClawGatewayProcessor` is what
translates between them and the socket.
"""

from dataclasses import dataclass

from pipecat.frames.frames import ControlFrame, DataFrame, SystemFrame


@dataclass
class OpenClawSendFrame(DataFrame):
    """Frame that starts a run.

    Parameters:
        message: What to send the agent, verbatim.
        session_key: Which OpenClaw session to run in. None uses the client's.
    """

    message: str
    session_key: str | None = None


@dataclass
class OpenClawSteerFrame(DataFrame):
    """Frame that redirects the run in flight onto a follow-up.

    The Gateway aborts the running turn and starts a replacement carrying the
    follow-up. Output keeps arriving on the run that is already streaming.

    Parameters:
        message: The follow-up, verbatim.
    """

    message: str


@dataclass
class OpenClawAbortFrame(SystemFrame):
    """Frame that stops the run in flight.

    A system frame, so it reaches the Gateway rather than queueing behind the
    output it is meant to stop.

    Parameters:
        reason: Logged, for working out later why something was aborted.
    """

    reason: str | None = None


@dataclass
class OpenClawRunStartedFrame(ControlFrame):
    """Frame indicating the Gateway accepted a run.

    Followed by zero or more :class:`OpenClawTextFrame` and exactly one of
    :class:`OpenClawRunCompletedFrame`, :class:`OpenClawRunCancelledFrame`, or
    :class:`OpenClawRunFailedFrame`.

    Parameters:
        run_id: The id the run is addressed by.
    """

    run_id: str


@dataclass
class OpenClawTextFrame(DataFrame):
    """Frame containing a chunk of a run's answer.

    Parameters:
        text: The chunk, as the agent produced it.
        run_id: The run that produced it.
    """

    text: str
    run_id: str


@dataclass
class OpenClawRunCompletedFrame(ControlFrame):
    """Frame indicating a run reached its answer.

    Parameters:
        run_id: The run that finished.
        text: The agent's final answer.
    """

    run_id: str
    text: str = ""


@dataclass
class OpenClawRunCancelledFrame(ControlFrame):
    """Frame indicating a run was stopped before it answered.

    Parameters:
        run_id: The run that was stopped.
        text: What the Gateway reported about the cancellation.
    """

    run_id: str
    text: str = ""


@dataclass
class OpenClawRunFailedFrame(ControlFrame):
    """Frame indicating a run ended without an answer.

    Parameters:
        run_id: The run that failed.
        error: Why it failed.
    """

    run_id: str
    error: str = ""
