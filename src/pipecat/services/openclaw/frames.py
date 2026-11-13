#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frames carrying OpenClaw Gateway traffic through a pipeline.

The command frames mirror the Gateway's three methods, and the run frames
mirror what its ``chat`` event stream reports.
:class:`~pipecat.services.openclaw.gateway.OpenClawGatewayService` is what
translates between them and the socket.
"""

from dataclasses import dataclass

from pipecat.frames.frames import ControlFrame, DataFrame, SystemFrame
from pipecat.services.openclaw.client import RunStatus


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
class OpenClawStartedFrame(ControlFrame):
    """Frame indicating the Gateway accepted a run.

    Followed by zero or more :class:`OpenClawTextFrame` and exactly one
    :class:`OpenClawEndFrame`.

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
class OpenClawEndFrame(ControlFrame):
    """Frame indicating a run ended, and how.

    Carries the whole answer, so a consumer that wants only that can ignore
    every :class:`OpenClawTextFrame` and stay stateless. A run that ends
    without one leaves the text empty, and the frames that came before it are
    what the agent managed to say.

    Parameters:
        run_id: The run that ended.
        status: Whether it answered, was stopped, or failed.
        text: The agent's answer, or why there isn't one.
    """

    run_id: str
    status: RunStatus = "completed"
    text: str = ""
