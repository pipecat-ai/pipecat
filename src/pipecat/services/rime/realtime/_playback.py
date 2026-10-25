#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bounded response ownership and Prism playback receipts."""

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import uuid4

from rimelabs_sdk import realtime as r

from pipecat.frames.frames import AudioPlaybackReportFrame, InterruptionCause


@dataclass
class Output:
    """One Prism output and its globally unique transport identity."""

    ref: r.OutputRef
    context_id: str = field(default_factory=lambda: uuid4().hex)
    text: str = ""
    report: AudioPlaybackReportFrame | None = None
    audio_bytes: int = 0
    closed: bool = False
    discarded: bool = False


@dataclass
class Response:
    """Response policy and handlers captured at its starting revision."""

    ref: r.ResponseRef
    interrupt_on_speech: bool
    handlers: dict
    input_ids: tuple[str, ...] = ()
    input_request_id: int | None = None
    client_message_id: str | None = None
    outputs: dict[str, Output] = field(default_factory=dict)
    ended: bool = False
    finishing: bool = False
    finished: bool = False
    stopped: bool = False
    cause: InterruptionCause | None = None
    audio_response: bool = True
    explicit_audio_response: bool = False
    calls: set[str] = field(default_factory=set)


class Playback:
    """Keep SDK references until every output has a terminal report."""

    def __init__(self):
        """Create an independent owner for one conversation."""
        self.responses: OrderedDict[str, Response] = OrderedDict()
        self.outputs: dict[str, tuple[Response, Output]] = {}
        self.destination: str | None = None
        self._destination_set = False

    def start(self, event: r.ResponseStarted, policy: bool, handlers: dict) -> Response:
        """Capture response state before releasing its output."""
        if len(self.responses) >= 128:
            key, oldest = next(iter(self.responses.items()))
            if not oldest.finished or oldest.calls:
                raise RuntimeError(
                    "Prism response retention limit has unresolved playback or tools"
                )
            self.responses.pop(key)
            for output in oldest.outputs.values():
                self.outputs.pop(output.context_id, None)
        response = Response(event.response, policy, handlers, event.input_item_ids)
        self.responses[event.response.response_id] = response
        return response

    def message(self, ref: r.OutputRef) -> Output:
        """Assign output identity independent of provider response IDs."""
        response = self.responses[ref.response.response_id]
        if len(response.outputs) >= 128:
            raise RuntimeError("Prism output retention limit exceeded")
        output = Output(ref, discarded=response.stopped)
        response.outputs[ref.item_id] = output
        self.outputs[output.context_id] = response, output
        return output

    def accept_report(self, frame: AudioPlaybackReportFrame) -> tuple[Response, Output] | None:
        """Deduplicate reports and enforce one authoritative destination."""
        owned = self.outputs.get(frame.context_id)
        if owned is None:
            return None
        response, output = owned
        if self._destination_set and frame.transport_destination != self.destination:
            raise ValueError("Prism requires one authoritative audio output destination")
        self._destination_set = True
        self.destination = frame.transport_destination
        if output.report is not None:
            return None
        if frame.played_ms is not None and (
            frame.played_ms < 0 or not math.isfinite(frame.played_ms)
        ):
            raise ValueError("Playback duration must be finite and nonnegative")
        output.report = frame
        output.discarded |= frame.outcome != "completed"
        return response, output

    def finished_report(self, response: Response) -> r.PlaybackFinished | None:
        """Report completion only after generation and every output have ended."""
        if (
            response.finished
            or not response.ended
            or any(output.report is None for output in response.outputs.values())
        ):
            return None
        reports = [
            output.report for output in response.outputs.values() if output.report is not None
        ]
        measured = [report.played_ms for report in reports if report.played_ms is not None]
        duration = sum(measured) if len(measured) == len(reports) else None
        response.finishing = True
        return r.PlaybackFinished(response=response.ref, played_ms=duration)
