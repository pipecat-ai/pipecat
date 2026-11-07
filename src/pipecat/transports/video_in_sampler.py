#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Selection of the incoming video frames an input transport passes on."""

import time

from pipecat.frames.frames import UserImageRequestFrame
from pipecat.utils.deprecation import warn_deprecated

# How early a frame may arrive and still count as due, at most half an interval.
# It covers jitter in the incoming stream, which is about the same at any framerate.
JITTER_TOLERANCE_SECS = 0.05


def _capture_framerate(framerate: int | None, on_request_only: bool, method: str) -> int | None:
    """The sampler framerate for a ``capture_participant_video()`` call's arguments.

    A sampler passes on only the frames that answer image requests at a
    framerate of ``0``, which is how ``on_request_only`` is applied.
    """
    if on_request_only:
        return 0
    if framerate == 0:
        warn_deprecated(
            f"`{method}(framerate=0)` is deprecated since 1.13.0 and will be removed "
            "in 2.0.0. Use `on_request_only=True` instead.",
            stacklevel=3,
        )
    return framerate


class _VideoInSampler:
    """Decides which frames of one incoming video source a transport passes on.

    A user's camera or screen share sends frames at its own rate, often 15-30 per
    second, while a bot usually wants fewer: a vision bot may want one a second,
    or only a frame when it asks for one. An input transport keeps one sampler per
    video source and calls :meth:`sample` for every frame that arrives; the
    sampler says whether to pass that frame on or drop it.

    The framerate decides which frames are passed on:

    - ``None``: every frame.
    - ``0``: only frames that answer an image request.
    - ``N > 0``: ``N`` frames per second, on the schedule described below.

    Whatever the framerate, an image request queued with :meth:`add_request` is
    answered by the next frame that arrives, which is passed on with the request.

    For ``N > 0``, frames are due on a fixed schedule, one interval (``1 / N``
    seconds) apart:

    - **On schedule.** A frame is due once its scheduled time comes, or up to a
      tolerance before it, since frames never arrive exactly on time. The next
      frame is scheduled one interval after this frame's *scheduled* time, not its
      arrival time, so the rate doesn't drift. A frame arriving late moves the
      schedule to its arrival, so a stream slower than the framerate passes every
      frame rather than falling further and further behind.
    - **Fresh start.** The first frame, a frame that answers a request, and the
      first frame after a stall (more than an interval past its scheduled time)
      start a new schedule from their arrival. After a stall this passes on one
      frame instead of a burst catching up on the missed ones.

    The tolerance is the smaller of :data:`JITTER_TOLERANCE_SECS` and half an
    interval. At low framerates the next frame after any frame is therefore close
    to a full interval later; at framerates near the stream's own rate it covers
    the stream's jitter, so every frame passes.
    """

    def __init__(self, framerate: int | None):
        """Initialize the sampler.

        Args:
            framerate: Frames per second to pass on. ``0`` passes on frames only
                to answer requests, and ``None`` passes on every frame.
        """
        self._framerate = framerate
        # When the next frame is due. Zero before the first frame, which then
        # counts as arriving after a stall and starts the schedule.
        self._next_time = 0.0
        self._requests: list[UserImageRequestFrame] = []

    @property
    def framerate(self) -> int | None:
        """The frames per second this sampler passes on."""
        return self._framerate

    @framerate.setter
    def framerate(self, framerate: int | None):
        self._framerate = framerate

    def add_request(self, request: UserImageRequestFrame):
        """Queue an image request for the next frame to answer.

        Args:
            request: The image request to answer.
        """
        self._requests.append(request)

    def take_requests(self) -> list[UserImageRequestFrame]:
        """Remove and return the image requests waiting for a frame.

        Returns:
            The waiting requests, oldest first.
        """
        requests, self._requests = self._requests, []
        return requests

    def sample(self) -> tuple[bool, UserImageRequestFrame | None]:
        """Decide whether to pass on the frame that just arrived.

        Returns:
            A tuple of ``(due, request)``: ``due`` is whether to pass the frame on, and
            ``request`` is the ``UserImageRequestFrame`` it answers, or ``None``.
        """
        # A waiting request is always answered by this frame.
        request = self._requests.pop(0) if self._requests else None

        if self._framerate is None:
            return True, request
        if self._framerate == 0:
            return request is not None, request

        now = time.time()
        interval = 1 / self._framerate
        tolerance = min(JITTER_TOLERANCE_SECS, interval / 2)

        if request is None and now < self._next_time - tolerance:
            # Too early: the next frame isn't due yet.
            return False, None

        if request or now - self._next_time > interval:
            # Fresh start (a request, the first frame, or a stall): the next
            # frame is due one interval from now.
            self._next_time = now + interval
        else:
            # On schedule: one interval after this frame's scheduled time, or
            # after its arrival if it came late.
            self._next_time = max(self._next_time, now - tolerance) + interval

        return True, request


class _VideoInSamplers:
    """An input transport's video samplers, by participant and video source."""

    def __init__(self):
        """Initialize with no sources being sampled."""
        self._samplers: dict[tuple[str, str], _VideoInSampler] = {}

    def capture(self, participant_id: str, video_source: str, framerate: int | None):
        """Start sampling a source, or change its framerate.

        Changing the framerate keeps the source's sampler, so image requests
        already waiting on it are still answered.

        Args:
            participant_id: The participant whose video this is.
            video_source: The video source, e.g. ``"camera"`` or ``"screenVideo"``.
            framerate: Frames per second to pass on. ``0`` passes on frames only
                to answer requests, and ``None`` passes on every frame.
        """
        sampler = self._samplers.get((participant_id, video_source))
        if sampler:
            sampler.framerate = framerate
        else:
            self._samplers[(participant_id, video_source)] = _VideoInSampler(framerate)

    def is_capturing(self, participant_id: str, video_source: str) -> bool:
        """Whether a source is being sampled.

        Args:
            participant_id: The participant whose video this is.
            video_source: The video source.

        Returns:
            Whether :meth:`capture` was called for the source.
        """
        return (participant_id, video_source) in self._samplers

    def add_request(
        self, participant_id: str, video_source: str, request: UserImageRequestFrame
    ) -> bool:
        """Queue an image request for the next frame of a source to answer.

        Args:
            participant_id: The participant whose video is requested.
            video_source: The video source requested.
            request: The image request to answer.

        Returns:
            Whether the request was queued; ``False`` if the source isn't being
            sampled, so nothing would answer it.
        """
        sampler = self._samplers.get((participant_id, video_source))
        if sampler:
            sampler.add_request(request)
        return sampler is not None

    def sample(
        self, participant_id: str, video_source: str
    ) -> tuple[bool, UserImageRequestFrame | None]:
        """Decide whether to pass on the frame that just arrived from a source.

        Args:
            participant_id: The participant the frame came from.
            video_source: The video source the frame came from.

        Returns:
            A tuple of ``(due, request)``, as from :meth:`_VideoInSampler.sample`.
            A frame from a source that isn't being sampled is never due.
        """
        sampler = self._samplers.get((participant_id, video_source))
        return sampler.sample() if sampler else (False, None)

    def take_requests(
        self, participant_id: str, video_source: str | None = None
    ) -> list[UserImageRequestFrame]:
        """Remove and return image requests waiting for a frame, keeping the samplers.

        Args:
            participant_id: The participant whose requests to take.
            video_source: The video source whose requests to take, or ``None`` for
                all of the participant's sources.

        Returns:
            The waiting requests.
        """
        requests = []
        for (participant, source), sampler in self._samplers.items():
            if participant == participant_id and video_source in (None, source):
                requests.extend(sampler.take_requests())
        return requests

    def remove_participant(self, participant_id: str) -> list[UserImageRequestFrame]:
        """Stop sampling every video source of a participant.

        Args:
            participant_id: The participant to forget, e.g. one who left.

        Returns:
            The image requests that were waiting on the participant's video.
        """
        requests = self.take_requests(participant_id)
        for key in [key for key in self._samplers if key[0] == participant_id]:
            del self._samplers[key]
        return requests

    def clear(self) -> list[UserImageRequestFrame]:
        """Stop sampling every video source.

        Returns:
            The image requests that were waiting on any of them.
        """
        requests = [r for sampler in self._samplers.values() for r in sampler.take_requests()]
        self._samplers.clear()
        return requests
