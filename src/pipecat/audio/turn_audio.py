#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Collect a conversation's audio, filed under the turn each clip belongs to.

:class:`~pipecat.processors.audio.audio_buffer_processor.AudioBufferProcessor`
reports turn audio a run of speech at a time, and
:class:`~pipecat.observers.turn_tracking_observer.TurnTrackingObserver` numbers
the turns. Lining the two up takes more than reading the current turn number when
a clip arrives, and both hazards are silent — the audio still arrives, just
attached to the wrong turn:

- **A turn holds several runs of speech.** The audio events fire every time a
  speaker stops, while a turn ends only once the other side takes over. A user
  who pauses mid-thought, or a bot resuming after a function call, reports more
  than once for one turn.
- **A barge-in reports the bot's audio late.** Interrupting the bot ends its turn
  and starts the next one, and only then does the cut-off audio arrive, by which
  point the tracker has moved on. ``on_turn_ended`` flags the interruption
  first, which is what puts the clip back on the turn it was spoken in.

Runs are kept separate so callers that need the pauses between them can have
them; :attr:`TurnAudio.user` and :attr:`TurnAudio.bot` join them for callers that
don't.
"""

from dataclasses import dataclass, field

from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


@dataclass
class TurnAudio:
    """One turn's audio, as raw mono PCM.

    Parameters:
        number: The turn this audio belongs to.
        user_runs: Each run of user speech during the turn, in order.
        bot_runs: Each run of bot speech during the turn, in order.
    """

    number: int
    user_runs: list[bytes] = field(default_factory=list)
    bot_runs: list[bytes] = field(default_factory=list)

    @property
    def user(self) -> bytes:
        """The turn's user speech, runs joined without the pauses between them."""
        return b"".join(self.user_runs)

    @property
    def bot(self) -> bytes:
        """The turn's bot speech, runs joined without the pauses between them."""
        return b"".join(self.bot_runs)


class TurnAudioCollector:
    """Collects an audio buffer processor's turn audio, by turn.

    Audio is retained in memory for the life of the collector, so a collector is
    scoped to one conversation.

    Example::

        audio_buffer = AudioBufferProcessor(enable_turn_audio=True)
        # ...place audio_buffer in the pipeline...

        collector = TurnAudioCollector()
        collector.attach(audio_buffer, worker.turn_tracking_observer)

        # Once the conversation is over:
        for turn in collector.turns():
            save(turn.number, turn.user, turn.bot, collector.sample_rate)
    """

    def __init__(self):
        """Initialize the collector."""
        self._turns: dict[int, TurnAudio] = {}
        self._sample_rate: int | None = None
        self._turn_number = 0
        self._interrupted_turn: int | None = None

    @property
    def sample_rate(self) -> int | None:
        """Sample rate of the collected audio, or None until some arrives."""
        return self._sample_rate

    def attach(
        self, audio_buffer: AudioBufferProcessor, turn_tracker: TurnTrackingObserver
    ) -> None:
        """Start collecting.

        Args:
            audio_buffer: The pipeline's audio buffer processor. It must be
                constructed with ``enable_turn_audio=True``, which is what makes
                it report turn audio at all.
            turn_tracker: The turn tracking observer for the same pipeline, e.g.
                a pipeline worker's ``turn_tracking_observer``. It owns turn
                numbering and reports what an interruption does to a turn.
        """

        @turn_tracker.event_handler("on_turn_started")
        async def on_turn_started(tracker, turn_number: int):
            self._turn_number = turn_number

        @turn_tracker.event_handler("on_turn_ended")
        async def on_turn_ended(tracker, turn_number: int, duration: float, was_interrupted: bool):
            # A turn ends as interrupted only when the bot was speaking, and the
            # barge-in that ends it starts the next turn before the cut-off audio
            # is reported.
            if was_interrupted:
                self._interrupted_turn = turn_number

        @audio_buffer.event_handler("on_user_turn_audio_data")
        async def on_user_turn_audio_data(buffer, audio, sample_rate, num_channels):
            self._store(bytes(audio), sample_rate, user=True)

        @audio_buffer.event_handler("on_bot_turn_audio_data")
        async def on_bot_turn_audio_data(buffer, audio, sample_rate, num_channels):
            self._store(bytes(audio), sample_rate, user=False)

    def turns(self) -> list[TurnAudio]:
        """Every turn that produced audio, in turn order."""
        return [self._turns[number] for number in sorted(self._turns)]

    def turn_numbers(self) -> list[int]:
        """The numbers of the turns that produced audio, in order."""
        return sorted(self._turns)

    def _store(self, audio: bytes, sample_rate: int, *, user: bool) -> None:
        # Turn 1 opens with the pipeline, so audio only goes unfiled when it
        # arrives before the tracker has announced any turn at all.
        if not audio or not self._turn_number:
            return
        self._sample_rate = sample_rate

        turn_number = self._turn_number
        if not user and self._interrupted_turn == self._turn_number - 1:
            # Only the turn that just ended can still have audio in flight.
            # Anything older ended without the bot speaking, e.g. at shutdown.
            turn_number = self._interrupted_turn
            self._interrupted_turn = None

        turn = self._turns.setdefault(turn_number, TurnAudio(number=turn_number))
        runs = turn.user_runs if user else turn.bot_runs
        runs.append(audio)
