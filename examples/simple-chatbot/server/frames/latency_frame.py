from pipecat.frames.frames import Frame


class LatencyFrame(Frame):
    """A custom frame that carries the server-measured latency for one turn."""

    def __init__(self, turn_id: int, latency_ms: float):
        super().__init__()
        self.turn_id = turn_id
        self.latency_ms = latency_ms
