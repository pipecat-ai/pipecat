"""Mock classes for pipecat.frames.frames to use in testing."""

class Frame:
    """Base Frame class."""
    pass


class UserStartedSpeakingFrame(Frame):
    """Frame emitted when a user starts speaking."""
    pass


class UserStoppedSpeakingFrame(Frame):
    """Frame emitted when a user stops speaking."""
    pass


class BotStartedSpeakingFrame(Frame):
    """Frame emitted when the bot starts speaking."""
    pass


class BotStoppedSpeakingFrame(Frame):
    """Frame emitted when the bot stops speaking."""
    pass


class BotInterruptionFrame(Frame):
    """Frame emitted when the bot is explicitly interrupted."""
    pass


class FrameDirection:
    """Mock for FrameDirection enum."""
    IN = "in"
    OUT = "out" 