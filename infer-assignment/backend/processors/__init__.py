"""Custom processors for voice agent recording."""

from .freeze_simulator import FreezeSimulatorProcessor
from .transcript_collector import TranscriptCollectorProcessor
from .bot_text_collector import BotTextCollectorProcessor

__all__ = [
    "FreezeSimulatorProcessor",
    "TranscriptCollectorProcessor", 
    "BotTextCollectorProcessor"
]
