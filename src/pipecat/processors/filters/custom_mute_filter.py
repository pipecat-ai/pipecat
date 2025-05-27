"""This file contains CustomMuteFilter which will handle the state to handle the interruptions."""

from loguru import logger

from pipecat.processors.filters.stt_mute_filter import STTMuteFilter


class TransportInputFilter:
    """This class will handle the state to handle the interruptions.

    It will mute the user when the api call node is started and unmute the user when the api call node is ended.
    """

    def __init__(self) -> None:
        self._is_mute = False

    async def should_mute(self, stt_mute_filter: STTMuteFilter) -> bool:
        """This function will return True if the frame is a STTMuteFrame."""
        return self._is_mute

    def mute(self) -> None:
        """This function will be called when the api call node is started."""
        logger.debug(f"-----------muting-----------")
        self._is_mute = True

    def unmute(self) -> None:
        """This function will be called when the api call node is ended."""
        logger.debug(f"-----------unmuting-----------")
        self._is_mute = False
