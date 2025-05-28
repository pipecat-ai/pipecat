from abc import abstractmethod

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LastTurnFrame,
    SetTransferCallDataFrame,
    TransferCallFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AbstractActionManager:
    """This class is responsible for managing the actions."""

    def __init__(self):
        pass

    @abstractmethod
    def handle_frame(self, frame: Frame):
        """Handle the frame."""
        pass

    @abstractmethod
    async def end_of_turn_event_listener(
        self, turn_count: int, duration: float, was_interrupted: bool
    ):
        """End of turn event listener."""
        pass


class AgentActionProcessor(FrameProcessor):
    """This class is responsible for managing the actions."""

    class EndCallActionManager(AbstractActionManager):
        """This class is responsible for managing the end call action."""

        def __init__(self, action_processor: "AgentActionProcessor"):
            self._is_last_turn = False
            self._action_processor = action_processor

        def _set_end_call(self, is_last_turn: bool):
            self._is_last_turn = is_last_turn

        async def _handle_end_call(self):
            await self._action_processor.push_frame(EndFrame())

        def handle_frame(self, frame: LastTurnFrame):
            """Handle the frame."""
            self._set_end_call(frame.is_last_turn)

        def _reset(self):
            self._is_last_turn = False

        async def end_of_turn_event_listener(
            self, turn_count: int, duration: float, was_interrupted: bool
        ):
            """End of turn event listener."""
            logger.debug(f"End of turn event listener: {turn_count}, {duration}, {was_interrupted}")
            if was_interrupted:
                self._reset()
            elif self._is_last_turn:
                await self._handle_end_call()

    class TransferCallActionManager(AbstractActionManager):
        """This class is responsible for managing the transfer call action."""

        def __init__(self, action_processor: "AgentActionProcessor"):
            self._transfer_call_number = None
            self._action_processor = action_processor

        def set_transfer_call_data(self, transfer_call_number: str, conversation_id: str):
            """Set the transfer call data to be used when end of turn is triggered."""
            self._transfer_call_number = transfer_call_number
            self._conversation_id = conversation_id

        async def _handle_transfer_call(self):
            logger.info(f"Handling transfer call to {self._transfer_call_number}")
            await self._action_processor.push_frame(
                TransferCallFrame(
                    transfer_call_number=self._transfer_call_number,
                    conversation_id=self._conversation_id,
                )
            )

        def handle_frame(self, frame: TransferCallFrame):
            """Handle the frame."""
            logger.info(f"Received TransferCallFrame: {frame}")
            self._transfer_call_number = frame.transfer_call_number
            self._conversation_id = frame.conversation_id

        async def end_of_turn_event_listener(
            self, turn_count: int, duration: float, was_interrupted: bool
        ):
            """End of turn event listener."""
            logger.debug(
                f"End of turn event listener at TransferCallActionManager: {turn_count}, {duration}, {was_interrupted}"
            )
            if self._transfer_call_number:
                logger.info("Transfer call triggered by end of turn event")
                await self._handle_transfer_call()

    def __init__(self, turn_tracking_observer: TurnTrackingObserver):
        """Initialize the action processor."""
        super().__init__()
        self._end_call_manager = self.EndCallActionManager(self)
        self._transfer_call_manager = self.TransferCallActionManager(self)

        end_of_turn_event_listener = self._end_call_manager.end_of_turn_event_listener
        transfer_call_event_listener = self._transfer_call_manager.end_of_turn_event_listener

        @turn_tracking_observer.event_handler("on_turn_ended")
        async def on_turn_ended(self, turn_count: int, duration: float, was_interrupted: bool):
            """On turn ended event listener."""
            await end_of_turn_event_listener(turn_count, duration, was_interrupted)
            await transfer_call_event_listener(turn_count, duration, was_interrupted)

    def set_transfer_call_data(self, transfer_call_number: str, conversation_id: str):
        """Set transfer call data to be triggered on end of turn."""
        self._transfer_call_manager.set_transfer_call_data(transfer_call_number, conversation_id)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process the frame."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LastTurnFrame):
            logger.debug(f"Processing LastTurnFrame: {frame}")
            self._end_call_manager.handle_frame(frame)
        elif isinstance(frame, TransferCallFrame):
            logger.info(f"Processing TransferCallFrame in AgentActionProcessor: {frame}")
            self._transfer_call_manager.handle_frame(frame)
        elif isinstance(frame, SetTransferCallDataFrame):
            logger.info(f"Processing SetTransferCallDataFrame in AgentActionProcessor: {frame}")
            self.set_transfer_call_data(frame.transfer_call_number, frame.conversation_id)

        # Always push the frame downstream so other processors (like serializers) can handle it
        await self.push_frame(frame=frame, direction=direction)
