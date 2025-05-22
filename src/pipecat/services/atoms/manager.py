from abc import abstractmethod

from pipecat.frames.frames import EndFrame, Frame, LastTurnFrame, TransferCallFrame
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

        async def end_of_turn_event_listener(
            self, turn_count: int, duration: float, was_interrupted: bool
        ):
            """End of turn event listener."""
            if self._is_last_turn:
                await self._handle_end_call()

    class TransferCallActionManager(AbstractActionManager):
        """This class is responsible for managing the transfer call action."""

        def __init__(self, action_processor: "AgentActionProcessor"):
            self._transfer_call_number = None
            self._reason = None
            self._conversation_id = None
            self._action_processor = action_processor

        def _handle_transfer_call(self, frame: TransferCallFrame):
            self._transfer_call_number = frame.transfer_call_number
            self._reason = frame.reason
            self._conversation_id = frame.conversation_id

        def handle_frame(self, frame: TransferCallFrame):
            """Handle the frame."""
            self._handle_transfer_call(frame)

        async def end_of_turn_event_listener(
            self, turn_count: int, duration: float, was_interrupted: bool
        ):
            """End of turn event listener."""
            pass

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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process the frame."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LastTurnFrame):
            self._end_call_manager.handle_frame(frame)
        elif isinstance(frame, TransferCallFrame):
            self._transfer_call_manager.handle_frame(frame)

        await self.push_frame(frame=frame)
