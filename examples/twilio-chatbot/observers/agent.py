from pipecat.observers.base_observer import BaseObserver, FrameDirection, FramePushed


class AgentObserver(BaseObserver):
    def __init__(self, agent: Agent):
        self._processed_frames = set()
        self._frame_history = deque(maxlen=100)
        self.agent = agent

    async def on_push_frame(self, data: FramePushed):
        """Process frame events for turn tracking."""
        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)
        self._frame_history.append(data.frame.id)

        # If we've exceeded our history size, remove the oldest frame ID
        # from the set of processed frames.
        if len(self._processed_frames) > len(self._frame_history):
            # Rebuild the set from the current deque contents
            self._processed_frames = set(self._frame_history)

        if isinstance(data.frame, StartFrame):
            # Start the first turn immediately when the pipeline starts
            if self._turn_count == 0:
                await self._start_turn(data)
        elif isinstance(data.frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(data)
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(data)
        # A BotStoppedSpeakingFrame can arrive after a UserStartedSpeakingFrame following an interruption
        # We only want to end the turn if the bot was previously speaking
        elif isinstance(data.frame, BotStoppedSpeakingFrame) and self._is_bot_speaking:
            await self._handle_bot_stopped_speaking(data)

    def on_agent_started(self, agent: Agent):
        pass

    def on_agent_ended(self, agent: Agent):
        pass
