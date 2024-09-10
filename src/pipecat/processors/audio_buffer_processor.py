from pipecat.frames.frames import (AudioRawFrame, BotStartedSpeakingFrame,
                                   BotStoppedSpeakingFrame, Frame,
                                   UserAudioFrame, UserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioBufferProcessor(FrameProcessor):

    def __init__(self):
        """
        Initialize the AudioBufferProcessor.

        This constructor sets up the initial state for audio processing:
        - audio_buffer: A bytearray to store incoming audio data.
        - num_channels: The number of audio channels (initialized as None).
        - sample_rate: The sample rate of the audio (initialized as None).
        - assistant_audio: A boolean flag to indicate if assistant audio is being processed.
        - user_audio: A boolean flag to indicate if user audio is being processed.

        The num_channels and sample_rate are set to None initially and will be
        populated when the first audio frame is processed.
        """
        super().__init__()
        self.audio_buffer = bytearray()
        self.num_channels = None
        self.sample_rate = None
        self.assistant_audio = False
        self.user_audio = False

    def has_audio(self):
        return (
            self.audio_buffer and
            len(self.audio_buffer) > 0 and
            self.num_channels and
            self.sample_rate
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) or isinstance(frame, UserAudioFrame):
            if self.num_channels is None:
                self.num_channels = frame.num_channels
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_audio = False

        if isinstance(frame, BotStartedSpeakingFrame):
            self.assistant_audio = True
            self.user_audio = False  # do not capture user audio if assistant is speaking
        if isinstance(frame, BotStoppedSpeakingFrame):
            self.assistant_audio = False
            # Capture user audio if assistant is not speaking, even if it's silence, the point
            # here is to capture so that the conversation is as close to reality as possible.
            # This is important for evaluation and metrics capture.
            self.user_audio = True

        # only include audio from the user if the user is speaking, this is because audio from the user's
        # mic is always coming in. if we include all the user's audio there will be a long latency before
        # the user starts speaking because all of the user's silence during the assistant's speech will have been
        # added to the buffer.
        if isinstance(frame, UserAudioFrame) and self.user_audio:
            self.audio_buffer.extend(frame.audio)

        # include all audio from the assistant
        if (
            isinstance(frame, AudioRawFrame)
            and not isinstance(frame, UserAudioFrame)
        ):
            self.audio_buffer.extend(frame.audio)

        # do not push the user's audio frame, doing so will result in echo
        if not isinstance(frame, UserAudioFrame):
            await self.push_frame(frame, direction)
