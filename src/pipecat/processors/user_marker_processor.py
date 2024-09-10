from pipecat.frames.frames import AudioRawFrame, Frame, UserAudioFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserMarkerProcessor(FrameProcessor):
    """
    This class extends FrameProcessor, used to mark the user's audio in the pipeline.
    This FrameProcessor must be inserted after transport.input() so that the only
    AudioRaw it receives are from the user.
    """

    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            frame = UserAudioFrame(frame)
        await self.push_frame(frame, direction)
