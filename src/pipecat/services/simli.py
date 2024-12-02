import asyncio

from pipecat.frames.frames import (
    Frame,
    OutputImageRawFrame,
    TTSAudioRawFrame,
    StartInterruptionFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import numpy as np
from av import AudioFrame
from av.audio.resampler import AudioResampler
from simli import SimliClient, SimliConfig


class SimliVideoService(FrameProcessor):
    def __init__(
        self, simliConfig: SimliConfig, useTurnServer=False, latencyInterval=60
    ):
        super().__init__()
        self.simliClient = SimliClient(simliConfig, useTurnServer, latencyInterval)

        self.pipecatResampler: AudioResampler = None
        self.name = "SimliAi"
        self.ready = False
        self.simliResampler = AudioResampler("s16", 1, 16000)
        self.AudioTask: asyncio.Task = None
        self.VideoTask: asyncio.Task = None

    async def startConnection(self):
        await self.simliClient.Initialize()
        self.ready = True
        # Create task to consume and process audio and video
        self.AudioTask = asyncio.create_task(self.consume_and_process_audio())
        self.VideoTask = asyncio.create_task(self.consume_and_process_video())

    async def consume_and_process_audio(self):
        async for audio_frame in self.simliClient.getAudioStreamIterator():
            # Process the audio frame
            try:
                resampledFrames = self.pipecatResampler.resample(audio_frame)
                for resampled_frame in resampledFrames:
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=resampled_frame.to_ndarray().tobytes(),
                            sample_rate=self.pipecatResampler.rate,
                            num_channels=1,
                        ),
                    )
            except Exception as e:
                print(e)
                import traceback

                traceback.print_exc()

    async def consume_and_process_video(self):
        async for video_frame in self.simliClient.getVideoStreamIterator(
            targetFormat="rgb24"
        ):
            # Process the video frame
            convertedFrame: OutputImageRawFrame = OutputImageRawFrame(
                image=video_frame.to_rgb().to_image().tobytes(),
                size=(video_frame.width, video_frame.height),
                format="RGB",
            )
            convertedFrame.pts = video_frame.pts
            await self.push_frame(
                convertedFrame,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            # Send audio frame to Simli
            try:
                if self.ready:
                    AudioFrame
                    oldFrame = AudioFrame.from_ndarray(
                        np.frombuffer(frame.audio, dtype=np.int16)[None, :],
                        layout=frame.num_channels,
                    )
                    oldFrame.sample_rate = frame.sample_rate
                    if self.pipecatResampler is None:
                        self.pipecatResampler = AudioResampler(
                            "s16", oldFrame.layout, oldFrame.sample_rate
                        )

                    resampledFrame = self.simliResampler.resample(oldFrame)
                    for frame in resampledFrame:
                        await self.simliClient.send(
                            frame.to_ndarray().astype(np.int16).tobytes()
                        )
                    return
                else:
                    print(
                        "Simli Connection is not Initialized properly, passing audio to next processor"
                    )
                    await self.push_frame(frame, direction)
            except Exception as e:
                print(e)
                import traceback

                traceback.print_exc()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.simliClient.stop()
            self.AudioTask.cancel()
            self.VideoTask.cancel()

        elif isinstance(frame, StartInterruptionFrame):
            await self.simliClient.clearBuffer()

        await self.push_frame(frame, direction)
