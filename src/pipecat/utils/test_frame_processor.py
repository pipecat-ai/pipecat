from typing import List
from pipecat.processors.frame_processor import FrameProcessor


class TestException(BaseException):
    pass


class TestFrameProcessor(FrameProcessor):
    def __init__(self, test_frames):
        self.test_frames = test_frames
        self._list_counter = 0
        super().__init__()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if not self.test_frames[0]:  # then we've run out of required frames but the generator is still going?
            raise TestException(f"Oops, got an extra frame, {frame}")
        if isinstance(self.test_frames[0], List):
            # We need to consume frames until we see the next frame type after this
            next_frame = self.test_frames[1]
            if isinstance(frame, next_frame):
                # we're done iterating the list I guess
                print(f"TestFrameProcessor got expected list exit frame: {frame}")
                # pop twice to get rid of the list, as well as the next frame
                self.test_frames.pop(0)
                self.test_frames.pop(0)
                self.list_counter = 0
            else:
                fl = self.test_frames[0]
                fl_el = fl[self._list_counter % len(fl)]
                if isinstance(frame, fl_el):
                    print(f"TestFrameProcessor got expected list frame: {frame}")
                    self._list_counter += 1
                else:
                    raise TestException(f"Inside a list, expected {fl_el} but got {frame}")

        else:
            if not isinstance(frame, self.test_frames[0]):
                raise TestException(f"Expected {self.test_frames[0]}, but got {frame}")
            print(f"TestFrameProcessor got expected frame: {frame}")
            self.test_frames.pop(0)
