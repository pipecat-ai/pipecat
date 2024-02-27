import unittest

from unittest.mock import MagicMock, patch

from dailyai.queue_frame import AudioQueueFrame, ImageQueueFrame
from dailyai.services.ai_services import PipeService

class TestDailyTransport(unittest.IsolatedAsyncioTestCase):
    def test_pipe_chain(self):
        pipe1 = PipeService()
