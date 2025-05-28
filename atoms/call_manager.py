import asyncio
import time

from loguru import logger


class CallManager:
    """Manages active calls for graceful shutdown."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.accepting_calls = True
        self.lock = asyncio.Lock()

    async def initiate_shutdown(self, max_wait_time: int = 300):
        """Initiate shutdown of the call manager."""
        async with self.lock:
            self.accepting_calls = False
            self.shutdown_event.set()

        logger.info(f"Waiting for  active calls to complete...")

        start_time = time.time()
        while self.active_calls and (time.time() - start_time) < max_wait_time:
            await asyncio.sleep(1)
            logger.info(f"Still waiting for calls...")

        if self.active_calls:
            logger.warning(f"Timeout reached. calls still active")
        else:
            logger.info("All calls completed successfully")

    async def can_accept_call(self) -> bool:
        """Check if a call can be accepted."""
        async with self.lock:
            if not self.accepting_calls:
                return False
            return True

    async def shutdown(self):
        """Shutdown the call manager."""
        self.accepting_calls = False
        self.shutdown_event.set()


call_manager = CallManager()
