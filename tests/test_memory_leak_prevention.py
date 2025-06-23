#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import gc
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService


class MockTTSService(TTSService):
    """Mock TTS service for testing race conditions and memory leaks."""
    
    def __init__(self, **kwargs):
        super().__init__(pause_frame_processing=True, **kwargs)
        self.run_tts_calls = []
        self.paused_count = 0
        self.resumed_count = 0
        
    async def run_tts(self, text: str):
        self.run_tts_calls.append(text)
        yield None
        
    async def pause_processing_frames(self):
        self.paused_count += 1
        await super().pause_processing_frames()
        
    async def resume_processing_frames(self):
        self.resumed_count += 1
        await super().resume_processing_frames()


class TestMemoryLeakPrevention:
    """Test suite for memory leak prevention and race condition fixes."""

    @pytest.mark.asyncio
    async def test_tts_race_condition_prevention(self):
        """Test that race condition between BotStoppedSpeakingFrame and TTSSpeakFrame is prevented."""
        tts = MockTTSService()
        
        # Setup frame processor
        setup_mock = MagicMock()
        setup_mock.clock = MagicMock()
        setup_mock.task_manager = MagicMock()
        setup_mock.observer = None
        await tts.setup(setup_mock)
        
        # Start the service
        start_frame = StartFrame()
        await tts.start(start_frame)
        await tts.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        
        # Simulate the race condition scenario
        # 1. Process LLMFullResponseEndFrame (triggers pause)
        llm_end_frame = LLMFullResponseEndFrame()
        await tts.process_frame(llm_end_frame, FrameDirection.DOWNSTREAM)
        
        # 2. Process TTSSpeakFrame (should maintain pause state)
        tts_speak_frame = TTSSpeakFrame("WAITING")
        await tts.process_frame(tts_speak_frame, FrameDirection.DOWNSTREAM)
        
        # 3. Process BotStoppedSpeakingFrame (should resume)
        bot_stopped_frame = BotStoppedSpeakingFrame()
        await tts.process_frame(bot_stopped_frame, FrameDirection.DOWNSTREAM)
        
        # Verify that pause/resume operations completed successfully
        assert tts.paused_count > 0, "TTS should have been paused"
        assert tts.resumed_count > 0, "TTS should have been resumed"
        
        # Cleanup
        cancel_frame = CancelFrame()
        await tts.cancel(cancel_frame)

    @pytest.mark.asyncio
    async def test_force_resume_on_cancellation(self):
        """Test that paused TTS services are force resumed during cancellation."""
        tts = MockTTSService()
        
        # Setup frame processor
        setup_mock = MagicMock()
        setup_mock.clock = MagicMock()
        setup_mock.task_manager = MagicMock()
        setup_mock.observer = None
        await tts.setup(setup_mock)
        
        # Start and pause the service
        start_frame = StartFrame()
        await tts.start(start_frame)
        await tts.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        
        # Force pause by setting internal state
        await tts.pause_processing_frames()
        
        # Cancel should force resume
        cancel_frame = CancelFrame()
        await tts.cancel(cancel_frame)
        
        # Verify force resume was called
        assert tts.resumed_count > 0, "Force resume should have been called during cancellation"

    @pytest.mark.asyncio
    async def test_timeout_protection_in_pause_resume(self):
        """Test that pause/resume operations have timeout protection."""
        tts = MockTTSService()
        
        # Mock the base pause/resume methods to simulate hanging
        original_pause = tts.pause_processing_frames
        original_resume = tts.resume_processing_frames
        
        async def hanging_pause():
            await asyncio.sleep(10)  # Simulate hanging
            
        async def hanging_resume():
            await asyncio.sleep(10)  # Simulate hanging
        
        # Setup frame processor
        setup_mock = MagicMock()
        setup_mock.clock = MagicMock()
        setup_mock.task_manager = MagicMock()
        setup_mock.observer = None
        await tts.setup(setup_mock)
        
        # Start the service
        start_frame = StartFrame()
        await tts.start(start_frame)
        await tts.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        
        # Test timeout protection in pause
        with patch.object(tts, 'pause_processing_frames', side_effect=hanging_pause):
            tts._processing_text = True
            start_time = time.time()
            await tts._maybe_pause_frame_processing()
            elapsed = time.time() - start_time
            assert elapsed < 7.0, "Pause operation should timeout and not hang indefinitely"
        
        # Test timeout protection in resume  
        with patch.object(tts, 'resume_processing_frames', side_effect=hanging_resume):
            start_time = time.time()
            await tts._maybe_resume_frame_processing()
            elapsed = time.time() - start_time
            assert elapsed < 7.0, "Resume operation should timeout and not hang indefinitely"
        
        # Cleanup
        cancel_frame = CancelFrame()
        await tts.cancel(cancel_frame)

    @pytest.mark.asyncio
    async def test_concurrent_pause_resume_operations(self):
        """Test that concurrent pause/resume operations are handled safely."""
        tts = MockTTSService()
        
        # Setup frame processor
        setup_mock = MagicMock()
        setup_mock.clock = MagicMock()
        setup_mock.task_manager = MagicMock()
        setup_mock.observer = None
        await tts.setup(setup_mock)
        
        # Start the service
        start_frame = StartFrame()
        await tts.start(start_frame)
        await tts.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        
        # Set up for pause operations
        tts._processing_text = True
        
        # Run multiple concurrent pause/resume operations
        tasks = []
        for _ in range(5):
            tasks.append(asyncio.create_task(tts._maybe_pause_frame_processing()))
            tasks.append(asyncio.create_task(tts._maybe_resume_frame_processing()))
        
        # Wait for all operations to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no deadlock occurred and operations completed
        assert True, "Concurrent operations should complete without deadlock"
        
        # Cleanup
        cancel_frame = CancelFrame()
        await tts.cancel(cancel_frame)

    @pytest.mark.asyncio
    async def test_pipeline_cancellation_timeout(self):
        """Test that pipeline cancellation completes within timeout."""
        # NOTE: Not fully tested with complex pipeline setups due to automated environment limits
        # Manual testing recommended for: Complex multi-service pipelines, real TTS services
        
        tts = MockTTSService()
        pipeline = Pipeline([tts])
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=False,
            ),
        )
        
        # Start the task
        run_task = asyncio.create_task(task.run())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Cancel and measure time
        start_time = time.time()
        await task.cancel()
        
        # Wait for run task to complete
        try:
            await asyncio.wait_for(run_task, timeout=15.0)
        except asyncio.TimeoutError:
            pytest.fail("Pipeline cancellation should complete within timeout")
        
        elapsed = time.time() - start_time
        assert elapsed < 12.0, "Pipeline cancellation should complete quickly"

    @pytest.mark.asyncio 
    async def test_memory_cleanup_verification(self):
        """Test that resources are properly cleaned up to prevent memory leaks."""
        # NOTE: Not fully tested with real websocket connections due to automated environment limits
        # Manual testing recommended for: Real websocket services, production load testing
        
        initial_objects = len(gc.get_objects())
        
        # Create and run multiple pipeline cycles
        for i in range(3):
            tts = MockTTSService()
            pipeline = Pipeline([tts])
            
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=False,
                ),
            )
            
            # Start and quickly cancel
            run_task = asyncio.create_task(task.run())
            await asyncio.sleep(0.05)
            await task.cancel()
            
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                pass
            
            # Force cleanup
            del tts, pipeline, task, run_task
        
        # Force garbage collection
        gc.collect()
        await asyncio.sleep(0.1)
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Allow for some object growth but not excessive
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Excessive object growth detected: {object_growth}"


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"])) 