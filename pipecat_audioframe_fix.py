#!/usr/bin/env python3

"""
Pipecat AudioRawFrame Fix

This demonstrates the proper fix for the AudioRawFrame issue and provides
a patch that could be applied to the framework.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="INFO")

# Import current Pipecat components
from pipecat.frames.frames import Frame, DataFrame, SystemFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner


# ==============================================================================
# CURRENT PROBLEMATIC DESIGN (from Pipecat source)
# ==============================================================================

@dataclass
class ProblematicAudioRawFrame:
    """Current AudioRawFrame design - CAUSES OBSERVER ERRORS."""
    
    audio: bytes
    sample_rate: int
    num_channels: int
    num_frames: int = field(default=0, init=False)

    def __post_init__(self):
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))
    
    # ❌ PROBLEM: This class doesn't inherit from Frame!
    # ❌ Missing: id, name, pts, metadata, broadcast_sibling_id
    # ❌ Observers expect these attributes and crash


# ==============================================================================
# PROPOSED FIX - Make AudioRawFrame inherit from Frame
# ==============================================================================

@dataclass
class FixedAudioRawFrame(Frame):
    """FIXED AudioRawFrame that properly inherits from Frame."""
    
    audio: bytes
    sample_rate: int
    num_channels: int
    num_frames: int = field(default=0, init=False)

    def __post_init__(self):
        super().__post_init__()  # Initialize Frame attributes (id, name, etc.)
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))
    
    def __str__(self):
        return f"{self.name}(size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


# ==============================================================================
# ALTERNATIVE FIX - Create proper base classes
# ==============================================================================

@dataclass
class AudioFrameData:
    """Pure audio data mixin - no Frame inheritance."""
    
    audio: bytes
    sample_rate: int
    num_channels: int
    num_frames: int = field(default=0, init=False)

    def __post_init__(self):
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))


@dataclass
class AudioRawFrameFixed(DataFrame, AudioFrameData):
    """AudioRawFrame that properly inherits from DataFrame + AudioFrameData."""
    
    def __post_init__(self):
        super().__post_init__()  # Initialize both DataFrame and AudioFrameData
    
    def __str__(self):
        return f"{self.name}(size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass  
class InputAudioRawFrameFixed(SystemFrame, AudioFrameData):
    """Fixed InputAudioRawFrame - inherits from SystemFrame + AudioFrameData."""
    
    def __post_init__(self):
        super().__post_init__()
    
    def __str__(self):
        return f"{self.name}(source: {getattr(self, 'transport_source', 'unknown')}, size: {len(self.audio)}, frames: {self.num_frames})"


@dataclass
class OutputAudioRawFrameFixed(DataFrame, AudioFrameData):
    """Fixed OutputAudioRawFrame - inherits from DataFrame + AudioFrameData."""
    
    def __post_init__(self):
        super().__post_init__()
    
    def __str__(self):
        return f"{self.name}(dest: {getattr(self, 'transport_destination', 'unknown')}, size: {len(self.audio)}, frames: {self.num_frames})"


# ==============================================================================
# TEST THE FIXES
# ==============================================================================

class AudioFrameTester(FrameProcessor):
    """Processor to test audio frame handling."""
    
    def __init__(self, name: str):
        super().__init__()
        self.tester_name = name
        self.frames_processed = 0
    
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        
        self.frames_processed += 1
        
        # Check for required Frame attributes
        required_attrs = ['id', 'name', 'pts', 'metadata']
        missing_attrs = [attr for attr in required_attrs if not hasattr(frame, attr)]
        
        if missing_attrs:
            logger.error(f"❌ {self.tester_name}: Frame {type(frame).__name__} missing: {missing_attrs}")
        else:
            logger.success(f"✅ {self.tester_name}: Frame {type(frame).__name__} has all required attributes")
            
        # Check audio-specific attributes
        if hasattr(frame, 'audio'):
            logger.info(f"🎵 Audio frame: {len(frame.audio)} bytes, {frame.sample_rate}Hz, {frame.num_channels}ch")
        
        await self.push_frame(frame, direction)


async def test_problematic_frames():
    """Test the current problematic design."""
    
    logger.warning("🚨 Testing PROBLEMATIC frames (will show missing attributes)")
    
    tester = AudioFrameTester("ProblematicTest")
    pipeline = Pipeline([tester])
    worker = PipelineWorker(pipeline)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    
    async def send_problematic_frames():
        await asyncio.sleep(0.1)
        
        # This will show the problem
        problematic_frame = ProblematicAudioRawFrame(
            audio=b"test_audio_data" * 10,
            sample_rate=16000,
            num_channels=1
        )
        
        logger.warning(f"Sending {type(problematic_frame).__name__}")
        # Note: We can't actually send this through the pipeline because it would crash
        # Just check attributes manually
        
        required_attrs = ['id', 'name', 'pts', 'metadata']
        missing_attrs = [attr for attr in required_attrs if not hasattr(problematic_frame, attr)]
        
        if missing_attrs:
            logger.error(f"❌ ProblematicAudioRawFrame missing: {missing_attrs}")
        
        # Send an EndFrame to close gracefully
        await worker.queue_frames([])  # Empty list
        
        # Manually end
        from pipecat.frames.frames import EndFrame
        await worker.queue_frames([EndFrame()])
    
    try:
        await asyncio.wait_for(
            asyncio.gather(runner.run(), send_problematic_frames()),
            timeout=2.0
        )
    except Exception as e:
        logger.warning(f"Expected issue: {e}")


async def test_fixed_frames():
    """Test the fixed frame designs."""
    
    logger.info("✅ Testing FIXED frames (should work correctly)")
    
    tester = AudioFrameTester("FixedTest") 
    pipeline = Pipeline([tester])
    worker = PipelineWorker(pipeline)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    
    async def send_fixed_frames():
        await asyncio.sleep(0.1)
        
        # Test all the fixed frame types
        frames_to_test = [
            FixedAudioRawFrame(b"test1" * 20, 16000, 1),
            AudioRawFrameFixed(b"test2" * 20, 16000, 1),
            InputAudioRawFrameFixed(b"test3" * 20, 16000, 1),
            OutputAudioRawFrameFixed(b"test4" * 20, 16000, 1)
        ]
        
        for frame in frames_to_test:
            logger.info(f"Sending {type(frame).__name__}")
            await worker.queue_frames([frame])
            await asyncio.sleep(0.2)
        
        from pipecat.frames.frames import EndFrame
        await worker.queue_frames([EndFrame()])
    
    try:
        await asyncio.wait_for(
            asyncio.gather(runner.run(), send_fixed_frames()),
            timeout=5.0
        )
        logger.success("✅ All fixed frames processed successfully!")
    except Exception as e:
        logger.error(f"❌ Fixed frames test failed: {e}")


def analyze_inheritance_hierarchy():
    """Analyze the inheritance hierarchy of different solutions."""
    
    logger.info("🔍 Inheritance Hierarchy Analysis")
    logger.info("=" * 50)
    
    frame_types = [
        ("ProblematicAudioRawFrame", ProblematicAudioRawFrame),
        ("FixedAudioRawFrame", FixedAudioRawFrame),
        ("AudioRawFrameFixed", AudioRawFrameFixed),
        ("InputAudioRawFrameFixed", InputAudioRawFrameFixed),
        ("OutputAudioRawFrameFixed", OutputAudioRawFrameFixed)
    ]
    
    for name, frame_class in frame_types:
        mro = [cls.__name__ for cls in frame_class.__mro__]
        logger.info(f"{name}:")
        logger.info(f"  MRO: {' -> '.join(mro)}")
        
        # Check what it inherits from
        inherits_from_frame = issubclass(frame_class, Frame)
        logger.info(f"  Inherits from Frame: {inherits_from_frame}")
        
        # Test instantiation
        try:
            if name == "ProblematicAudioRawFrame":
                instance = frame_class(b"test", 16000, 1)
            else:
                instance = frame_class(b"test", 16000, 1)
                
            has_frame_attrs = all(hasattr(instance, attr) for attr in ['id', 'name', 'pts'])
            logger.info(f"  Has Frame attributes: {has_frame_attrs}")
            
            if has_frame_attrs:
                logger.success(f"  ✅ {name} is properly designed")
            else:
                logger.error(f"  ❌ {name} lacks Frame attributes")
                
        except Exception as e:
            logger.error(f"  ❌ {name} instantiation failed: {e}")
        
        logger.info("")


def generate_framework_patch():
    """Generate a patch for the Pipecat framework."""
    
    logger.info("🔧 Generating Framework Patch")
    logger.info("=" * 40)
    
    patch_code = '''
# PIPECAT FRAMEWORK PATCH
# File: src/pipecat/frames/frames.py

# OPTION 1: Make AudioRawFrame inherit from Frame (Breaking Change)
@dataclass
class AudioRawFrame(Frame):
    """A frame containing a chunk of raw audio.
    
    Now properly inherits from Frame to work with observers.
    """
    audio: bytes
    sample_rate: int
    num_channels: int
    num_frames: int = field(default=0, init=False)

    def __post_init__(self):
        super().__post_init__()  # Initialize Frame attributes
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

# OPTION 2: Add validation in observers (Non-breaking)
def validate_frame_for_observer(frame):
    """Validate that frame has required attributes for observers."""
    required_attrs = ['id', 'name', 'pts', 'metadata']
    
    if not all(hasattr(frame, attr) for attr in required_attrs):
        raise ValueError(
            f"Frame {type(frame).__name__} missing required attributes. "
            f"Frames used in pipelines must inherit from Frame class. "
            f"Use InputAudioRawFrame or OutputAudioRawFrame instead of AudioRawFrame."
        )

# OPTION 3: Deprecate AudioRawFrame direct usage
class AudioRawFrame:
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "AudioRawFrame should not be used directly. "
            "Use InputAudioRawFrame for input or OutputAudioRawFrame for output.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... rest of implementation
'''
    
    logger.info("Recommended patch options:")
    logger.info(patch_code)


async def main():
    """Run comprehensive fix testing."""
    
    logger.info("🔧 Pipecat AudioRawFrame Fix Analysis")
    logger.info("=" * 50)
    
    analyze_inheritance_hierarchy()
    
    await test_problematic_frames()
    await asyncio.sleep(1)
    
    await test_fixed_frames()
    await asyncio.sleep(1)
    
    generate_framework_patch()
    
    logger.info("\n🎯 Summary:")
    logger.info("The AudioRawFrame issue can be fixed by:")
    logger.info("1. Making AudioRawFrame inherit from Frame (breaking change)")
    logger.info("2. Adding validation in observers (non-breaking)")  
    logger.info("3. Deprecating direct AudioRawFrame usage (gradual migration)")
    logger.info("4. Better documentation about proper frame usage")


if __name__ == "__main__":
    asyncio.run(main())