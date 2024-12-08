# Filters Documentation

## FrameFilter

A processor that filters frames based on their types. It automatically allows standard system-related frames (AppFrame, ControlFrame, and SystemFrame) to pass through, along with any additionally specified frame types.

**Key Features:**
- Selective frame passing based on type
- Built-in allowance for system frames
- Preserves frame direction during processing
- Type-based filtering for pipeline customization

## FunctionFilter

A flexible filter that uses a custom function to determine which frames should pass through the pipeline. It provides granular control over frame filtering based on any criteria you define.

**Key Features:**
- Custom function-based filtering
- Automatic passage of SystemFrames
- Boolean-based decision making
- Maintains directional frame flow
- Asynchronous function support

## NullFilter

A simple but effective filter that blocks all frame transmission through the pipeline. This filter acts as a complete stop for frame flow.

**Key Features:**
- Blocks all frame transmission
- Useful for testing scenarios
- Helps in debugging pipeline sections
- Simple implementation for frame blocking

## WakeCheckFilter

A sophisticated filter designed for wake phrase detection in transcription frames. It monitors conversation flow and manages wake states for different participants.

**Key Features:**
- Wake phrase detection in transcriptions
- Per-participant state tracking
- Configurable keepalive timeout
- Case-insensitive phrase matching
- Error handling with automatic error frame generation
- Conversation state management

**States:**
- IDLE: Initial state waiting for wake phrase detection
- AWAKE: Active state after wake phrase detection

**Functionality:**
- Accumulates transcription text
- Monitors for wake phrases
- Manages conversation timeouts
- Handles multiple participant tracking
- Provides error reporting

## WakeNotifierFilter

A monitoring filter that triggers notifications when specific frame types meet custom criteria. It observes frame flow without interrupting it.

**Key Features:**
- Type-specific frame monitoring
- Custom condition evaluation
- Non-blocking frame passage
- Asynchronous notification support
- Multiple frame type support
- Flexible predicate configuration

This filter combines frame type filtering with custom conditions to create a powerful notification system while maintaining pipeline flow.

---

Each filter serves a specific purpose in the pipeline system, from basic frame filtering to complex wake word detection and notification systems. They can be combined to create sophisticated processing pipelines tailored to specific needs.
