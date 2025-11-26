# TurnAwareTranscriptProcessor Example

## Overview

The `TurnAwareTranscriptProcessor` combines user and assistant transcript tracking with turn boundary detection. It correctly handles interruptions by only capturing what was actually spoken.

## Basic Usage

```python
from pipecat.processors.transcript_processor import TurnAwareTranscriptProcessor

# Create the processor
turn_processor = TurnAwareTranscriptProcessor()

# Register event handlers
@turn_processor.event_handler("on_turn_started")
async def handle_turn_started(processor, turn_number):
    print(f"Turn {turn_number} started")

@turn_processor.event_handler("on_turn_ended")
async def handle_turn_ended(processor, turn_number, user_text, assistant_text, was_interrupted):
    print(f"\nTurn {turn_number} ended:")
    print(f"  User said: {user_text}")
    print(f"  Assistant said: {assistant_text}")
    print(f"  Was interrupted: {was_interrupted}")

@turn_processor.event_handler("on_transcript_update")
async def handle_transcript_update(processor, frame):
    for msg in frame.messages:
        print(f"[{msg.role}]: {msg.content}")

# Add to pipeline
pipeline = Pipeline([
    transport.input(),
    stt,
    turn_processor,  # Process transcripts and track turns
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

## Features

1. **Turn Boundary Detection**: Automatically detects when turns start and end based on user and bot speaking patterns
2. **Interruption Handling**: Correctly captures only what was actually spoken when interruptions occur
3. **Real-time Transcripts**: Emits transcript messages for both user and assistant speech
4. **Turn Events**: Provides start/end events with accumulated transcripts for each turn

## Events

### on_turn_started
Emitted when a new turn begins (user starts speaking).

**Handler signature**: `async def handler(processor, turn_number)`

### on_turn_ended
Emitted when a turn ends with accumulated transcripts.

**Handler signature**: `async def handler(processor, turn_number, user_transcript, assistant_transcript, was_interrupted)`

### on_transcript_update  
Inherited from `BaseTranscriptProcessor`, emitted for individual transcript messages.

**Handler signature**: `async def handler(processor, frame)`

## Turn Logic

- Turns start when the user begins speaking (`UserStartedSpeakingFrame`)
- Turns end when:
  - The user starts speaking again (previous turn ends, new turn starts)
  - The bot is interrupted (`InterruptionFrame`)
  - The pipeline ends (`EndFrame`/`CancelFrame`)

## Integration with OpenTelemetry

You can use turn events to enrich OpenTelemetry spans:

```python
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver

turn_tracker = TurnTrackingObserver()
turn_tracer = TurnTraceObserver(turn_tracker)
turn_processor = TurnAwareTranscriptProcessor()

@turn_processor.event_handler("on_turn_ended")
async def add_transcripts_to_span(processor, turn_number, user_text, assistant_text, interrupted):
    # Get current span and add transcript data
    from opentelemetry import trace
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_attribute("turn.user_text", user_text)
        current_span.set_attribute("turn.assistant_text", assistant_text)
```

## Notes

- The processor handles async frame processing correctly by delaying turn end until frames are processed
- Works with word-level timestamps from TTS services like Cartesia
- Accumulates both user (`TranscriptionFrame`) and assistant (`TTSTextFrame`) speech
- Emits individual transcript messages in addition to turn-level aggregation
