# Frame Aggregators

## Types of Aggregators
Aggregators are processor class.They aggragates frames rather than process it.

### 1. Gated Aggregator
Controls the flow of frames using custom gates:

- Accumulates frames based on gate conditions
- Custom open/close functions determine flow
- Maintains gate state (open/closed)
- Supports directional frame flow
- Blocks or passes frames based on gate state

Example:
```plaintext
Gate closed -> Accumulate frames
Gate opens -> Release accumulated frames
System frames always pass through
```

### 2. Gated OpenAI LLM Context Aggregator
Manages OpenAI LLM context with controlled release:

- Holds last received context frame
- Releases context on notifier signal
- Handles start and cancel frames
- Maintains context state between releases

### 3. LLM Response Aggregator
Combines LLM responses into complete messages:

- Processes start/end response signals
- Accumulates text during active state
- Handles interim results
- Supports interruption handling

Sequence examples:
```plaintext
here X stands for final TextFrame
S E -> None              (Start-End only)
S T E -> X              (Start-Text-End)
S I T E -> X           (Start-Interim-Text-End)
```

### 4. OpenAI LLM Context Aggregator
Manages OpenAI-specific context and messages with comprehensive state handling:

#### Message Management
- Maintains ordered list of chat completion messages
- Supports message addition, extension, and batch updates
- Preserves message role and content structure
- Handles standard message conversions

#### Tool and Function Integration
- Manages tool choices and parameters
- Supports tool state transitions (NOT_GIVEN to active)
- Processes function calls with arguments
- Handles tool call results and callbacks

#### Image Processing
- Base64 encodes images for OpenAI format
- Creates multi-part messages with text and images
- Supports JPEG image conversion and storage
- Maintains image request context

#### Context Serialization
- Custom JSON encoding for message history
- Handles binary data in logs (truncated hex format)
- Provides formatted context for logging
- Supports persistent storage formatting

#### Special Features
- Function call progress tracking
- Message format standardization
- History initialization support
- Clean logging of sensitive/binary data

Example Message Structure:
```plaintext
{
    "role": "user",
    "content": [
        {"type": "text", "text": "description"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64..."}}
    ]
}
```

Example Function Call:
```plaintext
Input: Function name, arguments, tool call ID
Process: Execute function with context
Output: Function result frame with outcome
```

### 5. Sentence Aggregator
Combines text into complete sentences:

- Detects sentence endings
- Aggregates partial text
- Emits on complete sentences
- Handles end conditions

Example:
```plaintext
Input: "Hello," -> None
Input: " world." -> "Hello, world."
```

### 6. User Response Aggregator
Processes user speech input:

- Handles speaking start/stop events
- Processes transcriptions
- Manages interim results
- Creates unified output

### 7. Vision Image Frame Aggregator
Pairs text descriptions with images:

- Combines text and image data
- Creates unified vision outputs
- Sequential processing
- Maintains pairing relationship

Example:
```plaintext
Input: Description text
Input: Image data
Output: Combined vision content
```

