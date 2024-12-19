# A Frame's Progress

This guide walks you through the journey of a frame as it moves through the Pipecat system, from user input to final output. Understanding this process will help you grasp how Pipecat processes data and how different components interact.

## Overview of the Process

1. User Input and Transcription
2. Frame Creation and Pipeline Entry
3. Frame Processing
4. Audio Generation
5. Output and System Reset

## Detailed Frame Journey

### 1. User Input and Transcription

The process begins when a user speaks to the system.

- User says: "Hello, LLM"
- A cloud transcription service converts this to text
- The transcription is delivered to the Transport

![A transcript frame arrives](images/frame-progress-01.png)

### 2. Frame Creation and Pipeline Entry

The Transport creates a frame from the transcription and introduces it to the pipeline.

- A Transcription frame is created
- The frame is placed in the Pipeline's source queue

![Frame in source queue](images/frame-progress-02.png)

### 3. Frame Processing

The frame moves through various processors in the pipeline.

#### LLM User Message Aggregator
- Receives the Transcription frame
- Updates the LLM Context with the user's message
- Yields an LLM Message Frame with the updated context

![Update context](images/frame-progress-04.png)
![Update context](images/frame-progress-05.png)

#### LLM Frame Processor
- Creates a streaming chat completion based on the LLM context
- Yields Text Frames with chunks of the response

![LLM yields Text](images/frame-progress-06.png)
![LLM yields more Text](images/frame-progress-07.png)

### 4. Audio Generation

The text response is converted to audio.

#### TTS Frame Processor
- Aggregates Text Frames until a full sentence is formed
- Generates streaming audio based on the complete sentence
- Yields Audio frames

![TTS yields Audio](images/frame-progress-08.png)

### 5. Output and System Reset

The audio is prepared for output, and the system readies itself for the next interaction.

#### LLM Assistant Message Aggregator
- Passes Audio frames through unchanged
- Updates the LLM Context with the full LLM response when processing is complete

#### Pipeline Output
- Places Audio frames in the sink queue for the Transport to handle
- Continues processing frames in parallel

![sink queue](images/frame-progress-10.png)

#### System Reset
- After processing all frames, the system returns to a quiet state
- Waits for the next user input to restart the process

![response end](images/frame-progress-15.png)

## Key Concepts

- **Concurrent Processing**: The source and sink queues allow for parallel processing between the Pipeline and external components.
- **Frame Processor Convention**: Processors should immediately yield frames they don't process.
- **Context Updates**: Both user input and system responses update the LLM Context, maintaining the conversation state.

By understanding this flow, you can better conceptualize how Pipecat handles data processing and how to design efficient pipelines for your specific use cases.
