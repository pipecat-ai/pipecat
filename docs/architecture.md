# Pipecat architecture guide

## Frames

Frames can represent discrete chunks of data, for instance a chunk of text, a chunk of audio, or an image. They can also be used to as control flow, for instance a frame that indicates that there is no more data available, or that a user started or stopped talking. They can also represent more complex data structures, such as a message array used for an LLM completion.

## FrameProcessors

Frame processors operate on frames. Every frame processor implements a `process_frame` method that consumes one frame and produces zero or more frames. Frame processors can do simple transforms, such as concatenating text fragments into sentences, or they can treat frames as input for an AI Service, and emit chat completions based on message arrays or transform text into audio or images.

## Pipelines

Pipelines are lists of frame processors linked together. Frame processors can push frames upstream or downstream to their peers. A very simple pipeline might chain an LLM frame processor to a text-to-speech frame processor, with a transport as an output.

## Transports

Transports provide input and output frame processors to receive or send frames respectively. For example, the `DailyTransport` does this with a WebRTC session joined to a Daily.co room.
