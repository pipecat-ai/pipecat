# Daily AI SDK Architecture Guide

## Frames

Frames can represent discrete chunks of data, for instance a chunk of text, a chunk of audio, or an image. They can also be used to as control flow, for instance a frame that indicates that there is no more data available, or that a user started or stopped talking. They can also represent more complex data structures, such as a message array used for an LLM completion.

## FrameProcessors

Frame processors operate on frames. Every frame processor implements a `process_frame` method that consumes one frame and produces zero or more frames. Frame processors can do simple transforms, such as concatenating text fragments into sentences, or they can treat frames as input for an AI Service, and emit chat completions based on message arrays or transform text into audio or images.

## Pipelines

Pipelines are lists of frame processors that read from a source queue and send the processed frames to a sink queue. A very simple pipeline might chain an LLM frame processor to a text-to-speech frame processor, with a transport's send queue as its sync. Placing LLM message frames on the pipeline's source queue will cause the LLM's response to be spoken. See example #2 for an implementation of this.

## Transports

Transports provide a receive queue, which is input from "the outside world", and a sink queue, which is data that will be sent "to the outside world". The `LocalTransportService` does this with the local camera, mic, display and speaker. The `DailyTransportService` does this with a WebRTC session joined to a Daily.co room.
