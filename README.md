# Daily AI SDK

Build conversational, multi-modal AI apps with real-time voice and video, like this:

_Demo Video_

With built-in support for many of the best AI platforms (or [add your own](/docs)):

- Azure - DALL-E, ChatGPT, and Azure AI Text-to-Speech
- Deepgram - Speech-to-text, and Aura text-to-speech
- Eleven Labs text-to-speech
- Fal.ai image generation
- OpenAI DALL-E and ChatGPT
- Whisper local speech-to-text

## Step 1: Get Started

Installation here. Also sign up for a Daily account, I guess? also we need an ENV

Requires python 3.11 or later. Don't forget virtualenv

pip install vs download and build?

## Step 2: Build Things

Once you've got the SDK working, head over to the [docs folder](/docs) to start building!

---

# Old Readme

This SDK can help you build applications that participate in WebRTC meetings and use various AI services to interact with other participants.

## Build/Install

_Note that you may need to set up a virtual environment before following the instructions below. For instance, you might need to run the following from the root of the repo:_

```
python3 -m venv env
source env/bin/activate
```

From the root of this repo, run the following:

```
pip install -r requirements.txt
python -m build
```

This builds the package. To use the package locally (eg to run sample files), run

```
pip install --editable .
```

If you want to use this package from another directory, you can run:

```
pip install path_to_this_repo
```

## Running the samples

Tou can run the simple sample like so:

```
python src/examples/theoretical-to-real/01-say-one-thing.py -u <url of your Daily meeting> -k <your Daily API Key>
```

Note that the sample uses Azure's TTS and LLM services. You'll need to set the following environment variables for the sample to work:

```
AZURE_SPEECH_SERVICE_KEY
AZURE_SPEECH_SERVICE_REGION
AZURE_CHATGPT_KEY
AZURE_CHATGPT_ENDPOINT
AZURE_CHATGPT_DEPLOYMENT_ID
```

If you have those environment variables stored in an .env file, you can quickly load them into your terminal's environment by running this:

```bash
export $(grep -v '^#' .env | xargs)
```

## Overview

The Daily AI SDK allows you to build applications that can participate in WebRTC sessions and interact with AI Services. Some examples of what you can build with this:

- conversational bots that interact 1:1 with a user, using voice recognition and text-to-speech
- assistant bots that aggregate transcriptions from multiple participants in a meeting and provide realtime summaries or other AI-generated output.
- image-recognition bots
- etc

## Concepts

### Transport Service

The SDK provides one “transport service”, which is a wrapper around Daily’s `daily-python` client (tk add link). You can use this service to listen for events related to a WebRTC session, such as “a participant joined the meeting”.
The transport service also exposes a send queue, and a receive queue. You can use the send queue to send audio and video to the WebRTC session, and you can listen to the receive queue to see audio, video and transcription data from the WebRTC session.

### AI Services

The AI Service classes provide wrappers around various AI providers, and allow you to query LLMs, convert text to speech and make images from text. The audio and images can then be placed on the transport service’s send queue, where they’ll be sent to the WebRTC session.

### Queue Frames

Communication between the transport service and AI services, and between various AI services, takes place in Queue Frames. These frames contain an indication of the type of data as well as the data itself.

## Using Transports, AI Services and Frames

AI Services all define a `.run` method. This method consumes and generates `QueueFrame` frames. The kind of frames that can be consumed and generated depend on the kind of service. For instance, an LLM AI Service consumes `LLM_MESSAGE` frames (which define a history of interaction with an LLM) and emit `TEXT` frames (the response from the LLM).

The `.run` method is an `AsyncIterable`, and it takes an `iterable`, `AsyncIterable` or `asyncio.Queue` that produces QueueFrames as a parameter. This makes it easy to chain AI Services, and consume input from the Transport’s `receive_queue` .

AI Services also have a `.run_to_queue` method. This method is not an AsyncIterable, but instead sends processed QueueFrames to a queue. This makes it easy to send the output of an AI Service to the Transport’s `send_queue`.

AI Services also define convenience functions that let you bypass creating QueueFrames for some simple cases (eg. using the TTS service to convert a string to audio output and send that audio to the transport’s `send_queue`). See below for examples.

## Examples

### Say Something

The base TTS AI service exposes a `.say` method. After creating a transport and TTS service, you can use this method like so:

```
transport = DailyTransportService(...)
tts = AzureTTSService()
await tts.say("hello world", transport.send_queue)
```

This will call the TTS service to render the text to audio frames, then put the audio frames on the transport’s send queue. The transport will then send those frames along to the WebRTC session.

### Speak an LLM response

Given a system prompt contained in a `messages` array, you can emit the LLM’s response as audio with a chain like this:

```
transport = DailyTransportService(...) # setup parameters omitted
tts = AzureTTSService()
llm = AzureLLMService()
messages = [...] # system prompt omitted for brevity

await tts.run_to_queue(
  transport.send_queue,
  llm.run([QueueFrame.LLM_MESSAGES, messages])
)
```

In this code, the LLM service object sends the messages to Azure’s OpenAI implementation, which streams chunks back asynchronously. Those chunks are aggregated by the TTS Service to ensure the best audio response (TTS works best when it gets complete sentence, so it can inflect correctly), then sent to Azure’s TTS service, converted to audio frames, and sent to the WebRTC session via the Daily transport.

### Pre-cache an LLM response

Sometimes LLMs can be slower than we’d like for natural-feeling communication. Here’s an example where we take advantage of the time it takes to speak some pre-defined text to get a head start on the LLM response:

(TK link to 04- sample)

In this sample, we set up a buffer queue to receive the audio frames from the LLM response before while we are joining the call and start an asynchronous task to start filling this buffer:

```
    buffer_queue = asyncio.Queue()
    llm_response_task = asyncio.create_task(
        elevenlabs_tts.run_to_queue(
            buffer_queue,
            llm.run([QueueFrame(FrameType.LLM_MESSAGE, messages)]),
            True,
        )
    )
```

Then, when we’ve joined the call, we speak the static text:

```
        await azure_tts.say("My friend...", transport.send_queue)
```

As that text is being spoken, the asynchronous LLM task continues in the background. When the text is done, we pull the frames off the buffer queue and put them in the transport’s `send_queue`:

```
        async def buffer_to_send_queue():
            while True:
                frame = await buffer_queue.get()
                await transport.send_queue.put(frame)
                buffer_queue.task_done()
                if frame.frame_type == FrameType.END_STREAM:
                    break

        await asyncio.gather(llm_response_task, buffer_to_send_queue())

```

One thing to note here is the last parameter to `run_to_queue` in the first code clause above: this causes the `run_to_queue` method to send an `END_STREAM` frame when it’s done rendering. This lets us know when to stop our `buffer_to_send_queue` task above.
