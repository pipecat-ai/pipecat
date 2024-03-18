# 01: Say One Thing

_video here - youtube?_

This example uses a text-to-speech (TTS) service to say one predefined sentence. But first, a quick overview of the general structure of these examples.

## Running the demos

All of the demos have something like this at the bottom of the file:

```python
if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
```

### `configure()`

The `configure()` function comes from `examples/foundational/support/runner.py`, and it allows you to configure the examples from the command line directly, or using environment variables:

```bash
python 01-say-one-thing.py -u https://YOUR_DOMAIN.daily.co/YOUR_ROOM -k YOUR_API_KEY
# or
DAILY_ROOM_URL=https://YOUR_DOMAIN.daily.co/YOUR_ROOM DAILY_API_KEY=YOUR_API_KEY python 01-say-one-thing.py
# or set DAILY_ROOM_URL and DAILY_API_KEY in a .env file
python 01-say-one-thing.py
```

You'll need a Daily account to run these demos. You can sign up for free at [daily.co](https://daily.co). Once you've signed up you can create a room from the [Dashboard](https://dashboard.daily.co/rooms), and grab [your API key](https://dashboard.daily.co/developers) while you're there.

Some functionality (such as transcription) requires the bot to have owner privileges in the room. `runner.py` uses the Daily REST API to create a meeting token with owner privileges. You can learn more about meeting tokens in the [Daily docs](https://docs.daily.co/reference/rest-api/meeting-tokens).

### `asyncio.run()`

The AI SDK makes heavy use of Python's `asyncio` module. [This is a reasonable intro to the topic](https://builtin.com/data-science/asyncio) if you haven't worked with `asyncio` and coroutines before.

You can learn a bit more about the specifics of how the Daily AI SDK uses coroutines in the [Architecture Guide](../architecture.md).

## The `main()` function

All of the examples have a `main()` function with a similar structure:

- Configure the transport
- Configure the AI service(s) used in the demo
- Configure any event listeners
- Define a processing pipeline
- Run the example's coroutine(s)

### Configuring the transport

The first section of the `main()` function configures the transport object:

```python
meeting_duration_minutes = 5
transport = DailyTransportService(
    room_url,
    None,
    "Say One Thing",
    meeting_duration_minutes,
)
transport.mic_enabled = True
```

The [Architecture Guide](../architecture.md) explains the transport object in more detail. In this case, we're configuring a Daily transport object and enabling the virtual microphone, so our bot can play audio.

### Configuring the services

As described in the [Architecture Guide](../architecture.md), 'a 'Service' is a class that processes 'Frames' as part of a 'Pipeline'. In this demo app, we'll only need one service: a text-to-speech generator. We can create an instance of the `ElevenLabsTTSService` class with this line of code:

```python
tts = ElevenLabsTTSService(aiohttp_session=session, api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=os.getenv("ELEVENLABS_VOICE_ID"))
```

You'll need to make sure and set those environment variables somewhere. The easiest way to do that is to copy the `example.env` file in the repo and rename it to `.env`, and then add your credentials to that file. `runner.py` loads the `python-dotenv` module and initializes it, making the values in that file available in the environment.

### Configuring event listeners

This part isn't strictly necessary for an app like this. You could include the contents of the `on_participant_joined` function directly in the body of the `main()` function, and it would run as soon as you started the script from the command line.

Instead, we can use an event handler to wait to run that code until someone else joins the meeting. We'll define a function called `greet_user()`, and use the `@transport.event_handler("on_participant_joined")` decorator to tell the SDK that we want to run that function whenever a user joins the room.

```python
@transport.event_handler("on_participant_joined")
async def greet_user(transport, participant):
    if participant["info"]["isLocal"]:
        return

    await tts.say(
        "Hello there, " + participant["info"]["userName"] + "!",
        transport.send_queue,
    )

    # wait for the output queue to be empty, then leave the meeting
    await transport.stop_when_done()
```

### Defining a processing pipeline

In this example, we don't actually have much of a processing pipeline! In fact, we're doing the whole thing inside the `greet_user()` function already.

Pipelines usually look like a bunch of nested calls to the `run()` or `run_to_queue()` function from different Services. In this example, we're using the `say()` function from the TTS service. This is effectively a convenience wrapper around the `run_to_queue()` function, which we'll discuss more later. It's important to `await` this function to ensure that the speech frames are queued for playback before the next line of code, because of the `stop_when_done()` function being called immediately afterward.

The output of the `say()` function goes to the transport's `send_queue`. This queue is the all-important connection between the world of the Services pipeline that's generating frames asynchronously and the ordered playback of audio and visual media in the WebRTC call.

### Running the coroutines

In this example, we don't actually have any separate processing pipelines—everything happens as a result of an event from the transport. So we only need to run the transport's coroutine, and await its completion:

```python
await transport.run()
```

In future examples, we'll run more processes in parallel. For now, this script can run until the transport exits—which will happen based on calling `stop_when_done()` in the `greet_user()` function.

## Next Steps

Next, we'll start connecting multiple AI services together by building a service pipeline.

## [02 - LLM Say One Thing »](02-llm-say-one-thing.md)
