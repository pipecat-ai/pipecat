- Added `HeyGenHttpTTSService`, a HeyGen Voice text-to-speech service that streams raw PCM from HeyGen's `/v3/models/audio/tts/stream` endpoint. It defaults to the `heygen-voice-1-turbo` model and the public voice Nadine, and is offered by `pipecat create`.

    ```python
    from pipecat.services.heygen.tts import HeyGenHttpTTSService

    tts = HeyGenHttpTTSService(
        api_key=os.environ["HEYGEN_API_KEY"],
        aiohttp_session=session,
        settings=HeyGenHttpTTSService.Settings(voice="<voice_id>"),
    )
    ```
