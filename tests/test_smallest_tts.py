from pipecat.services.smallest.tts import SmallestTTSModel, SmallestTTSService, SmallestTTSSettings
from pipecat.transcriptions.language import Language


def test_smallest_tts_uses_live_endpoint_and_payload_model():
    service = SmallestTTSService(
        api_key="test-key",
        settings=SmallestTTSSettings(
            model=SmallestTTSModel.LIGHTNING_V3_1.value,
            voice="sophia",
            language=Language.EN,
            speed=1.2,
        ),
    )

    assert service._build_websocket_url() == "wss://api.smallest.ai/waves/v1/tts/live"

    message = service._build_msg("hello")
    assert message["text"] == "hello"
    assert message["voice_id"] == "sophia"
    assert message["model"] == "lightning_v3.1"
    assert message["language"] == "en"
    assert message["speed"] == 1.2
    assert message["word_timestamps"] is True
    assert message["output_format"] == "pcm"


def test_smallest_tts_defaults_to_pro_model_voice_pair():
    service = SmallestTTSService(api_key="test-key")

    assert service._settings.model == "lightning_v3.1_pro"
    assert service._settings.voice == "meher"
