import numpy as np

from pipecat.audio.vad.ten import TenVadAnalyzer


def test_ten_vad_init():
    vad = TenVadAnalyzer(sample_rate=16000)
    assert vad is not None


def test_ten_vad_sample_rate():
    vad = TenVadAnalyzer(sample_rate=16000)
    vad.set_sample_rate(16000)


def test_ten_vad_frame_requirement():
    vad = TenVadAnalyzer(sample_rate=16000)
    assert vad.num_frames_required() > 0


def test_ten_vad_confidence_range():
    vad = TenVadAnalyzer(sample_rate=16000)

    silence = np.zeros(256, dtype=np.int16).tobytes()
    confidence = vad.voice_confidence(silence)

    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
