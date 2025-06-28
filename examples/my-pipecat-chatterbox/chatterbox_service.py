import io
import torch
import torchaudio as ta

from pipecat.services.tts.tts import TTSService
from pipecat.frames.frames import AudioFrame

# Import the library you want to wrap
from chatterbox.tts import ChatterboxTTS


class ChatterboxTTSService(TTSService):
    """
    A custom Pipecat TTS service for the local 'resemble-ai/chatterbox' library.
    """

    def __init__(self):
        super().__init__()

        # Determine the best device for your M2 Mac (Metal Performance Shaders)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Chatterbox model on device: {self.device}")

        # Load the model only ONCE when the service is created.
        # This is very important for performance.
        self.model = ChatterboxTTS.from_pretrained(device=self.device)

    async def run_tts(self, text: str):
        """
        This is the main function Pipecat will call.
        It receives text and must yield AudioFrame objects.
        """
        print(f"Synthesizing text: '{text}'")

        # 1. Generate the audio waveform using the Chatterbox model
        wav = self.model.generate(text)

        # 2. Pipecat needs the raw audio bytes. We'll save the waveform
        #    to an in-memory WAV file to get those bytes.
        buffer = io.BytesIO()
        ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()

        # 3. Create an AudioFrame with the bytes and yield it to the pipeline.
        #    This sends the audio on to the next step (usually the transport).
        yield AudioFrame(audio_bytes)
