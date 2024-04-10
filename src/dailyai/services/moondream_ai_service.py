from dailyai.pipeline.frames import ImageFrame, VisionImageFrame
from dailyai.services.ai_services import VisionService

from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch


def detect_device():
    """
    Detects the appropriate device to run on, and return the device and dtype.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


class MoondreamService(VisionService):
    def __init__(
        self,
        model_id="vikhyatk/moondream2",
        revision="2024-04-02",
        device=None
    ):
        super().__init__()

        if not device:
            device, dtype = detect_device()
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to(device=device, dtype=dtype)
        self._model.eval()

    async def run_vision(self, frame: VisionImageFrame) -> str:
        image = Image.frombytes("RGB", (frame.size[0], frame.size[1]), frame.image)
        image_embeds = self._model.encode_image(image)
        description = self._model.answer_question(
            image_embeds=image_embeds,
            question=frame.text,
            tokenizer=self._tokenizer)
        return description
