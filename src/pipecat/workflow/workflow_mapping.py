from ..services.cartesia import CartesiaTTSService
from ..services.openai import OpenAILLMService
from ..services.deepgram import DeepgramSTTService
from ..transports.services.daily import DailyTransport
from ..processors.frame_processor import FrameProcessor

# Map workflow types to their corresponding Python classes
WORKFLOW_MAPPING = {
    "inputs/audio_input": DailyTransport,
    "processors/speech_to_text": DeepgramSTTService,
    "processors/llm": OpenAILLMService,
    "processors/text_to_speech": CartesiaTTSService,
    "outputs/audio_output": DailyTransport,
}


def get_processor_class(node_type: str) -> type[FrameProcessor]:
    return WORKFLOW_MAPPING.get(node_type, FrameProcessor)
