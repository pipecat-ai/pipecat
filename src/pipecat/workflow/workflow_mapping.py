from ..services.cartesia import CartesiaTTSService
from ..services.openai import OpenAILLMService
from ..services.deepgram import DeepgramSTTService
from ..transports.services.daily import DailyTransport
from ..processors.aggregators.openai_llm_context import OpenAILLMContext
from ..processors.frame_processor import FrameProcessor

# Map workflow types to their corresponding Python classes
WORKFLOW_MAPPING = {
    "frames/audio_input": DailyTransport,
    "frame_processors/speech_to_text": DeepgramSTTService,
    "frame_processors/llm": OpenAILLMService,
    "frame_processors/text_to_speech": CartesiaTTSService,
    "frame_processors/audio_output_transport": DailyTransport,
}


def get_processor_class(node_type: str) -> type[FrameProcessor]:
    return WORKFLOW_MAPPING.get(node_type, FrameProcessor)
