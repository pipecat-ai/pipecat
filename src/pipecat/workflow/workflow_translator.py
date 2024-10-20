import json

from typing import List, Dict, Any
from .workflow_mapping import get_processor_class
from ..processors.frame_processor import FrameProcessor
from ..transports.services.daily import DailyParams
from ..processors.aggregators.openai_llm_context import OpenAILLMContext
from ..audio.vad.silero import SileroVADAnalyzer


def load_workflow(file_path: str) -> Dict[str, Any]:
    print(f"Loading workflow from file: {file_path}")
    try:
        with open(file_path, "r") as f:
            workflow = json.load(f)
        print(f"Workflow loaded successfully: {workflow}")
        return workflow
    except Exception as e:
        print(f"Error loading workflow: {e}")
        raise


def create_processor(node: Dict[str, Any], next_node: Dict[str, Any] = None) -> FrameProcessor:
    print(f"Creating processor for node: {node['id']} of type: {node['type']}")
    processor_class = get_processor_class(node["type"])
    print(f"Processor class: {processor_class}")

    # Extract relevant properties for initialization
    init_params = {}
    if node["type"] == "frames/audio_input":
        init_params = {
            "room_url": node["properties"]["daily_url"],
            "token": None,
            "bot_name": "PipecatBot",
            "params": DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        }
    elif node["type"] == "frame_processors/speech_to_text":
        init_params = {
            "api_key": "sample_api_key",
        }
    elif node["type"] == "frame_processors/text_to_speech":
        init_params = {
            "api_key": node["properties"]["cartesia_api_key"],
            "voice_id": node["properties"]["voice"],
            "model": node["properties"]["model"],
        }

    print(f"Initialization parameters: {init_params}")
    processor = processor_class(**init_params)
    print(f"Processor created: {processor}")
    return processor


def create_pipeline(workflow: Dict[str, Any]) -> List[FrameProcessor]:
    print("Creating pipeline from workflow")
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = workflow["links"]

    print(f"Nodes: {nodes}")
    print(f"Links: {links}")

    # Create a dictionary to store processors
    processors = {}
    daily_transport = None
    llm_service = None
    context_aggregator = None

    # Create processors for each node
    for node_id, node in nodes.items():
        print(f"Creating processor for node: {node_id}")

        if node["type"] == "frames/audio_input":
            daily_transport = create_processor(node)
            processors[node_id] = {"processor": daily_transport, "type": node["type"]}
        elif node["type"] == "frame_processors/audio_output_transport":
            if daily_transport is None:
                raise ValueError("Audio output transport node found before audio input node")
            processors[node_id] = {"processor": daily_transport, "type": node["type"]}
        elif node["type"] == "frame_processors/llm":
            llm_service = create_processor(node)
            processors[node_id] = {"processor": llm_service, "type": node["type"]}
            context = OpenAILLMContext(
                [{"role": "system", "content": "You are a helpful assistant."}]
            )
            context_aggregator = llm_service.create_context_aggregator(context)
        else:
            processors[node_id] = {"processor": create_processor(node), "type": node["type"]}

    # Create the pipeline based on the links
    pipeline = []
    for link in links:
        source_id, _, _, target_id, _, _ = link
        print(f"Processing link: {source_id} -> {target_id}")

        if source_id not in pipeline:
            print(f"Adding source processor: {source_id}")
            if processors[source_id]["type"] == "frames/audio_input":
                pipeline.append(processors[source_id]["processor"].input())
            else:
                pipeline.append(processors[source_id]["processor"])

            # Add context_aggregator.user() before LLM
            if processors[target_id]["type"] == "frame_processors/llm" and context_aggregator:
                pipeline.append(context_aggregator.user())

        if target_id not in pipeline and target_id in processors:
            print(f"Adding target processor: {target_id}")
            if processors[target_id]["type"] == "frame_processors/audio_output_transport":
                pipeline.append(processors[target_id]["processor"].output())
            else:
                pipeline.append(processors[target_id]["processor"])

            # Add context_aggregator.assistant() after audio output transport
            if (
                processors[target_id]["type"] == "frame_processors/audio_output_transport"
                and context_aggregator
            ):
                pipeline.append(context_aggregator.assistant())

    print(f"Pipeline created with {len(pipeline)} processors")
    print(f"Pipeline: {pipeline}")

    return pipeline


def translate_workflow(file_path: str) -> List[FrameProcessor]:
    print(f"Translating workflow from file: {file_path}")
    workflow = load_workflow(file_path)
    pipeline = create_pipeline(workflow)
    print("Workflow translation completed")
    return pipeline
