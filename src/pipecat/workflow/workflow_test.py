import asyncio
import os
from dotenv import load_dotenv
from ..pipeline.pipeline import Pipeline
from ..pipeline.runner import PipelineRunner
from ..pipeline.task import PipelineTask, PipelineParams
from .workflow_translator import translate_workflow
from ..frames.frames import LLMMessagesFrame
from ..transports.services.daily import DailyTransport

load_dotenv(override=True)


async def main():
    print("Starting workflow test")

    # Update the path to the workflow.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workflow_path = os.path.join(script_dir, "workflow.json")
    print(f"Workflow path: {workflow_path}")

    # Translate the workflow to a list of processors
    print("Translating workflow to processors")
    processors = translate_workflow(workflow_path)
    print(f"Processors created: {processors}")

    # Create a pipeline from the processors
    print("Creating pipeline")
    pipeline = Pipeline(processors)
    print(f"Pipeline created: {pipeline}")

    # Create a pipeline task
    print("Creating pipeline task")
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    print(f"Pipeline task created: {task}")

    # Create a pipeline runner
    print("Creating pipeline runner")
    runner = PipelineRunner()
    print(f"Pipeline runner created: {runner}")

    # # Add event handler
    # daily_transport = next(p for p in processors if isinstance(p, DailyTransport))

    # @daily_transport.event_handler("on_first_participant_joined")
    # async def on_first_participant_joined(transport, participant):
    #     transport.capture_participant_transcription(participant["id"])
    #     # Kick off the conversation.
    #     messages = [{"role": "system", "content": "Please introduce yourself to the user."}]
    #     await task.queue_frames([LLMMessagesFrame(messages)])

    # Run the pipeline
    print("Running the pipeline")
    try:
        await runner.run(task)
        print("Pipeline execution completed successfully")
    except Exception as e:
        print(f"Error during pipeline execution: {e}")

    print("Workflow test completed")


if __name__ == "__main__":
    print("Starting main execution")
    asyncio.run(main())
    print("Main execution completed")
