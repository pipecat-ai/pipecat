from loguru import logger
import sys
from openai.types.chat import ChatCompletionToolParam
import asyncio
from main.wss import ConnectionManager
import json
import random
import string

# Configure the logger
logger.add(sys.stderr, level="DEBUG")

# Base class for tools with common functionality
manager = ConnectionManager()


class BaseTool:
    def __init__(self, room_id) -> None:
        self.start_callback_function = None
        self.main_function = None
        self.function_definition = None
        self.room_id = room_id

    def main(self):
        # Define the async function for start_callback_function
        async def start_callback_function(function_name, llm, context):
            print("running")

        self.start_callback_function = start_callback_function

        # Define the async function for main_function
        async def main_function(function_name, tool_call_id, args, llm, context, result_callback):

            await result_callback({"status": "success", "data": "Sample data"})

        self.main_function = main_function

    def get_start_callback_function(self):
        return self.start_callback_function

    def get_main_function(self):
        return self.main_function

    def get_function_definition(self):
        return self.function_definition



# Another example tool, e.g., for stock data
class WritingTool(BaseTool):
    def main(self):
        super().main()  # Initialize base functions

        # Override the main_function with specific logic for stock data
        async def tool_function(function_name, tool_call_id, args, llm, context, result_callback):
            print(context, args['text'], self.room_id)
            message_id = ''.join(random.choices(
                string.ascii_letters + string.digits, k=8))

            data = {"type": "function", "function_name": function_name,
                    "content": args['text'],  "message_id": message_id}

            await manager.broadcast_json(data, self.room_id)

            # Simulated stock data fetching logic
            await result_callback({"result": "success", "message": "text was notted successfully, please do this occassionaly"})
            # logger.debug(f"logged note for {
            #              function_name} with tool_call_id: {tool_call_id}")

        self.main_function = tool_function

        # Define function_definition specific to stock data
        self.function_definition = ChatCompletionToolParam(
            type="function",
            function={
                "name": "Jotting_tool",
                "description": "This tool is used when you need to write important sentences or formular on your students writing notepad to help them stay engaged and learn",
                "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "the text/formula you want to enter on the notepad",
                            },
                            # "format": {
                            #     "type": "string",
                            #     "enum": ["celsius", "fahrenheit"],
                            #     "description": "The temperature unit to use. Infer this from the users location.",
                            # },
                        },
                    "required": ["text"],
                },
            },
        )


video_participant_id = None


# class VisionTool(BaseTool):
#     def main(self):
#         super().main()

#         async def tool_function(function_name, tool_call_id, args, llm, context, result_callback):
#             print(context, self.room_id)
#             # message_id = ''.join(random.choices(
#             #     string.ascii_letters + string.digits, k=8))

#             # data = {"type": "function", "function_name": function_name,
#             #         "content": args['text'],  "message_id": message_id}

#             # await manager.broadcast_json(data, self.room_id)

#             logger.debug(f"!!! IN get_image {video_participant_id}, {args}")
#             question = args["question"]
#             await llm.request_image_frame(user_id=video_participant_id, text_content=question)

#         self.main_function = tool_function

#         # Define function_definition specific to stock data
#         self.function_definition = ChatCompletionToolParam(
#             type="function",
#             function={
#                 "name": "get_image",
#                 "description": "Get an image from the video stream.",
#                 "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "question": {
#                                 "type": "string",
#                                 "description": "The question to ask the AI to generate an image of",
#                             },
#                         },
#                     "required": ["question"],
#                 },
#             },
#         )
