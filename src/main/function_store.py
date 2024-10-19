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
            # logger.debug(f"Starting {
            #              self.function_definition['function']['name']} with function_name: {function_name} and Room_id: {self.room_id}")

        self.start_callback_function = start_callback_function

        # Define the async function for main_function
        async def main_function(function_name, tool_call_id, args, llm, context, result_callback):
            # Placeholder logic; to be overridden in child classes
            # logger.debug(f"Fetching data for {
            #              function_name} with tool_call_id: {tool_call_id}")
            await result_callback({"status": "success", "data": "Sample data"})

        self.main_function = main_function

    def get_start_callback_function(self):
        return self.start_callback_function

    def get_main_function(self):
        return self.main_function

    def get_function_definition(self):
        return self.function_definition


# Example tool for weather data, inheriting from BaseTool
class WeatherTool(BaseTool):
    def main(self):
        super().main()  # Initialize base functions

        # Override the main_function with specific logic for weather
        async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
            # Simulated weather fetching logic
            await result_callback({"conditions": "sunny", "temperature": "75F"})
            # logger.debug(f"Fetched weather data for {
            #              function_name} with tool_call_id: {tool_call_id}")

        self.main_function = fetch_weather_from_api

        # Define function_definition specific to weather
        self.function_definition = ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                    "required": ["location", "format"],
                },
            },
        )


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


class VisionTool(BaseTool):
    def main(self):
        super().main()  # Initialize base functions

        # Override the main_function with specific logic for stock data
        async def tool_function(function_name, tool_call_id, args, llm, context, result_callback):
            print(context, args['text'], self.room_id)
            # message_id = ''.join(random.choices(
            #     string.ascii_letters + string.digits, k=8))

            # data = {"type": "function", "function_name": function_name,
            #         "content": args['text'],  "message_id": message_id}

            # await manager.broadcast_json(data, self.room_id)

            logger.debug(f"!!! IN get_image {video_participant_id}, {arguments}")
            question = args["question"]
            await llm.request_image_frame(user_id=video_participant_id, text_content=question)

        self.main_function = tool_function

        # Define function_definition specific to stock data
        self.function_definition = ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_image",
                    "description": "Get an image from the video stream.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the AI to generate an image of",
                            },
                        },
                        "required": ["question"],
                    },
                },
            )


# # Test function to run async functions of tools
# async def test_tool(tool_instance):
#     # Mock arguments for testing
#     async def mock_result_callback(result):
#         logger.debug(f"Result from callback: {result}")

#     function_name = tool_instance.get_function_definition()[
#         'function']['name']  # Adjusted line
#     tool_call_id = "123"
#     args = {}
#     llm = None  # Replace with actual LLM object
#     context = None  # Replace with actual context object

#     # Call start_callback function
#     start_callback = tool_instance.get_start_callback_function()
#     await start_callback(function_name, llm, context)

#     # Call main_function
#     main_func = tool_instance.get_main_function()
#     await main_func(function_name, tool_call_id, args, llm, context, result_callback=mock_result_callback)


# if __name__ == "__main__":
#     # Initialize tools
#     weather_tool = WeatherTool()
#     weather_tool.main()

#     stock_tool = WritingTool()
#     stock_tool.main()

#     # Run test cases for tools
#     asyncio.run(test_tool(weather_tool))
#     asyncio.run(test_tool(stock_tool))

#     # Print function definitions
#     print(f"Weather Tool definition: {weather_tool.get_function_definition()}")
#     print(f"Stock Tool definition: {stock_tool.get_function_definition()}")
