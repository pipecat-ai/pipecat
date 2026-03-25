from dataclasses import dataclass
from typing import Any, Optional
import asyncio
from openai import AsyncOpenAI
import json

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


@dataclass
class LLMResponse:
    text: str
    raw: Optional[Any] = None


class MyLLMService(LLMService):

    def __init__(self, name: str = "MyLLM service", api_key: str = None, model: str = None):
          
          super().__init__(name=name)
          self.api_key = api_key
          self.model = model


    def create_tools(self):
        TOOLS = [
        {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current system time",
            "parameters": {}
            }
        },
        {
            "type": "function",
            "function": {
        "name": "get_current_weather",
        "description": "Get the current weather of the city that is provided",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA or only the city name"
                }
            },
            "required": ["location"]
            }
        } 
       },
       {
    "type": "function",
    "function": {
        "name": "calculate_distance",
        "description": "Calculate the distance between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "city1": {
                    "type": "string",
                    "description": "The first city"
                },
                "city2": {
                    "type": "string",
                    "description": "The second city"
                }
            },
            "required": ["city1", "city2"]
        }
    }
},
{
    "type": "function",
    "function": {
        "name": "calculate_amount",
        "description": "Calculate the amount for the distance",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "description": "The distance"
                }
            },
            "required": ["distance"]
        }
    }
}
        ]

        return TOOLS
    

    def tool_calls(self, tool_call, arguments):
        if tool_call == "get_current_time":
            result = self.get_current_time()
        elif tool_call == "get_current_weather":
            result = self.get_current_weather(arguments["location"])
        elif tool_call == "calculate_distance":
            result = self.calculate_distance(arguments["city1"], arguments["city2"])
        elif tool_call == "calculate_amount":
            result = self.calculate_amount(arguments["distance"])
        return result

    def get_current_weather(self, location):
        return f"The current weather of {location} is sunny"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # print("================================================")
        # print("into my LLM process_frame")
        # print("frame", frame, "direction", direction)

        """Process frames: run LLM on context frames and push all other frames through.

        StartFrame, EndFrame, CancelFrame, etc. must be pushed so the pipeline
        can complete startup (agent ready) and shutdown.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            # print("into if frame")
            try:
                # print("frame", frame)
                # print("direction", direction)
                params = self.get_llm_adapter().get_llm_invocation_params(frame.context)
                messages = params["messages"]
                response = await self.generate(messages)
                await self.push_frame(LLMFullResponseStartFrame())
                if response.text:
                    await self.push_frame(LLMTextFrame(text=response.text))
            except Exception as e:
                await self.push_error(error_msg=str(e) or type(e).__name__, exception=e)
            finally:
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            # print("into else frame")
            await self.push_frame(frame, direction)
          
    
    async def generate(self, messages: list) -> LLMResponse:
        client = AsyncOpenAI(api_key=self.api_key)

        messages = self._to_provider_messages(messages)

        if not messages:
            raise "No messages provided"

        await asyncio.sleep(0.5)
        last_message = messages[-1]["content"]

        if not last_message:
            return LLMResponse(text= "", raw = None)

        try:
            print("into MyLLM generate try block")

            while True:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.create_tools(),
                    tool_choice="auto"
                )
                
                print()
                print("================================================")
                print("response.choices[0].message", response.choices[0].message)

                tool_calls = response.choices[0].message.tool_calls
                
                print()
                print("================================================")
                print("tool_calls", tool_calls)

                if not tool_calls:
                    response_text = response.choices[0].message.content
                    print("if not tool_calls response", response_text)
                    return LLMResponse(
                        text=response_text,
                        raw=response
                    )

                assistant_message = response.choices[0].message

                print()
                print("================================================")
                print("assistant_message", assistant_message)

                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls,
                })

                print()
                print("================================================")
                print("messages", messages)

                for tc in tool_calls:
                    result = self.tool_calls(tc.function.name, json.loads(tc.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

                print()
                print("================================================")
                print("messages after tool calls", messages)

                print()
                print("================================================")
                print(f"Processed {len(tool_calls)} tool call(s), looping for next response...")

        except asyncio.TimeoutError:
            raise RuntimeError("LLM request timed out")
            
      
      # To get and provie the message from pipecat as json to LLM
      #The function is declared with _ to call with the class
    def _to_provider_messages(self, messages):
        return messages
      
    async def summarise_text(self,old_text):
        client = AsyncOpenAI(api_key=self.api_key)

        summary_prompt = (
            "Summarize the conversation briefly. Keep important facts only.\n\n"
            "Conversation:\n"
            f"{old_text}"
        )

        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
            {"role": "system", "content": summary_prompt}
            ]
        )
          

        summary = response.choices[0].message.content

        return summary


    def get_current_time(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def calculate_distance(self, city1, city2):
        return f"The distance between {city1} and {city2} is 100 miles"

    def calculate_amount(self, distance):
        return f"The amount for the distance {distance} is 1000rs"

    def book_cab(self, city1, city2):
        return f"The cab has been booked from {city1} to {city2}"

    def get_cab_details(self, city1, city2):
        return f"The cab details for the city {city1} to {city2} are as follows: The cab is a sedan, the color is blue, the license plate number is ABC123, the driver's name is John Doe, the driver's phone number is 1234567890"


        
        


          
          
    


