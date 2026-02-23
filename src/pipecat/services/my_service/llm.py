from dataclasses import dataclass
from typing import Any, Optional
import asyncio
from openai import AsyncOpenAI

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
        }
        ]

        return TOOLS
    

    def tool_calls(self, tool_call):
        if tool_call == "get_current_time":
            result = self.get_current_time()
            return result

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
        print("================================================")
        print("into my LLM generate")
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
            response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.create_tools(), 
            tool_choice="auto"
            )

            tool_call = response.choices[0].message.tool_calls

            if tool_call:
                tool_call_id = response.choices[0].message.tool_calls[0].id
                assistant_message = response.choices[0].message

                messages.append({
                    "role": "assistant",
                     "content": assistant_message.content,
                     "tool_calls": assistant_message.tool_calls,
                    })


                result = self.tool_calls(tool_call[0].function.name)

                messages.append({
                    "role": "tool",
                     "tool_call_id": tool_call_id,
                     "content": result
                })

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )

                response_text = response.choices[0].message.content

                print("before return LLMResponse")

                return LLMResponse(
                text=response_text,
                raw=response
                )
                
            else:
                response_text = response.choices[0].message.content
                print("response_text", response_text)
                print("================================================")
                return LLMResponse(
                text=response_text,
                raw=response
                )
            
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



        
        


          
          
    


