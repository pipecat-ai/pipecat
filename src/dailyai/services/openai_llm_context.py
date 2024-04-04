from typing import List

try:
    from openai._types import NOT_GIVEN, NotGiven

    from openai.types.chat import (
        ChatCompletionToolParam,
        ChatCompletionToolChoiceOptionParam,
        ChatCompletionMessageParam,
    )
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use OpenAI, you need to `pip install dailyai[openai]`. Also, set `OPENAI_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class OpenAILLMContext:

    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    ):
        self.messages: List[ChatCompletionMessageParam] = messages if messages else [
        ]
        self.tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self.tools: List[ChatCompletionToolParam] | NotGiven = tools

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        context = OpenAILLMContext()
        for message in messages:
            context.add_message({
                "content": message["content"],
                "role": message["role"],
                "name": message["name"] if "name" in message else message["role"]
            })
        return context

    # def __deepcopy__(self, memo):

    def add_message(self, message: ChatCompletionMessageParam):
        self.messages.append(message)

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        return self.messages

    def set_tool_choice(
        self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    ):
        self.tool_choice = tool_choice

    def set_tools(
            self,
            tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN):
        if tools != NOT_GIVEN and len(tools) == 0:
            tools = NOT_GIVEN

        self.tools = tools
