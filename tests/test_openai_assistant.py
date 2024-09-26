import asyncio
import os
from pipecat.services.openai import OpenAILLMService, BaseOpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from dotenv import load_dotenv

load_dotenv()

async def test_assistant_response():
    # Set up the OpenAILLMService with the assistant ID
    service = OpenAILLMService(
        model="gpt-3.5-turbo",  # This model might not be used when using an assistant
        params=BaseOpenAILLMService.InputParams(
            assistant_id=os.environ.get("OPENAI_ASSISTANT_ID")
        ),
        api_key=os.environ.get("OPENAI_API_KEY")  # Make sure to set this environment variable
    )

    # Create a context with the test message
    context = OpenAILLMContext()
    context.add_message({"role": "user", "content": "tell me about your brand"})

    # Get the chat completions
    response_stream = await service.get_chat_completions(context, context.get_messages())

    # Process the response
    full_response = ""
    async for chunk in response_stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                print(content, end='', flush=True)  # Print the response as it comes in

    print("\n\nFull response:")
    print(full_response)

if __name__ == "__main__":
    asyncio.run(test_assistant_response())