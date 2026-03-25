import asyncio
import dotenv
import os
from dotenv import load_dotenv

load_dotenv()

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.my_service.llm import MyLLMService
from pipecat.services.my_service.tts import MyTTSService

async def check_conversation_length(conversation):
    SUMMARY_TAG = "[SUMMARY]"
    count = 0
    existing_summary = []

    if len(conversation) > 10:
        # print("into if len of convo greater than 5")
        system_message = []
        for convo in conversation:
            if convo.get("role") == "system":
                if convo["content"].startswith(SUMMARY_TAG):
                    existing_summary = convo
                    count+= 1
                    if count > 2:
                        break
                else:
                    system_message.append(convo)
                    count+= 1
                    if count > 2:
                        break
                    

        NON_SYSTEM = [m for m in conversation if m.get("role") != "system"]

        old_messages = NON_SYSTEM[:-5]

        new_messages = NON_SYSTEM[-5:]

        old_text = messages_to_text(old_messages, existing_summary if existing_summary else None)
        print("================================================")
        print("existing_summary", existing_summary)


        llm = MyLLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")

        summary = await llm.summarise_text(old_text)
        # print("================================================")
        # print("summary", summary)

        summary = [{"role": "system",  "content": f"{SUMMARY_TAG}\n{summary}"}]

        conversation = system_message + summary + new_messages
            
        return conversation


def messages_to_text(messages, existing_summary=None):
    parts = []

    if existing_summary:
        parts.append(f"Summary so far:\n{existing_summary}")

    for m in messages:
        parts.append(f"{m.get('role')}: {m.get('content')}")

    return "\n".join(parts)



async def test_tool_calls():
    llm = MyLLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")
    messages = [{
        "role": "system", "content": "You are a helpful assistant."
    }]

    print("Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = await llm.generate(messages)
        final_text = response.text

        messages.append({"role": "assistant", "content": final_text})

        print(f"\nAssistant: {final_text}\n")


if __name__ == "__main__":
    asyncio.run(test_tool_calls())