import asyncio
import os

from google import genai
from google.genai.types import (
    Content,
    LiveConnectConfig,
    Modality,
    Part,
)


async def run_gemini():
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    GOOGLE_CLOUD_LOCATION = "us-central1"
    MODEL_ID = "gemini-2.0-flash-live-preview-04-09"

    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

    async with client.aio.live.connect(
        model=MODEL_ID,
        config=LiveConnectConfig(response_modalities=[Modality.TEXT]),
    ) as session:
        while True:
            turns = []

            text_input = input("Enter your message: ")
            print("> ", text_input, "\n")
            if text_input == "change_james":  # Update system instruction
                turns.append(
                    Content(
                        role="system",
                        parts=[
                            Part(text="Your name is James Bond. Help the user with their queries.")
                        ],
                    )
                )
            if text_input == "change_lisa":  # Update system instruction again
                turns.append(
                    Content(
                        role="system",
                        parts=[
                            Part(text="Your name is Lisa Rome. Help the user with their queries.")
                        ],
                    )
                )
            turns.append(Content(role="user", parts=[Part(text=text_input)]))
            print(f"current turns: {turns}")
            await session.send_client_content(turns=turns)

            response = []

            async for message in session.receive():
                if message.text:
                    response.append(message.text)

            print("".join(response))


asyncio.run(run_gemini())
