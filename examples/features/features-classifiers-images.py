#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ask a classifier about an image, such as a frame from the user's camera.

By default the questions go to Cloudflare's Clef; with ``--llm`` they go to
an OpenAI vision model through ``LLMClassifier`` instead.

Usage::

    CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_KEY=... python features-classifiers-images.py
    OPENAI_API_KEY=... python features-classifiers-images.py --llm gpt-4.1-mini
"""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ClassifierImage,
    ScoreQuestion,
    YesNoQuestion,
)
from pipecat.classifiers.cloudflare.clef.classifier import ClefClassifier
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.frames.frames import ImageRawFrame
from pipecat.services.openai.llm import OpenAILLMService

load_dotenv(override=True)

IMAGE_PATH = Path(__file__).parent.parent / "assets" / "cat.jpg"


async def main(classifier: BaseClassifier):
    # A camera frame arrives as raw pixels in an image frame, such as the
    # UserImageRawFrame a transport pushes. Here one is read from a file.
    pixels = Image.open(IMAGE_PATH).convert("RGB")
    frame = ImageRawFrame(image=pixels.tobytes(), size=pixels.size, format="RGB")
    image = await ClassifierImage.from_frame(frame)

    try:
        # The state says what the questions are about, and the image goes with it.
        results = await classifier.yes_no(
            "A frame from the user's camera.",
            {"person": YesNoQuestion(instructions="Is there a person in the frame?")},
            images=[image],
        )
        print(f"person: {results['person'].probability:.2f}")

        results = await classifier.choice(
            "A frame from the user's camera.",
            {
                "subject": ChoiceQuestion(
                    instructions="What is the user showing the camera?",
                    options={
                        "face": "their own face",
                        "pet": "an animal",
                        "document": "a document, card or screen",
                        "nothing": None,
                    },
                )
            },
            images=[image],
        )
        subject = results["subject"]
        print(f"subject: {subject.choice} ({subject.confidence:.2f})")

        results = await classifier.score(
            "A frame from the user's camera.",
            {
                "distance": ScoreQuestion(
                    instructions="How close to the camera is the subject?",
                    levels=["far away", "a few steps away", "close-up"],
                )
            },
            images=[image],
        )
        print(f"distance: {results['distance'].score:.2f} on a scale of 3")
    finally:
        await classifier.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", metavar="MODEL", help="use an OpenAI vision model instead of Clef")
    args = parser.parse_args()

    if args.llm:
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(model=args.llm),
        )
        asyncio.run(main(LLMClassifier(llm=llm)))
    else:
        classifier = ClefClassifier(
            account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
            api_key=os.environ["CLOUDFLARE_API_KEY"],
        )
        asyncio.run(main(classifier))
