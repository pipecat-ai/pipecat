#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
from collections.abc import Mapping, Sequence

import pytest
from PIL import Image

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ClassifierError,
    ClassifierImage,
    ClassifierQuestion,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.frames.frames import ImageRawFrame

QUESTION = {"answer": YesNoQuestion(instructions="is this red?")}


class _TextClassifier(BaseClassifier):
    """A classifier that cannot see images: its _ask takes no images."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def _ask(self, state, questions, images=()):
        self.calls += 1
        return {name: YesNoResult(probability=0.5) for name in questions}, None


class _ImageClassifier(BaseClassifier):
    """A classifier that sees images and records what it was given."""

    def __init__(self):
        super().__init__()
        self.images: list[Sequence[ClassifierImage]] = []

    @property
    def supports_images(self) -> bool:
        return True

    async def _ask(
        self,
        state,
        questions: Mapping[str, ClassifierQuestion],
        images: Sequence[ClassifierImage] = (),
    ):
        self.images.append(images)
        return {name: YesNoResult(probability=0.9) for name in questions}, None


def _jpeg() -> ClassifierImage:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return ClassifierImage(data=buffer.getvalue(), content_type="image/jpeg")


class TestImages:
    @pytest.mark.asyncio
    async def test_a_classifier_that_cannot_see_images_refuses_them(self):
        classifier = _TextClassifier()

        assert not classifier.supports_images
        with pytest.raises(ClassifierError, match="cannot see images"):
            await classifier.yes_no("a frame", QUESTION, images=[_jpeg()])
        assert classifier.calls == 0

    @pytest.mark.asyncio
    async def test_a_classifier_that_cannot_see_images_still_answers_text(self):
        classifier = _TextClassifier()

        results = await classifier.yes_no("hello", QUESTION)

        assert results["answer"].probability == 0.5

    @pytest.mark.asyncio
    async def test_images_reach_a_classifier_that_sees_them(self):
        classifier = _ImageClassifier()
        image = _jpeg()

        await classifier.yes_no("a frame", QUESTION, images=[image])
        await classifier.ask("a frame", QUESTION, images=(image, image))
        await classifier.yes_no("no frame", QUESTION)

        assert classifier.images == [[image], [image, image], []]

    def test_the_image_bytes_stay_out_of_the_repr(self):
        assert "data" not in repr(_jpeg())


class TestImageType:
    @pytest.mark.parametrize(
        "pillow_format,content_type",
        [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
    )
    def test_the_type_is_read_from_the_bytes(self, pillow_format, content_type):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, format=pillow_format)

        assert ClassifierImage(data=buffer.getvalue()).content_type == content_type

    def test_a_given_type_is_kept(self):
        assert ClassifierImage(data=b"png bytes", content_type="image/png").content_type == (
            "image/png"
        )

    def test_another_image_type_is_an_error(self):
        gif = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(gif, format="GIF")

        with pytest.raises(ValueError, match="PNG, JPEG or WebP, got image/gif"):
            ClassifierImage(data=gif.getvalue())

    def test_bytes_that_are_not_an_image_are_an_error(self):
        with pytest.raises(ValueError, match="not an image"):
            ClassifierImage(data=b"not an image")


class TestImageFromFrame:
    @pytest.mark.asyncio
    async def test_raw_pixels_are_encoded_as_a_jpeg(self):
        pixels = Image.new("RGB", (16, 9), "red")
        frame = ImageRawFrame(image=pixels.tobytes(), size=(16, 9), format="RGB")

        image = await ClassifierImage.from_frame(frame)

        assert image.content_type == "image/jpeg"
        decoded = Image.open(io.BytesIO(image.data))
        assert (decoded.format, decoded.size) == ("JPEG", (16, 9))

    @pytest.mark.asyncio
    async def test_pixels_with_alpha_are_encoded_without_it(self):
        pixels = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        frame = ImageRawFrame(image=pixels.tobytes(), size=(4, 4), format="RGBA")

        image = await ClassifierImage.from_frame(frame)

        assert Image.open(io.BytesIO(image.data)).mode == "RGB"

    @pytest.mark.asyncio
    async def test_an_encoded_frame_keeps_its_bytes(self):
        png = io.BytesIO()
        Image.new("RGB", (4, 4), "blue").save(png, format="PNG")
        frame = ImageRawFrame(image=png.getvalue(), size=(4, 4), format="image/png")

        image = await ClassifierImage.from_frame(frame)

        assert (image.content_type, image.data) == ("image/png", png.getvalue())

    @pytest.mark.asyncio
    async def test_an_encoded_frame_of_another_type_is_an_error(self):
        frame = ImageRawFrame(image=b"gif bytes", size=(4, 4), format="image/gif")

        with pytest.raises(ValueError):
            await ClassifierImage.from_frame(frame)
