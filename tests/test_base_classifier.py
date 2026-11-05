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
    ClassifierAudio,
    ClassifierError,
    ClassifierImage,
    ClassifierQuestion,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.frames.frames import ImageRawFrame

QUESTION = {"answer": YesNoQuestion(instructions="is this red?")}


class _TextClassifier(BaseClassifier):
    """A classifier that can neither see nor hear: its _ask takes no audio."""

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


class _AudioClassifier(BaseClassifier):
    """A classifier that hears clips of 1 to 3 seconds and records what it was given."""

    def __init__(self, *, sample_rate: int | None = None):
        super().__init__()
        self._sample_rate = sample_rate
        self.clips: list[ClassifierAudio | None] = []
        self.states: list = []

    @property
    def supports_audio(self) -> bool:
        return True

    @property
    def min_audio_secs(self) -> float:
        return 1.0

    @property
    def max_audio_secs(self) -> float:
        return 3.0

    @property
    def audio_sample_rate(self) -> int | None:
        return self._sample_rate

    async def _ask(self, state, questions, images=(), audio=None):
        self.clips.append(audio)
        self.states.append(state)
        return {name: YesNoResult(probability=0.7) for name in questions}, None


def _jpeg() -> ClassifierImage:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return ClassifierImage(data=buffer.getvalue(), content_type="image/jpeg")


def _clip(secs: float, sample_rate: int = 16000) -> ClassifierAudio:
    return ClassifierAudio(audio=bytes(int(secs * sample_rate) * 2), sample_rate=sample_rate)


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


class TestAudio:
    @pytest.mark.asyncio
    async def test_a_classifier_that_cannot_hear_refuses_a_clip(self):
        classifier = _TextClassifier()

        assert not classifier.supports_audio
        with pytest.raises(ClassifierError, match="cannot hear audio"):
            await classifier.yes_no("a caller", QUESTION, audio=_clip(2))
        assert classifier.calls == 0

    @pytest.mark.asyncio
    async def test_a_clip_reaches_a_classifier_that_hears_with_its_state(self):
        classifier = _AudioClassifier()
        clip = _clip(2)

        results = await classifier.yes_no("a caller", QUESTION, audio=clip)

        assert results == {"answer": YesNoResult(probability=0.7)}
        assert classifier.clips == [clip]
        assert classifier.states == ["a caller"]

    @pytest.mark.asyncio
    async def test_a_classifier_that_hears_still_answers_without_a_clip(self):
        classifier = _AudioClassifier()

        await classifier.ask("a caller", QUESTION)

        assert classifier.clips == [None]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("secs", [0.5, 3.5])
    async def test_a_clip_of_a_length_the_classifier_doesnt_accept_is_refused(self, secs):
        classifier = _AudioClassifier()

        with pytest.raises(ClassifierError, match="of audio"):
            await classifier.ask("a caller", QUESTION, audio=_clip(secs))
        assert classifier.clips == []

    @pytest.mark.asyncio
    async def test_a_clip_is_resampled_to_the_classifiers_sample_rate(self):
        classifier = _AudioClassifier(sample_rate=16000)

        await classifier.ask("a caller", QUESTION, audio=_clip(2, sample_rate=8000))

        [clip] = classifier.clips
        assert clip.sample_rate == 16000
        assert clip.duration_secs == pytest.approx(2, abs=0.05)

    @pytest.mark.asyncio
    async def test_a_clip_keeps_its_own_rate_when_the_classifier_takes_any(self):
        classifier = _AudioClassifier()
        clip = _clip(2, sample_rate=8000)

        await classifier.ask("a caller", QUESTION, audio=clip)

        assert classifier.clips == [clip]

    def test_a_clip_knows_how_long_it_is(self):
        assert _clip(2.5).duration_secs == 2.5

    def test_the_audio_bytes_stay_out_of_the_repr(self):
        assert "audio=" not in repr(_clip(1))


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
