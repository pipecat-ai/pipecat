#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import dataclasses

import pipecat.frames.protobufs.frames_pb2 as frame_protos

from pipecat.frames.frames import AudioRawFrame, Frame, TextFrame, TranscriptionFrame
from pipecat.serializers.base_serializer import FrameSerializer

from loguru import logger


class ProtobufFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        TextFrame: "text",
        AudioRawFrame: "audio",
        TranscriptionFrame: "transcription"
    }

    SERIALIZABLE_FIELDS = {v: k for k, v in SERIALIZABLE_TYPES.items()}

    def __init__(self):
        pass

    def serialize(self, frame: Frame) -> str | bytes | None:
        proto_frame = frame_protos.Frame()
        if type(frame) not in self.SERIALIZABLE_TYPES:
            raise ValueError(
                f"Frame type {type(frame)} is not serializable. You may need to add it to ProtobufFrameSerializer.SERIALIZABLE_FIELDS.")

        # ignoring linter errors; we check that type(frame) is in this dict above
        proto_optional_name = self.SERIALIZABLE_TYPES[type(frame)]  # type: ignore
        for field in dataclasses.fields(frame):  # type: ignore
            setattr(getattr(proto_frame, proto_optional_name), field.name,
                    getattr(frame, field.name))

        result = proto_frame.SerializeToString()
        return result

    def deserialize(self, data: str | bytes) -> Frame | None:
        """Returns a Frame object from a Frame protobuf. Used to convert frames
        passed over the wire as protobufs to Frame objects used in pipelines
        and frame processors.

        >>> serializer = ProtobufFrameSerializer()
        >>> serializer.deserialize(
        ...     serializer.serialize(AudioFrame(data=b'1234567890')))
        AudioFrame(data=b'1234567890')

        >>> serializer.deserialize(
        ...     serializer.serialize(TextFrame(text='hello world')))
        TextFrame(text='hello world')

        >>> serializer.deserialize(serializer.serialize(TranscriptionFrame(
        ...     text="Hello there!", participantId="123", timestamp="2021-01-01")))
        TranscriptionFrame(text='Hello there!', participantId='123', timestamp='2021-01-01')
        """

        proto = frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which not in self.SERIALIZABLE_FIELDS:
            logger.error("Unable to deserialize a valid frame")
            return None

        class_name = self.SERIALIZABLE_FIELDS[which]
        args = getattr(proto, which)
        args_dict = {}
        for field in proto.DESCRIPTOR.fields_by_name[which].message_type.fields:
            args_dict[field.name] = getattr(args, field.name)

        # Remove special fields if needed
        id = getattr(args, "id")
        name = getattr(args, "name")
        if not id:
            del args_dict["id"]
        if not name:
            del args_dict["name"]

        # Create the instance
        instance = class_name(**args_dict)

        # Set special fields
        if id:
            setattr(instance, "id", getattr(args, "id"))
        if name:
            setattr(instance, "name", getattr(args, "name"))

        return instance
