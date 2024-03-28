import dataclasses
from typing import Text
from dailyai.pipeline.frames import AudioFrame, Frame, TextFrame, TranscriptionFrame
import dailyai.pipeline.protobufs.frames_pb2 as frame_protos
from dailyai.serializers.abstract_frame_serializer import FrameSerializer


class ProtobufFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        TextFrame: "text",
        AudioFrame: "audio",
        TranscriptionFrame: "transcription"
    }

    SERIALIZABLE_FIELDS = {v: k for k, v in SERIALIZABLE_TYPES.items()}

    def __init__(self):
        pass

    def serialize(self, frame: Frame) -> bytes:
        proto_frame = frame_protos.Frame()
        if type(frame) not in self.SERIALIZABLE_TYPES:
            raise ValueError(
                f"Frame type {type(frame)} is not serializable. You may need to add it to ProtobufFrameSerializer.SERIALIZABLE_FIELDS.")

        # ignoring linter errors; we check that type(frame) is in this dict above
        proto_optional_name = self.SERIALIZABLE_TYPES[type(frame)]  # type: ignore
        for field in dataclasses.fields(frame):  # type: ignore
            setattr(getattr(proto_frame, proto_optional_name), field.name,
                    getattr(frame, field.name))

        return proto_frame.SerializeToString()

    def deserialize(self, data: bytes) -> Frame:
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
            raise ValueError(
                "Proto does not contain a valid frame. You may need to add a new case to ProtobufFrameSerializer.deserialize.")

        class_name = self.SERIALIZABLE_FIELDS[which]
        args = getattr(proto, which)
        args_dict = {}
        for field in proto.DESCRIPTOR.fields_by_name[which].message_type.fields:
            args_dict[field.name] = getattr(args, field.name)
        return class_name(**args_dict)
