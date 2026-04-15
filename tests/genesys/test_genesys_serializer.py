#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Genesys AudioHook serializer."""

import json

import pytest

from pipecat.frames.frames import InputDTMFFrame, OutputTransportMessageUrgentFrame
from pipecat.serializers.genesys import AudioHookChannel, GenesysAudioHookSerializer


class TestGenesysAudioHookSerializer:
    """Tests for GenesysAudioHookSerializer."""

    # ==================== Initialization Tests ====================

    def test_create_serializer_default_params(self):
        """Test creating serializer with default parameters."""
        serializer = GenesysAudioHookSerializer()

        # session_id is auto-generated as UUID
        assert serializer.session_id != ""
        assert len(serializer.session_id) == 36  # UUID format
        assert serializer.is_open is False
        assert serializer.is_paused is False

    def test_create_serializer_with_custom_params(self):
        """Test creating serializer with custom parameters."""
        params = GenesysAudioHookSerializer.InputParams(
            channel=AudioHookChannel.BOTH,
            sample_rate=16000,
            supported_languages=["es-ES", "en-US"],
            selected_language="es-ES",
            start_paused=True,
        )
        serializer = GenesysAudioHookSerializer(params=params)

        assert serializer.session_id != ""

    # ==================== Response Creation Tests ====================

    def test_create_opened_response(self):
        """Test creating an opened response message."""
        serializer = GenesysAudioHookSerializer()

        msg = serializer.create_opened_response()

        assert msg["type"] == "opened"
        assert msg["version"] == "2"
        assert msg["id"] == serializer.session_id
        assert "parameters" in msg
        assert serializer.is_open is True

    def test_create_opened_response_with_languages(self):
        """Test creating an opened response with language options."""
        serializer = GenesysAudioHookSerializer()

        msg = serializer.create_opened_response(
            supported_languages=["es", "en", "fr"],
            selected_language="es",
        )

        assert msg["parameters"]["supportedLanguages"] == ["es", "en", "fr"]
        assert msg["parameters"]["selectedLanguage"] == "es"

    def test_create_pong_response(self):
        """Test creating a pong response message."""
        serializer = GenesysAudioHookSerializer()

        msg = serializer.create_pong_response()

        assert msg["type"] == "pong"
        assert msg["id"] == serializer.session_id

    def test_create_closed_response(self):
        """Test creating a closed response message."""
        serializer = GenesysAudioHookSerializer()
        serializer._is_open = True

        msg = serializer.create_closed_response()

        assert msg["type"] == "closed"
        assert serializer.is_open is False
        assert "parameters" not in msg  # No parameters when no output_variables

    def test_create_closed_response_with_output_variables(self):
        """Test creating a closed response with custom output variables."""
        serializer = GenesysAudioHookSerializer()
        serializer._is_open = True

        msg = serializer.create_closed_response(
            output_variables={
                "intent": "billing_inquiry",
                "customer_verified": True,
                "summary": "Customer asked about their bill",
            }
        )

        assert msg["type"] == "closed"
        assert msg["parameters"]["outputVariables"]["intent"] == "billing_inquiry"
        assert msg["parameters"]["outputVariables"]["customer_verified"] is True
        assert msg["parameters"]["outputVariables"]["summary"] == "Customer asked about their bill"

    def test_create_resumed_response(self):
        """Test creating a resumed response message."""
        serializer = GenesysAudioHookSerializer()
        serializer._is_paused = True

        msg = serializer.create_resumed_response()

        assert msg["type"] == "resumed"
        assert serializer.is_paused is False

    def test_create_disconnect_message(self):
        """Test creating a disconnect message."""
        serializer = GenesysAudioHookSerializer()

        msg = serializer.create_disconnect_message(
            reason="completed",
            action="transfer",
        )

        assert msg["type"] == "disconnect"
        assert msg["parameters"]["reason"] == "completed"
        assert msg["parameters"]["outputVariables"]["action"] == "transfer"

    def test_create_disconnect_message_with_output_variables(self):
        """Test creating a disconnect message with custom output variables."""
        serializer = GenesysAudioHookSerializer()

        msg = serializer.create_disconnect_message(
            reason="completed",
            action="finished",
            output_variables={"result": "success", "code": "123"},
        )

        assert msg["parameters"]["outputVariables"]["result"] == "success"
        assert msg["parameters"]["outputVariables"]["code"] == "123"

    def test_create_error_message(self):
        """Test creating an error message."""
        serializer = GenesysAudioHookSerializer()

        msg = serializer.create_error_message(
            code=500,
            message="Internal error",
            retryable=True,
        )

        assert msg["type"] == "error"
        assert msg["parameters"]["code"] == 500
        assert msg["parameters"]["message"] == "Internal error"
        assert msg["parameters"]["retryable"] is True

    # ==================== Message Handling Tests ====================

    @pytest.mark.asyncio
    async def test_handle_open_message(self, sample_open_message):
        """Test handling an open message returns opened frame."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_open_message))

        # Now returns OutputTransportMessageUrgentFrame with opened response
        assert isinstance(result, OutputTransportMessageUrgentFrame)
        assert result.message["type"] == "opened"
        assert serializer.session_id == "test-session-123"
        assert serializer.conversation_id == "conv-456"

    @pytest.mark.asyncio
    async def test_handle_open_message_extracts_participant(self, sample_open_message):
        """Test that open message extracts participant info."""
        serializer = GenesysAudioHookSerializer()

        await serializer.deserialize(json.dumps(sample_open_message))

        assert serializer.participant is not None
        assert serializer.participant["ani"] == "+1234567890"
        assert serializer.participant["dnis"] == "+0987654321"

    @pytest.mark.asyncio
    async def test_handle_open_message_uses_params(self, sample_open_message):
        """Test that open message uses InputParams for response."""
        params = GenesysAudioHookSerializer.InputParams(
            supported_languages=["es-ES", "en-US"],
            selected_language="es-ES",
            start_paused=True,
        )
        serializer = GenesysAudioHookSerializer(params=params)

        result = await serializer.deserialize(json.dumps(sample_open_message))

        assert isinstance(result, OutputTransportMessageUrgentFrame)
        assert result.message["parameters"]["supportedLanguages"] == ["es-ES", "en-US"]
        assert result.message["parameters"]["selectedLanguage"] == "es-ES"
        assert result.message["parameters"]["startPaused"] is True

    @pytest.mark.asyncio
    async def test_handle_open_message_extracts_input_variables(
        self, sample_open_message_with_input_variables
    ):
        """Test that open message extracts inputVariables from Genesys."""
        serializer = GenesysAudioHookSerializer()

        await serializer.deserialize(json.dumps(sample_open_message_with_input_variables))

        assert serializer.input_variables is not None
        assert serializer.input_variables["customer_id"] == "cust-789"
        assert serializer.input_variables["queue_name"] == "billing"
        assert serializer.input_variables["priority"] == "high"
        assert serializer.input_variables["language"] == "es-ES"

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, sample_ping_message):
        """Test handling a ping message returns pong frame."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_ping_message))

        assert isinstance(result, OutputTransportMessageUrgentFrame)
        assert result.message["type"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_close_message(self, sample_close_message):
        """Test handling a close message returns closed frame."""
        serializer = GenesysAudioHookSerializer()
        serializer._is_open = True

        result = await serializer.deserialize(json.dumps(sample_close_message))

        assert isinstance(result, OutputTransportMessageUrgentFrame)
        assert result.message["type"] == "closed"
        assert serializer.is_open is False

    @pytest.mark.asyncio
    async def test_handle_close_message_includes_output_variables(self, sample_close_message):
        """Test that close response includes output variables when set."""
        serializer = GenesysAudioHookSerializer()
        serializer._is_open = True

        # Set output variables before close
        serializer.set_output_variables(
            {"intent": "support", "resolved": True, "transfer_to": "agent_queue"}
        )

        result = await serializer.deserialize(json.dumps(sample_close_message))

        assert isinstance(result, OutputTransportMessageUrgentFrame)
        assert result.message["type"] == "closed"
        assert result.message["parameters"]["outputVariables"]["intent"] == "support"
        assert result.message["parameters"]["outputVariables"]["resolved"] is True
        assert result.message["parameters"]["outputVariables"]["transfer_to"] == "agent_queue"

    # ==================== Output Variables Tests ====================

    def test_set_output_variables(self):
        """Test setting output variables."""
        serializer = GenesysAudioHookSerializer()

        assert serializer.output_variables is None

        serializer.set_output_variables({"intent": "billing", "score": 0.95})

        assert serializer.output_variables is not None
        assert serializer.output_variables["intent"] == "billing"
        assert serializer.output_variables["score"] == 0.95

    def test_set_output_variables_overwrites(self):
        """Test that setting output variables overwrites previous values."""
        serializer = GenesysAudioHookSerializer()

        serializer.set_output_variables({"first": "value"})
        serializer.set_output_variables({"second": "value"})

        assert "first" not in serializer.output_variables
        assert serializer.output_variables["second"] == "value"

    @pytest.mark.asyncio
    async def test_handle_pause_message(self, sample_pause_message):
        """Test handling a pause message."""
        serializer = GenesysAudioHookSerializer()
        serializer._is_open = True

        result = await serializer.deserialize(json.dumps(sample_pause_message))

        assert result is None  # Pause is handled internally
        assert serializer.is_paused is True

    @pytest.mark.asyncio
    async def test_handle_update_message(self, sample_update_message):
        """Test handling an update message."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_update_message))

        assert result is None  # Update is handled internally
        assert serializer.participant is not None
        assert serializer.participant["name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_handle_error_message(self, sample_error_message):
        """Test handling an error message."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_error_message))

        assert result is None  # Error is logged but returns None

    # ==================== DTMF Tests ====================

    @pytest.mark.asyncio
    async def test_handle_dtmf_digit(self, sample_dtmf_message):
        """Test handling a DTMF digit message."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_dtmf_message))

        assert isinstance(result, InputDTMFFrame)
        assert result.button.value == "5"

    @pytest.mark.asyncio
    async def test_handle_dtmf_star(self, sample_dtmf_star_message):
        """Test handling a DTMF star (*) message."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_dtmf_star_message))

        assert isinstance(result, InputDTMFFrame)
        assert result.button.value == "*"

    @pytest.mark.asyncio
    async def test_handle_dtmf_hash(self, sample_dtmf_hash_message):
        """Test handling a DTMF hash (#) message."""
        serializer = GenesysAudioHookSerializer()

        result = await serializer.deserialize(json.dumps(sample_dtmf_hash_message))

        assert isinstance(result, InputDTMFFrame)
        assert result.button.value == "#"

    @pytest.mark.asyncio
    async def test_handle_dtmf_empty_digit(self):
        """Test handling a DTMF message without digit."""
        serializer = GenesysAudioHookSerializer()

        dtmf_msg = {
            "version": "2",
            "type": "dtmf",
            "seq": 6,
            "id": "test-session-123",
            "parameters": {},
        }

        result = await serializer.deserialize(json.dumps(dtmf_msg))

        assert result is None  # No digit provided

    # ==================== Sequence Number Tests ====================

    def test_sequence_numbers_increment(self):
        """Test that sequence numbers increment correctly."""
        serializer = GenesysAudioHookSerializer()

        response1 = serializer.create_pong_response()
        response2 = serializer.create_pong_response()
        response3 = serializer.create_pong_response()

        assert response1["seq"] == 1
        assert response2["seq"] == 2
        assert response3["seq"] == 3
