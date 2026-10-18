#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for pipecat.utils.security.redaction.redact_secrets."""

import unittest

from pipecat.frames.frames import ErrorFrame
from pipecat.utils.security.redaction import redact_secrets


class TestRedactSecrets(unittest.TestCase):
    def test_masks_a_credential_in_a_printed_header_dict(self):
        message = (
            "Sarvam API error: headers: {'api-subscription-key': 'sk_live_abc123', "
            "'User-Agent': 'sarvamai/0.1.28'}, status_code: 429"
        )
        self.assertEqual(
            redact_secrets(message),
            "Sarvam API error: headers: {'api-subscription-key': '[REDACTED]', "
            "'User-Agent': 'sarvamai/0.1.28'}, status_code: 429",
        )

    def test_masks_a_presigned_urls_credentials_and_keeps_the_rest(self):
        message = (
            "Timed out fetching file from URL: 'https://bucket.s3.amazonaws.com/doc.pdf"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAEXAMPLE%2F20261014"
            "&X-Amz-Signature=deadbeef&X-Amz-Security-Token=FQoGZXIvYXdz'"
        )
        self.assertEqual(
            redact_secrets(message),
            "Timed out fetching file from URL: 'https://bucket.s3.amazonaws.com/doc.pdf"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=[REDACTED]"
            "&X-Amz-Signature=[REDACTED]&X-Amz-Security-Token=[REDACTED]'",
        )

    def test_masks_a_key_in_a_query_string(self):
        self.assertEqual(
            redact_secrets("wss://api.example.com/ws?api_key=SECRET&version=v1 isn't a valid URI"),
            "wss://api.example.com/ws?api_key=[REDACTED]&version=v1 isn't a valid URI",
        )

    def test_keeps_an_auth_scheme_and_masks_its_credential(self):
        self.assertEqual(
            redact_secrets("Authorization: Bearer abc123"), "Authorization: Bearer [REDACTED]"
        )

    def test_masks_a_camel_case_credential_name(self):
        self.assertEqual(
            redact_secrets("apiKey=foo, accessToken: bar"),
            "apiKey=[REDACTED], accessToken: [REDACTED]",
        )

    def test_masks_a_urls_password(self):
        self.assertEqual(
            redact_secrets("Could not connect to redis://default:hunter2@redis:6379/0"),
            "Could not connect to redis://default:[REDACTED]@redis:6379/0",
        )

    def test_masks_tokens_recognizable_by_format(self):
        self.assertEqual(
            redact_secrets(
                "rejected sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA for AKIAIOSFODNN7EXAMPLE "
                "joining https://acme.daily.co/room?t=eyJhbGciOiJIUzI1NiJ9.eyJyIjoicm9vbSJ9."
                "c2lnbmF0dXJl"
            ),
            "rejected [REDACTED] for [REDACTED] joining https://acme.daily.co/room?t=[REDACTED]",
        )

    def test_masks_each_provider_token_format(self):
        for token in [
            "sk-proj-" + "a" * 74 + "T3BlbkFJ" + "b" * 74,  # OpenAI
            "ABIA" + "A" * 16,  # AWS
            "ABSK" + "a" * 120 + "==",  # Amazon Bedrock, long-lived
            "bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29t" + "a" * 40,  # short-lived
            "AIza" + "a" * 35,  # Google
            "github_pat_" + "a" * 82,  # GitHub
            "xapp-1-A0123456789-0123456789012-" + "a" * 64,  # Slack
            "hf_" + "a" * 34,  # Hugging Face
            "pplx-" + "a" * 48,  # Perplexity
            "SK" + "0" * 32,  # Twilio
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NSJ9.",  # unsigned JWT
        ]:
            with self.subTest(token=token[:20]):
                self.assertEqual(
                    redact_secrets(f"failed with {token} here"), "failed with [REDACTED] here"
                )

    def test_masks_a_slack_webhooks_secret_path(self):
        self.assertEqual(
            redact_secrets("posting to https://hooks.slack.com/services/" + "a" * 44),
            "posting to https://hooks.slack.com/services/[REDACTED]",
        )

    def test_masks_a_multiline_private_key(self):
        pem = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----"
        self.assertEqual(
            redact_secrets(f'{{"private_key": "{pem}", "client_email": "svc@x"}}'),
            '{"private_key": "[REDACTED]", "client_email": "svc@x"}',
        )

    def test_masks_a_value_after_any_assignment_separator(self):
        self.assertEqual(
            redact_secrets("apiKey := abc123, 'secret' => 'def456', creds=ghi789"),
            "apiKey := [REDACTED], 'secret' => '[REDACTED]', creds=[REDACTED]",
        )

    def test_masks_an_aws_signature_authorization_header(self):
        self.assertEqual(
            redact_secrets(
                "Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20261014/"
                "us-east-1/s3/aws4_request, SignedHeaders=host, Signature=fe5f80f77d5fa3be"
            ),
            "Authorization: AWS4-HMAC-SHA256 [REDACTED], SignedHeaders=host, Signature=[REDACTED]",
        )

    def test_masks_a_quoted_value_cut_short_by_truncation(self):
        self.assertEqual(
            redact_secrets("PGMQueue(host='db', password='hunter2..."),
            "PGMQueue(host='db', password='[REDACTED]",
        )

    def test_masks_a_resumable_upload_session_id(self):
        self.assertEqual(
            redact_secrets("https://example.googleapis.com/upload/files?upload_id=ABC123"),
            "https://example.googleapis.com/upload/files?upload_id=[REDACTED]",
        )

    def test_leaves_a_message_without_credentials_as_written(self):
        for message in [
            "wss://api.elevenlabs.io/v1/speech-to-text?model_id=scribe&keyterms=pipecat",
            "Error code: 400 - {'error': {'message': 'max_tokens: 100 is too low'}}",
            'Cartesia API error (status 400): {"error": "bad voice", "voice_id": "abc"}',
            "Error: timeout at 12:30",
            "",
        ]:
            with self.subTest(message=message):
                self.assertEqual(redact_secrets(message), message)

    def test_redacting_twice_reads_the_same_as_redacting_once(self):
        message = (
            "Failed to fetch file from URL 'https://x/a?sig=abc&se=2026': 403, "
            "url='https://x/a?sig=abc&se=2026' Authorization: Bearer abc"
        )
        once = redact_secrets(message)
        self.assertEqual(redact_secrets(once), once)


class TestErrorFrameRedaction(unittest.TestCase):
    def test_an_error_frame_masks_the_credentials_its_message_quotes(self):
        frame = ErrorFrame(error="Sarvam API error: headers: {'api-subscription-key': 'k'}")
        self.assertEqual(
            frame.error, "Sarvam API error: headers: {'api-subscription-key': '[REDACTED]'}"
        )


if __name__ == "__main__":
    unittest.main()
