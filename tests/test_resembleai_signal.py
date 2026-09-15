#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import Any

from pipecat.services.resembleai.signal import ResembleSignal


class _FakeSignalTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def score_text(self, text: str, *, request_timeout: float) -> dict[str, Any]:
        self.calls.append({"op": "score_text", "text": text, "request_timeout": request_timeout})
        return _signal_item(input_modality="text")

    async def score_file(
        self,
        file: bytes,
        *,
        filename: str,
        media_type: str | None,
        content_type: str | None,
        request_timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "op": "score_file",
                "file": file,
                "filename": filename,
                "media_type": media_type,
                "content_type": content_type,
                "request_timeout": request_timeout,
            }
        )
        return _signal_item(input_modality="audio")

    async def list_submissions(
        self,
        *,
        page: int,
        per_page: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "op": "list_submissions",
                "page": page,
                "per_page": per_page,
                "request_timeout": request_timeout,
            }
        )
        return {"success": True, "items": [], "page": page, "per_page": per_page}

    async def delete_submission(self, submission_id: str | int, *, request_timeout: float) -> None:
        self.calls.append(
            {
                "op": "delete_submission",
                "submission_id": submission_id,
                "request_timeout": request_timeout,
            }
        )

    async def list_custom_categories(self, *, request_timeout: float) -> dict[str, Any]:
        self.calls.append({"op": "list_custom_categories", "request_timeout": request_timeout})
        return {"success": True, "custom_categories": []}

    async def create_custom_category(
        self,
        payload: dict[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {"op": "create_custom_category", "payload": payload, "request_timeout": request_timeout}
        )
        return {"id": 99, **payload, "status": "pending"}

    async def get_custom_category(
        self,
        category_id: str | int,
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "op": "get_custom_category",
                "category_id": category_id,
                "request_timeout": request_timeout,
            }
        )
        return {"id": category_id, "name": "Tech Support Scam"}

    async def update_custom_category(
        self,
        category_id: str | int,
        payload: dict[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "op": "update_custom_category",
                "category_id": category_id,
                "payload": payload,
                "request_timeout": request_timeout,
            }
        )
        return {"id": category_id, **payload}

    async def delete_custom_category(
        self, category_id: str | int, *, request_timeout: float
    ) -> None:
        self.calls.append(
            {
                "op": "delete_custom_category",
                "category_id": category_id,
                "request_timeout": request_timeout,
            }
        )

    async def update_settings(
        self, payload: dict[str, Any], *, request_timeout: float
    ) -> dict[str, Any]:
        self.calls.append(
            {"op": "update_settings", "payload": payload, "request_timeout": request_timeout}
        )
        return {"success": True, "settings": payload}


def _signal_item(*, input_modality: str) -> dict[str, Any]:
    return {
        "id": 12345,
        "input_modality": input_modality,
        "verdict": "fraud",
        "top_category": {
            "name": "CEO Impersonation / Wire Diversion",
            "icon": "wire",
            "score": 0.91,
        },
        "category_scores": [
            {"name": "CEO Impersonation / Wire Diversion", "icon": "wire", "score": 0.91},
            {"name": "Vendor Invoice / Payment Redirection", "icon": "invoice", "score": 0.7},
        ],
        "benign_score": 0.12,
        "margin_over_second": 0.21,
        "examples": ["transfer the funds immediately"],
        "top_matches": [
            {"category": "CEO Impersonation / Wire Diversion", "text": "wire funds", "score": 0.92}
        ],
        "duration_seconds": None,
        "created_at": "2026-06-14T10:30:45.123Z",
    }


class TestResembleSignal(unittest.IsolatedAsyncioTestCase):
    async def test_score_text_returns_stable_payload(self) -> None:
        transport = _FakeSignalTransport()
        signal = ResembleSignal(transport=transport, request_timeout=12.0)

        result = await signal.score_text(" wire the funds now ")

        self.assertEqual(result.verdict, "fraud")
        self.assertEqual(result.input_modality, "text")
        self.assertEqual(result.score, 0.91)
        self.assertEqual(result.recommended_action, "block")
        self.assertEqual(
            result.top_category.name if result.top_category else None,
            "CEO Impersonation / Wire Diversion",
        )
        self.assertEqual(len(result.category_scores), 2)
        self.assertEqual(
            transport.calls[0],
            {"op": "score_text", "text": "wire the funds now", "request_timeout": 12.0},
        )
        self.assertEqual(
            result.to_dict(),
            {
                "id": 12345,
                "verdict": "fraud",
                "score": 0.91,
                "recommended_action": "block",
                "input_modality": "text",
                "top_category": {
                    "name": "CEO Impersonation / Wire Diversion",
                    "score": 0.91,
                    "icon": "wire",
                },
                "category_scores": [
                    {
                        "name": "CEO Impersonation / Wire Diversion",
                        "score": 0.91,
                        "icon": "wire",
                    },
                    {
                        "name": "Vendor Invoice / Payment Redirection",
                        "score": 0.7,
                        "icon": "invoice",
                    },
                ],
                "benign_score": 0.12,
                "margin_over_second": 0.21,
                "examples": ["transfer the funds immediately"],
                "top_matches": [
                    {
                        "category": "CEO Impersonation / Wire Diversion",
                        "text": "wire funds",
                        "score": 0.92,
                    }
                ],
                "duration_seconds": None,
                "created_at": "2026-06-14T10:30:45.123Z",
            },
        )

    async def test_score_file_forwards_media_metadata(self) -> None:
        transport = _FakeSignalTransport()
        signal = ResembleSignal(transport=transport)

        result = await signal.score_file(
            b"wav",
            filename="call.wav",
            media_type="audio",
            content_type="audio/wav",
            request_timeout=3.0,
        )

        self.assertEqual(result.input_modality, "audio")
        self.assertEqual(
            transport.calls[0],
            {
                "op": "score_file",
                "file": b"wav",
                "filename": "call.wav",
                "media_type": "audio",
                "content_type": "audio/wav",
                "request_timeout": 3.0,
            },
        )

    async def test_management_helpers_validate_and_forward_payloads(self) -> None:
        transport = _FakeSignalTransport()
        signal = ResembleSignal(transport=transport)

        created = await signal.create_custom_category(
            name="Tech Support Scam",
            scenarios="virus warning\nverify your identity now",
            enabled=True,
        )
        await signal.update_custom_category(99, scenarios=["new scenario"], enabled=False)
        await signal.update_settings(use_builtin_categories=False)

        self.assertEqual(created["status"], "pending")
        self.assertEqual(
            transport.calls[0]["payload"],
            {
                "name": "Tech Support Scam",
                "scenarios": ["virus warning", "verify your identity now"],
                "enabled": True,
            },
        )
        self.assertEqual(
            transport.calls[1]["payload"], {"scenarios": ["new scenario"], "enabled": False}
        )
        self.assertEqual(transport.calls[2]["payload"], {"use_builtin_categories": False})

    async def test_empty_inputs_are_rejected(self) -> None:
        signal = ResembleSignal(transport=_FakeSignalTransport())

        with self.assertRaisesRegex(ValueError, "text is required"):
            await signal.score_text(" ")
        with self.assertRaisesRegex(ValueError, "file is required"):
            await signal.score_file(b"")
        with self.assertRaisesRegex(ValueError, "per_page must be between"):
            await signal.list_submissions(per_page=101)
        with self.assertRaisesRegex(ValueError, "at least one custom category field"):
            await signal.update_custom_category(99)


if __name__ == "__main__":
    unittest.main()
