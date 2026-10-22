#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The web page that runs setups interactively, served by FastAPI.

``pipecat textflow serve`` runs :func:`create_app` under uvicorn. The page is a
single static file; everything it shows comes from three endpoints:

- ``GET /api/presets`` lists the bundled setups.
- ``GET /api/transforms`` lists the registered transforms and their options.
- ``POST /api/trace`` runs a setup over a response and returns the trace.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .setup import TextFlowSetup, presets
from .trace import TextFlowTrace, run_textflow
from .transforms import registered_transforms

STATIC_DIR = Path(__file__).parent / "static"


class TraceRequest(BaseModel):
    """The body of ``POST /api/trace``.

    Parameters:
        setup: The setup to run.
        text: The LLM's response. The setup's sample text when None.
    """

    setup: TextFlowSetup
    text: str | None = None


def transform_catalog() -> list[dict[str, Any]]:
    """The registered transforms as the page lists them."""
    return [
        {
            "name": entry.name,
            "description": entry.description,
            "options": [
                {
                    "name": option.name,
                    "kind": option.kind,
                    "default": option.default,
                    "description": option.description,
                }
                for option in entry.options
            ],
        }
        for entry in registered_transforms()
    ]


def create_app():
    """Build the FastAPI application.

    Raises:
        ImportError: If FastAPI is not installed; it comes with the ``runner`` extra.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "pipecat textflow serve needs FastAPI and uvicorn. "
            "Install them with: pip install 'pipecat-ai[runner]'"
        ) from e

    app = FastAPI(title="Pipecat text flow")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/presets")
    async def list_presets() -> list[TextFlowSetup]:
        return presets()

    @app.get("/api/transforms")
    async def list_transforms() -> list[dict[str, Any]]:
        return transform_catalog()

    @app.post("/api/trace")
    async def trace(request: TraceRequest) -> TextFlowTrace:
        try:
            return await run_textflow(request.setup, request.text)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


def serve(host: str = "127.0.0.1", port: int = 7865):
    """Run the page under uvicorn until interrupted."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
