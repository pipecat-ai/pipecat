#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The web page that runs setups interactively, served by FastAPI.

``pipecat textflow serve`` runs :func:`create_app` under uvicorn. The page is a
single static file; everything it shows comes from three endpoints:

- ``GET /api/config`` says which bot file, if any, the server was started with.
- ``GET /api/presets`` lists the bundled setups.
- ``GET /api/transforms`` lists the registered transforms and their options.
- ``POST /api/bot`` captures a bot file and returns the setup that describes it.
- ``POST /api/trace`` runs a setup over a response and returns the trace.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .capture import BotSummary, TextFlowCaptureError, setup_for_bot
from .setup import TextFlowSetup, presets
from .trace import TextFlowRunError, TextFlowTrace, run_textflow
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


class BotRequest(BaseModel):
    """The body of ``POST /api/bot``.

    Parameters:
        path: The bot file to capture.
    """

    path: str


class BotResponse(BaseModel):
    """What ``POST /api/bot`` returns.

    Parameters:
        setup: A setup describing the bot, with ``bot`` set to its path.
        bot: What was found in the file.
    """

    setup: TextFlowSetup
    bot: BotSummary


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


def create_app(bot: str | None = None):
    """Build the FastAPI application.

    Args:
        bot: A bot file the page loads when it opens.

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

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {"bot": bot}

    @app.get("/api/presets")
    async def list_presets() -> list[TextFlowSetup]:
        return presets()

    @app.post("/api/bot")
    async def load_bot(request: BotRequest) -> BotResponse:
        try:
            setup, summary = await setup_for_bot(request.path)
        except TextFlowCaptureError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return BotResponse(setup=setup, bot=summary)

    @app.get("/api/transforms")
    async def list_transforms() -> list[dict[str, Any]]:
        return transform_catalog()

    @app.post("/api/trace")
    async def trace(request: TraceRequest) -> TextFlowTrace:
        try:
            return await run_textflow(request.setup, request.text)
        except (KeyError, TextFlowCaptureError, TextFlowRunError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app


def serve(host: str = "127.0.0.1", port: int = 7865, bot: str | None = None):
    """Run the page under uvicorn until interrupted.

    Args:
        host: The interface to listen on.
        port: The port to listen on.
        bot: A bot file the page loads when it opens.
    """
    import uvicorn

    uvicorn.run(create_app(bot), host=host, port=port, log_level="warning")
