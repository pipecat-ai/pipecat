#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Subprocess worker that runs a single eval scenario in its own process.

The suite (:mod:`pipecat.evals.suite`) spawns one of these per (bot, scenario)
run so each harness loads its STT/VAD/turn models in its own interpreter. That
isolation matters under concurrency: model construction (ONNX session creation)
holds the GIL for hundreds of milliseconds, and in a single shared process those
loads would freeze the event loop that paces every *other* concurrent run's
real-time audio, garbling their recordings. A process per run keeps each load on
its own GIL.

Invoked as ``python -m pipecat.evals._session_subprocess <config.json>``. The
config (written by the suite) carries the scenario path, bot URL, and run
options; the worker writes the :class:`~pipecat.evals.results.EvalResult` back as
JSON to the ``result_path`` named in the config. The worker silences the console
and (under ``debug``) writes the harness's per-pipeline logs itself, so the suite
only has to read back the result.
"""

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from loguru import logger

from pipecat.evals.harness import EvalSession
from pipecat.evals.results import EvalResult
from pipecat.evals.scenario import EvalScenario
from pipecat.evals.suite import capture_pipeline_logs


async def _run(config: dict) -> EvalResult:
    """Build and run the session described by ``config``."""
    scenario = EvalScenario.load(Path(config["scenario_path"]))
    session = EvalSession.from_scenario(
        scenario,
        config["bot_url"],
        connect_timeout_s=config["connect_timeout_s"],
        default_timeout_ms=config["default_timeout_ms"],
        record_path=config.get("record_path"),
        cache_dir=config.get("cache_dir"),
        use_cache=config["use_cache"],
        stop_bot=config["stop_bot"],
        trigger_disconnect=config.get("trigger_disconnect", False),
    )
    return await session.run()


def main() -> int:
    """Run one scenario from a config file and write its result as JSON."""
    config = json.loads(Path(sys.argv[1]).read_text())
    # No console sink: the worker's stdout/stderr is captured to a log file by the
    # suite, and the result comes back through the result file, not stdout.
    logger.remove()

    # capture_pipeline_logs writes <prefix>.debug.log (only under debug) from the
    # harness's per-pipeline loguru output -- now in this worker process.
    with capture_pipeline_logs(
        Path(config["logs_dir"]),
        config["prefix"],
        name=config["scenario_name"],
        enabled=config["debug"],
    ):
        result = asyncio.run(_run(config))

    Path(config["result_path"]).write_text(json.dumps(dataclasses.asdict(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
