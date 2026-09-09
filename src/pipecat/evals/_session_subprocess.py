#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Runs one scenario, scripted or simulated, in its own process.

The suite spawns one of these per run so each harness loads its audio
models on its own GIL; in one shared process those loads would stall the
event loop that paces every other run's audio.

Invoked as ``python -m pipecat.evals._session_subprocess <config.json>``.
The config carries the scenario path, the bot URL, and the run's params;
the result is written back as JSON to the ``result_path`` it names.
"""

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from loguru import logger

from pipecat.evals.results import EvalScriptResult, EvalSimulationResult
from pipecat.evals.scenario import load_scenario_file
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.evals.suite import capture_pipeline_logs


async def _run(config: dict) -> EvalScriptResult | EvalSimulationResult:
    """Build and run the session for the scenario file in ``config``, whichever kind it is."""
    loaded = load_scenario_file(Path(config["scenario_path"]))
    params = EvalSessionParams.model_validate(config["params"])
    session = EvalSession.from_scenario(loaded, config["bot_url"], params=params)
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
