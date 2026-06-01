#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Entry point loaded by pipecat-cli.

Registered in this package's ``pyproject.toml`` under the
``[project.entry-points."pipecat_cli.extensions"]`` group. pipecat-cli
discovers and mounts the exported ``entrypoint_cli_typer`` app as a
subcommand of ``pipecat`` (the entry-point name becomes the subcommand name).
"""

from pipecat.cli.commands.eval import eval_app

entrypoint_cli_typer = eval_app
