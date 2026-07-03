"""Entry point for ``python -m pipecat.evals``.

Invokes the same typer app the ``pipecat`` CLI mounts as its ``eval`` subcommand,
so both ``python -m pipecat.evals run ...`` and ``pipecat eval run ...`` execute
identical code.
"""

from pipecat.cli.commands.eval import eval_app

eval_app()
