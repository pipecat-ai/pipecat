# Local Editable Installs for Examples

When an example's `requirements.txt` file contains a line like `-e ../../..[extras]`, it is a deliberate instruction to install the `pipecat-ai` package in editable mode from the local source tree.

This is necessary because the `pipecat.examples` module is not included in the package when it is distributed via PyPI. To run examples that import from `pipecat.examples`, the test environment must use the local, unpackaged source code.

Do not replace this syntax with a standard `pipecat-ai[extras]` line, as this will cause import errors for examples that rely on the local `pipecat.examples` module.