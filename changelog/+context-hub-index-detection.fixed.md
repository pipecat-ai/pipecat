Fixed `pipecat init` repeatedly offering to build a Context Hub index that
already exists, and the stale-index warning going quiet, when the index was
built by a newer Context Hub than the CLI knows about.
