# Pipecat Evals

This directory contains a set of utilities to help test Pipecat, specifically
its examples.

## Release Evals

Before any Pipecat release, we make sure that all (or most) of the examples work
flawlessly.  We have 100+ examples, and checking each one manually was very
time-consuming (and painful!), especially because we aim to release often.

To make this process easier, we designed these "release evals," which do the
following:

- Start one of the foundational examples (the user bot)
- Start an eval bot

The user bot (i.e. the example) introduces itself, and the eval bot then asks a
question. The user bot replies, and the eval bot verifies the response.

For example, the eval bot might ask:

"What's 2 plus 2?"

The user bot replies:

"2 plus 2 is 4."

The eval bot (powered by an LLM) evaluates the response and emits a result.  It
also explains why it thinks the answer is valid or invalid.

To run the release evals:

```sh
uv run run-release-evals.py -a -v
```

This runs all the evals and stores logs and audio (`-a`) for each test.

You can also specify which tests to run. For example, to run all `07` series
tests:

```sh
uv run run-release-evals.py -p 07 -a -v
```

## Script Evals

You can also run evals for a single example (not part of the release set):

```sh
uv run run-eval.py -p "A simple math addition" -a -v YOUR_EXAMPLE_SCRIPT
```

Your script needs to follow any of the foundation examples pattern.
