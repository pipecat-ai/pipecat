#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The backend the ``LLMWithBackend`` examples share: an engineering assistant's slow half.

The backend runs Claude with tools that stand in for real engineering work,
each taking as long as the real thing plausibly would:

- **Code changes**, a loop: ``search_codebase`` finds the files, ``read_file``
  reads them, ``apply_patch`` changes one, ``run_tests`` checks the result. The
  first test run after a change to the client trips a neighbouring test that
  wants each retry delay reported to a metrics module, so the backend has to
  read that test, patch again and run again before it can report the change
  done. A minute or so end to end, most of it the model's own steps.
- **Research**: ``search_docs`` then ``read_doc``, a few seconds each.
- **Quick lookups**: ``check_ci_status`` and ``list_open_prs``, a second each.

The slow tools are async (``cancel_on_interruption=False``), so the backend's
model keeps taking new messages while they run and can cancel them when a
message makes their results unwanted.

Each frontend example builds its backend with :func:`build_backend` and gives
its own model :data:`FRONTEND_INSTRUCTIONS`, so the three differ only in the
frontend service.
"""

import asyncio
import os
import re
from datetime import datetime

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.llm import BackendLLMWorker

FRONTEND_INSTRUCTIONS = """You are Pip, the voice assistant of the Acme engineering team. Your
responses are spoken aloud, so keep them to one or two natural sentences without any
formatting. You can make code changes and run the tests, research the team's internal
docs, and check CI and the open pull requests; that work takes a moment."""

BACKEND_INSTRUCTIONS = """You do the engineering work for the Acme team's assistant. Use your
tools: search the codebase, read files, apply patches and run the tests for code changes;
search and read the internal docs for research; check CI and list pull requests for status.
When asked for a code change, keep going until the tests pass, then report what you changed
and the test result. Keep what you write short and concrete: file names, what changed, what
the tests said."""


# The fake repository: what the tools find, read and change.

_FILES = {
    "src/http_client.py": (
        "import time\n\n\n"
        "class TransientError(Exception):\n"
        "    pass\n\n\n"
        "class RetriesExhausted(Exception):\n"
        "    pass\n\n\n"
        "class HttpClient:\n"
        "    RETRIES = 3\n"
        "    BACKOFF_SECS = 1\n\n"
        "    def __init__(self, transport):\n"
        "        self._transport = transport\n\n"
        "    def get(self, url):\n"
        "        for attempt in range(self.RETRIES):\n"
        "            try:\n"
        "                return self._transport(url)\n"
        "            except TransientError:\n"
        "                time.sleep(self.BACKOFF_SECS)\n"
        "        raise RetriesExhausted(url)\n"
    ),
    "tests/test_http_client.py": (
        "import time\n\n"
        "from src.http_client import HttpClient, TransientError\n\n\n"
        "def flaky(fail_times):\n"
        "    calls = []\n\n"
        "    def transport(url):\n"
        "        calls.append(url)\n"
        "        if len(calls) <= fail_times:\n"
        "            raise TransientError()\n"
        "        return 'ok'\n\n"
        "    return transport\n\n\n"
        "def test_retry_backoff(monkeypatch):\n"
        "    sleeps = []\n"
        "    monkeypatch.setattr(time, 'sleep', sleeps.append)\n"
        "    HttpClient(flaky(2)).get('https://example.test/flaky')\n"
        "    # The backoff between retries must grow, or a busy upstream\n"
        "    # gets hammered at a fixed rate.\n"
        "    assert sleeps == sorted(sleeps) and sleeps[0] < sleeps[-1]\n\n\n"
        "def test_first_retry_waits_one_base_interval(monkeypatch):\n"
        "    sleeps = []\n"
        "    monkeypatch.setattr(time, 'sleep', sleeps.append)\n"
        "    HttpClient(flaky(1)).get('https://example.test/flaky')\n"
        "    assert sleeps[0] == HttpClient.BACKOFF_SECS\n"
    ),
    "src/metrics.py": (
        "_backoffs = []\n\n\n"
        "def record_backoff(attempt, delay):\n"
        "    # Report one retry delay, so the retry dashboard can chart them.\n"
        "    _backoffs.append((attempt, delay))\n"
    ),
    "tests/test_retry_metrics.py": (
        "import time\n\n"
        "from src import metrics\n"
        "from src.http_client import HttpClient\n"
        "from tests.test_http_client import flaky\n\n\n"
        "def test_every_retry_delay_is_recorded(monkeypatch):\n"
        "    monkeypatch.setattr(time, 'sleep', lambda secs: None)\n"
        "    recorded = []\n"
        "    monkeypatch.setattr(metrics, 'record_backoff', lambda attempt, delay: recorded.append(delay))\n"
        "    HttpClient(flaky(2)).get('https://example.test/flaky')\n"
        "    # The dashboard charts what the client actually waited, so the client\n"
        "    # reports each delay before sleeping it.\n"
        "    assert len(recorded) == 2\n"
    ),
}

_DOCS = {
    "rfc-12": {
        "title": "RFC 12: Retry policy for outbound HTTP",
        "summary": (
            "Retries of outbound HTTP calls back off exponentially from a base of one second, "
            "doubling on each attempt up to a cap of thirty seconds, with full jitter. Fixed "
            "delays are disallowed because they synchronize clients against a struggling "
            "upstream."
        ),
    },
    "guide-http": {
        "title": "HTTP client guide",
        "summary": (
            "HttpClient wraps requests with retries, timeouts and tracing. Retry behaviour is "
            "governed by RFC 12; the client's fixed BACKOFF_SECS sleep predates it."
        ),
    },
    "oncall-runbook": {
        "title": "On-call runbook: upstream timeouts",
        "summary": (
            "When the payments upstream times out, check its status page first, then the "
            "retry dashboard for a spike in RetriesExhausted."
        ),
    },
}

# Changes applied to each file, in order. The test run reads this to decide
# what passes.
_patches: dict[str, list[str]] = {}


async def _work(seconds: float) -> None:
    """Take about as long as the real thing would."""
    await asyncio.sleep(seconds * float(os.getenv("BACKEND_WORK_SCALE", "1")))


async def search_codebase(params: FunctionCallParams, query: str):
    """Search the codebase for files relevant to a query.

    Args:
        query: What to look for: a symbol, a file name, or a description.
    """
    await _work(2)
    terms = query.lower().split()
    hits = [
        path
        for path, content in _FILES.items()
        if any(term in path.lower() or term in content.lower() for term in terms)
    ] or list(_FILES)
    await params.result_callback({"files": hits})


async def read_file(params: FunctionCallParams, path: str):
    """Read a file from the codebase.

    Args:
        path: The file's path, as search_codebase reports it.
    """
    await _work(1.5)
    if path not in _FILES:
        await params.result_callback({"error": f"no such file: {path}"})
        return
    await params.result_callback({"path": path, "content": _FILES[path]})


async def apply_patch(params: FunctionCallParams, path: str, content: str):
    """Replace a file's content in the codebase.

    Args:
        path: The file to change.
        content: The file's new content, in full.
    """
    await _work(2)
    if path not in _FILES:
        await params.result_callback({"error": f"no such file: {path}"})
        return
    _FILES[path] = content
    _patches.setdefault(path, []).append(content)
    await params.result_callback({"path": path, "applied": True, "lines": content.count("\n")})


@tool_options(cancel_on_interruption=False)
async def run_tests(params: FunctionCallParams, path: str | None = None):
    """Run the test suite, or one test file. Takes a while.

    Args:
        path: A test file to run alone; the whole suite when omitted.
    """
    await _work(5)
    # Two tests read the client as last patched: the backoff test wants the
    # sleep to depend on the attempt, and a neighbouring metrics test the
    # backend has not seen wants each delay reported to the metrics module.
    # A fix that does both at once passes both; one that does only the first
    # trips the second, and the backend has to read that test and patch again.
    client = (_patches.get("src/http_client.py") or [_FILES["src/http_client.py"]])[-1]
    failures = []
    if re.search(r"sleep\(\s*(\w+\.)?BACKOFF_SECS\s*\)", client):
        failures.append(
            {
                "test": "tests/test_http_client.py::test_retry_backoff",
                "message": "assert [1, 1] == sorted([1, 1]) and 1 < 1: the retry delays do not grow",
            }
        )
    if "record_backoff(" not in client:
        failures.append(
            {
                "test": "tests/test_retry_metrics.py::test_every_retry_delay_is_recorded",
                "message": (
                    "assert 0 == 2: no retry delay was recorded; the client must call "
                    "metrics.record_backoff(attempt, delay) before each sleep (see "
                    "tests/test_retry_metrics.py)"
                ),
            }
        )
    await params.result_callback(
        {"passed": 42 - len(failures), "failed": len(failures), "failures": failures}
    )


@tool_options(cancel_on_interruption=False)
async def search_docs(params: FunctionCallParams, query: str):
    """Search the team's internal docs. Takes a few seconds.

    Args:
        query: What to look for.
    """
    await _work(3)
    terms = query.lower().split()
    hits = [
        {"id": doc_id, "title": doc["title"]}
        for doc_id, doc in _DOCS.items()
        if any(term in doc["title"].lower() or term in doc["summary"].lower() for term in terms)
    ] or [{"id": doc_id, "title": doc["title"]} for doc_id, doc in _DOCS.items()]
    await params.result_callback({"docs": hits})


async def read_doc(params: FunctionCallParams, doc_id: str):
    """Read one of the team's internal docs.

    Args:
        doc_id: The doc's id, as search_docs reports it.
    """
    await _work(2)
    doc = _DOCS.get(doc_id)
    if doc is None:
        await params.result_callback({"error": f"no such doc: {doc_id}"})
        return
    await params.result_callback({"id": doc_id, **doc})


async def check_ci_status(params: FunctionCallParams, branch: str = "main"):
    """Check the CI status of a branch.

    Args:
        branch: The branch; main when omitted.
    """
    await _work(1)
    await params.result_callback(
        {
            "branch": branch,
            "status": "green" if branch == "main" else "running",
            "last_run": datetime.now().strftime("%H:%M"),
        }
    )


async def list_open_prs(params: FunctionCallParams):
    """List the team's open pull requests."""
    await _work(1)
    await params.result_callback(
        {
            "pull_requests": [
                {"number": 412, "title": "Add tracing to the payments client", "author": "ana"},
                {"number": 415, "title": "Bump httpx to 0.28", "author": "renovate"},
                {
                    "number": 418,
                    "title": "Retry policy: exponential backoff (RFC 12)",
                    "author": "sam",
                },
            ]
        }
    )


def build_backend() -> BackendLLMWorker:
    """Build the backend worker the examples share."""
    return BackendLLMWorker(
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                # Thinking summaries reach the frontend as silent "Backend (thinking):"
                # messages, so it can say how the work is going if asked. The budget
                # keeps a step to seconds: left to decide for itself, the model can
                # think for a minute over one file.
                thinking=AnthropicLLMService.ThinkingConfig(
                    type="enabled", budget_tokens=2048, display="summarized"
                ),
                # Thinking counts against max_tokens; the budget plus a patch with
                # a file's whole content must fit.
                max_tokens=16384,
            ),
        ),
        context=LLMContext(
            tools=[
                search_codebase,
                read_file,
                apply_patch,
                run_tests,
                search_docs,
                read_doc,
                check_ci_status,
                list_open_prs,
            ]
        ),
    )
