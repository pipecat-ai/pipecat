#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Offline loading and isolation of the packaged Punkt models."""

import subprocess
import sys
from importlib.resources import as_file, files
from zipfile import ZipFile

import nltk

from pipecat.utils.string import _load_punkt_tokenizer, _sent_tokenizer, match_endofsentence


def test_sentence_detection_without_external_data(monkeypatch):
    _sent_tokenizer.cache_clear()
    monkeypatch.setattr(nltk.data, "path", [])

    def download(*args, **kwargs):
        raise AssertionError("Sentence detection must not download data")

    monkeypatch.setattr(nltk, "download", download)
    try:
        assert match_endofsentence("For Mr. Smith. Next") == len("For Mr. Smith.")
        assert match_endofsentence("こんにちは。次") == len("こんにちは。")
        assert nltk.data.path == []
    finally:
        _sent_tokenizer.cache_clear()


def test_all_bundled_models_load():
    with as_file(files("pipecat.utils.text.data").joinpath("punkt_tab.zip")) as archive:
        with ZipFile(archive) as zipped:
            languages = {
                name.split("/")[1]
                for name in zipped.namelist()
                if name.endswith("/ortho_context.tab")
            }
        assert len(languages) == 19
        for language in languages:
            assert _load_punkt_tokenizer(language)("Hello world.") == ["Hello world."]


def test_import_does_not_load_nltk():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pipecat.utils.string; import sys; assert 'nltk' not in sys.modules",
        ],
        check=True,
    )
