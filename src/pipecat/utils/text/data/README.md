# Punkt tokenizer models

`punkt_tab.zip` is an unmodified copy of the full NLTK Punkt model package.
Its embedded `punkt_tab/README` records the model authors and training sources.

- Source: https://github.com/nltk/nltk_data/blob/4f15a3d89eefe9748ec1c05be495d91289197155/packages/tokenizers/punkt_tab.zip
- Revision date: February 17, 2025
- SHA-256: `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`

## License information

Pipecat's code is licensed under BSD-2-Clause. NLTK's code is licensed under
Apache-2.0. Neither code license establishes the license of these model files.
Upstream lists `punkt_tab` as having no license attribute under packages with
unclarified terms. The distribution's `LicenseRef-Punkt-Unclarified` identifies
this unresolved model-data status; it is not a license grant or a claim that
the models are covered by BSD-2-Clause or Apache-2.0.

The following upstream documents are included unchanged, from NLTK data
revision `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` (July 1, 2026):

- [LICENSE](https://github.com/nltk/nltk_data/blob/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/LICENSE): the repository-wide Apache-2.0 license, subject to the package-specific distinctions below.
- [LICENSE-OVERVIEW.md](https://github.com/nltk/nltk_data/blob/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/LICENSE-OVERVIEW.md): the distinction between repository and data-package licenses, including Punkt's unclarified status.
- [DATASET-LICENSES.md](https://github.com/nltk/nltk_data/blob/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/DATASET-LICENSES.md): the package-by-package license inventory.

These documents describe the upstream repository's data collection; Pipecat
includes only `punkt_tab`, not the other datasets listed in that inventory.
The model archive's embedded README preserves its author and training-source
attributions. Providing these notices documents the uncertainty and does not
resolve the missing model-data license grant.

## Updating the bundle

To update, obtain the archive from a pinned upstream revision, verify its
checksum, update this record and the pinned license documents, and run the
sentence-tokenization tests for all bundled languages. Verify that the wheel
and source distribution include the archive and all notices. The loader reads
the models directly from the archive;
no runtime download or persistent extraction is needed.
