# Punkt tokenizer models

`punkt_tab.zip` is an unmodified copy of the full NLTK Punkt model package.
Its embedded `punkt_tab/README` records the model authors and training sources.

- Source: https://github.com/nltk/nltk_data/blob/4f15a3d89eefe9748ec1c05be495d91289197155/packages/tokenizers/punkt_tab.zip
- Revision date: February 17, 2025
- SHA-256: `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`

NLTK's code license does not establish the license of these model files.
Upstream explicitly identifies Punkt's data license as unclear:
https://github.com/nltk/nltk_data/blob/gh-pages/LICENSE-OVERVIEW.md.
Redistribution clearance is unresolved; this bundle is for local validation
pending that determination.

To update, obtain the archive from a pinned upstream revision, verify its
checksum, update this record, and run the sentence-tokenization tests for all
bundled languages. The loader reads the models directly from the archive;
no runtime download or persistent extraction is needed.
