# DATASET-LICENSES.md

This document provides a grouped summary of licenses for all data packages present in the [`nltk_data`](https://github.com/nltk/nltk_data) repository, based on the current `index.xml` file. Each package is listed by its exact `id` and `name`, and grouped by license type as declared in the metadata.

> **Disclaimer:**  
> This information is provided as a convenience to users and is not legal advice.  
> **You must verify the license for each dataset with the original source if your use case is sensitive (especially for commercial or redistributive use).**  
> Licenses or terms can change over time; this file may become outdated if not maintained.

---

## MIT License

- averaged_perceptron_tagger — Averaged Perceptron Tagger
- averaged_perceptron_tagger_eng — Averaged Perceptron Tagger (JSON)
- averaged_perceptron_tagger_ru — Averaged Perceptron Tagger (Russian)
- averaged_perceptron_tagger_rus — Averaged Perceptron Tagger (Russian)
- vader_lexicon — VADER Sentiment Lexicon

---

## Creative Commons Licenses

### Creative Commons Attribution 4.0 International

- opinion_lexicon — Opinion Lexicon
- product_reviews_1 — Product Reviews (5 Products)
- product_reviews_2 — Product Reviews (9 Products)
- pros_cons — Pros and Cons
- subjectivity — Subjectivity Dataset v1.0

### Creative Commons Attribution 3.0 Unported License

- framenet_v17 — FrameNet 1.7

### Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States

- universal_treebanks_v20 — Universal Treebanks Version 2.0

### Creative Commons Attribution 3.0 (unspecified region)

- sentiwordnet — SentiWordNet

### CC0 1.0 Universal

- panlex_swadesh — PanLex Swadesh Corpora

### CC By SA 3.0 (Wiktionary) & UBY 1.0 (UBY)

- extended_omw — Extended Open Multilingual WordNet

---

## GNU Licenses

### GNU General Public License

- pl196x — Polish language of the XX century sixties

### GNU Free Documentation License

- swadesh — Swadesh Wordlists
- gazetteers — Gazetteer Lists (note: for some files only; others may be public domain)

### GNU Lesser General Public License

- nonbreaking_prefixes — Non-Breaking Prefixes (Moses Decoder)

---

## Public Domain

- genesis — Genesis Corpus
- gutenberg — Project Gutenberg Selections
- inaugural — C-Span Inaugural Address Corpus
- shakespeare — Shakespeare XML Corpus Sample
- udhr — Universal Declaration of Human Rights Corpus
- udhr2 — Universal Declaration of Human Rights Corpus (Unicode Version)
- words — Word Lists

---

## “Distributed with Permission” / “May be used with Permission” / “Freely Redistributable”

> **Warning:**  
> These are not standard open licenses. Terms may prohibit redistribution, modification, or commercial use.  
> **You must consult the upstream source for the actual terms and whether permission applies to your use case.**

- alpino — Alpino Dutch Treebank
- indian — Indian Language POS-Tagged Corpus
- lin_thesaurus — Lin's Dependency Thesaurus
- mac_morpho — MAC-MORPHO: Brazilian Portuguese news text with part-of-speech tags
- paradigms — Paradigm Corpus
- nombank.1.0 — NomBank Corpus 1.0
- propbank — Proposition Bank Corpus 1.0
- senseval — SENSEVAL 2 Corpus: Sense Tagged Text
- verbnet — VerbNet Lexicon, Version 2.1
- verbnet3 — VerbNet Lexicon, Version 3.3
- maxent_treebank_pos_tagger — Treebank Part of Speech Tagger (Maximum entropy)
- maxent_treebank_pos_tagger_tab — Treebank Part of Speech Tagger (Maximum entropy)
- maxent_ne_chunker — ACE Named Entity Chunker (Maximum entropy)
- maxent_ne_chunker_tab — ACE Named Entity Chunker (Maximum entropy)
- pil — The Patient Information Leaflet (PIL) Corpus
- pe08 — Cross-Framework and Cross-Domain Parser Evaluation Shared Task
- kimmo — PC-KIMMO Data Files
- jeita — JEITA Public Morphologically Tagged Corpus
- knbc — KNB Corpus (Annotated blog corpus)

---

## “Non-commercial Use Only” / Educational Use

- brown — Brown Corpus
- brown_tei — Brown Corpus (TEI XML Version)
- framenet_v15 — FrameNet 1.5
- floresta — Portuguese Treebank
- masc_tagged — MASC Tagged Corpus
- nps_chat — NPS Chat

---

## “See LICENSE Files” (Aggregated/Mixed Licensing)

> **Warning:**  
> These packages include files from multiple sources, each with their own license. See LICENSE files inside the package and verify terms for your use case.

- omw — Open Multilingual Wordnet
- omw-1.4 — Open Multilingual Wordnet

---

## Special Cases, Custom, or Unique Licenses

- bcp47 — BCP-47 Language Tags ("IETF Trust and Unicode Inc."; custom)
- wordnet — WordNet ("Permission to use, copy, modify and distribute this software and database and its documentation for any purpose and without fee or royalty")
- wordnet31 — Wordnet 3.1 (same as above)
- wordnet2021 / wordnet2022 / english_wordnet — Open English Wordnet (combines WordNet License and Creative Commons Attribution)
- twitter_samples — Twitter Samples ("Must be used subject to Twitter Developer Agreement")
- switchboard — Switchboard Corpus Sample ("Permission is granted for use of this material in accordance with the Open Content License")
- dependency_treebank — Dependency Parsed Treebank (fragment of Penn Treebank; non-commercial, no redistribution)
- ptb — Penn Treebank (stub for full corpus)
- treebank — Penn Treebank Sample (fragment; non-commercial, no redistribution)
- conll2000 — CONLL 2000 Chunking Corpus (research use only)
- conll2002 — CONLL 2002 Named Entity Recognition Corpus (see website)
- conll2007 — Dependency Treebanks from CoNLL 2007 (Catalan and Basque Subset, see website)
- ieer — NIST IE-ER DATA SAMPLE (see website)
- reuters — Reuters-21578 benchmark corpus, ApteMod version (Reuters Ltd. copyright)
- timit — TIMIT Corpus Sample (Creative Commons Attribution-NonCommercial-ShareAlike 3.0)

---

## Unclarified, Unknown, Ambiguous, or Citation-Only

The following packages have:  
- No `license` attribute  
- An empty or ambiguous value  
- A citation request instead of a license  
- Or otherwise ambiguous status

> **Warning:**  
> These packages lack open, standard, or clearly documented licenses.  
> Citation requests do **not** constitute a license.  
> Despite long-standing and ongoing efforts (see [nltk_data issue #241](https://github.com/nltk/nltk_data/issues/241) and related discussions), clarification has not been possible for these cases.  
> **If you need to use any of these for commercial or redistributive purposes, consult a qualified legal professional.**

- abc — Australian Broadcasting Commission 2006
- basque_grammars — Grammars for Basque
- biocreative_ppi — BioCreAtIvE (Critical Assessment of Information Extraction Systems in Biology)
- bllip_wsj_no_aux — BLLIP Parser: WSJ Model
- book_grammars — Grammars from NLTK Book
- cess_cat — CESS-CAT Treebank (citation requested, not a license)
- cess_esp — CESS-ESP Treebank (citation requested, not a license)
- chat80 — Chat-80 Data Files
- city_database — City Database
- cmudict — The Carnegie Mellon Pronouncing Dictionary (0.6)
- comparative_sentences — Comparative Sentence Dataset (ambiguous license)
- comtrans — ComTrans Corpus Sample
- dolch — Dolch Word List
- europarl_raw — Sample European Parliament Proceedings Parallel Corpus
- framenet_v15 — FrameNet 1.5 (non-commercial use only)
- gazetteers — Gazetteer Lists (mixed per-file)
- large_grammars — Large context-free and feature-based grammars
- machado — Machado de Assis -- Obra Completa ("Public Domain", verify at source)
- moses_sample — Moses Sample Models
- mwa_ppdb — Monolingual word aligner (subset of Paraphrase Database)
- names — Names Corpus, Version 1.3 (1994-03-29)
- nonbreaking_prefixes — Non-Breaking Prefixes (empty license field)
- punkt — Punkt Tokenizer Models (no license attribute)
- punkt_tab — Punkt Tokenizer Models (no license attribute)
- porter_test — Porter Stemmer Test Files
- ppattach — Prepositional Phrase Attachment Corpus
- problem_reports — Problem Report Corpus
- qc — Experimental Data for Question Classification
- rslp — RSLP Stemmer (Removedor de Sufixos da Lingua Portuguesa)
- rte — PASCAL RTE Challenges 1, 2, and 3
- sample_grammars — Sample Grammars
- semcor — SemCor 3.0
- sentence_polarity — Sentence Polarity Dataset v1.0 (ambiguous license)
- smultron — SMULTRON Corpus Sample
- snowball_data — Snowball Data
- spanish_grammars — Grammars for Spanish
- state_union — C-Span State of the Union Address Corpus
- stopwords — Stopwords Corpus
- tagsets — Help on Tagsets
- tagsets_json — Help on Tagsets (JSON)
- toolbox — Toolbox Sample Files
- unicode_samples — Unicode Samples
- webtext — Web Text Corpus
- wmt15_eval — Evaluation data from WMT15
- word2vec_sample — Word2Vec Sample
- wordnet_ic — WordNet-InfoContent
- ycoe — York-Toronto-Helsinki Parsed Corpus of Old English Prose

---

## Packages with Citation Requests Instead of Licenses

> **Note:**  
> These packages specifically request citation for use, but do not provide a license. Citation requests are not a license.

- cess_cat — CESS-CAT Treebank
- cess_esp — CESS-ESP Treebank

---

## Packages Citing Source Website or “See Website” for Terms

> **Note:**  
> These packages refer users to an external website for their licensing terms.

- conll2002 — CONLL 2002 Named Entity Recognition Corpus
- conll2007 — Dependency Treebanks from CoNLL 2007 (Catalan and Basque Subset)
- ieer — NIST IE-ER DATA SAMPLE
- reuters — The Reuters-21578 benchmark corpus, ApteMod version

---

## Maintenance

**If you add, update, or remove any data packages, update this file accordingly to ensure continued transparency and compliance.**  
If you find omissions, errors, or outdated information, please open an issue or pull request.

---