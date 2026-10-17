# Data Directory Catalog & Schema Specifications

This catalog provides a comprehensive reference of all datasets, lexicons, feature matrices, and pre-computed embeddings utilized across the **Natural Language Processing with Probabilistic Models** repository.

---

## Directory Architecture

```
data/
├── corpora/                     # Raw and annotated text corpora, plus offline NLTK resources
│   ├── shakespeare_corpus.txt   # Complete Shakespeare plays corpus for vocabulary and CBOW training
│   ├── twitter_corpus.txt       # Filtered English Twitter status corpus for N-gram language modeling
│   ├── wsj_corpus_train.pos     # Penn Treebank Wall Street Journal (Sections 02-21) POS training set
│   ├── wsj_corpus_test.pos      # Penn Treebank Wall Street Journal (Section 24) POS ground truth test set
│   ├── wsj_test_tokens.txt      # Penn Treebank WSJ Section 24 unannotated test token sequence
│   ├── tokenizers/              # Pre-downloaded NLTK Punkt tokenizer pickle files for offline execution
│   └── nltk_data/               # Standardized NLTK data structure hierarchy
├── dictionaries/                # Lexicons, vocabulary lookup tables, and analogy mappings
│   └── hmm_vocab_lexicon.txt    # Vocabulary frequency lexicon for Hidden Markov Model POS tagger
├── features/                    # Tabular feature datasets and transition/emission matrices
│   └── pos_tag_transition_matrix.csv # Tabular state transition probability matrix across Penn Treebank POS tags
└── embeddings/                  # Pre-computed word vectors and pickled weight dictionaries
    └── cbow_word_embeddings.pickle   # Pickled 50-dimensional continuous bag-of-words embedding weights
```

---

## Dataset Schema Catalog

### 1. `data/corpora/`

| File Name | Format | Encoding | File Size | Dimensions / Records | Primary Notebook Mapping | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `shakespeare_corpus.txt` | `.txt` | UTF-8 | 300 KB (306,996 B) | 60,996 tokens, 5,778 unique words | [`03_autocorrect_and_minimum_edit_distance.ipynb`](../03_autocorrect_and_minimum_edit_distance.ipynb), [`16_word_embeddings_continuous_bag_of_words.ipynb`](../16_word_embeddings_continuous_bag_of_words.ipynb) | Unannotated raw text of Shakespeare plays. Used to derive unigram language probabilities for autocorrection and context-target word pairs for CBOW neural network training. |
| `twitter_corpus.txt` | `.txt` | UTF-8 | 3.26 MB (3,341,555 B) | 47,961 tweets, 14,976 unique vocabulary tokens | [`10_autocomplete_ngram_language_model.ipynb`](../10_autocomplete_ngram_language_model.ipynb) | English Twitter social messages separated by newlines. Utilized for sentence tokenization, boundary padding (`<s>`, `</s>`), N-gram count matrix construction, and perplexity evaluation. |
| `wsj_corpus_train.pos` | `.pos` | UTF-8 | 7.90 MB (8,279,089 B) | 989,860 token-tag pairs, 46 unique POS tags | [`04_pos_text_processing_and_unknown_words.ipynb`](../04_pos_text_processing_and_unknown_words.ipynb), [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) | Penn Treebank Wall Street Journal sections 02-21. Each line represents a `token\ttag` record with empty lines demarcating sentence boundaries. Used to compute transition and emission matrices. |
| `wsj_corpus_test.pos` | `.pos` | UTF-8 | 280 KB (286,063 B) | 34,199 token-tag pairs | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) | Penn Treebank WSJ section 24 with ground-truth POS tags used for computing end-to-end Viterbi tagger prediction accuracy and error analysis. |
| `wsj_test_tokens.txt` | `.txt` | UTF-8 | 176 KB (180,266 B) | 34,199 lines (tokens) | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) | Test tokens from WSJ section 24 stripped of POS annotations. Fed into `preprocess(vocab, ...)` and decoded by the Viterbi algorithm. |
| `tokenizers/punkt/` | `.pickle` | Binary | ~400 KB - 2 MB per language | 17 language models | [`07_ngram_corpus_preprocessing.ipynb`](../07_ngram_corpus_preprocessing.ipynb), [`11_cbow_data_preparation.ipynb`](../11_cbow_data_preparation.ipynb), [`15_cbow_word_embeddings_practice.ipynb`](../15_cbow_word_embeddings_practice.ipynb) | Pre-trained unsupervised Punkt sentence boundary and tokenization models enabling 100% offline reproducibility without external network downloads. |

### 2. `data/dictionaries/`

| File Name | Format | Encoding | File Size | Dimensions / Records | Primary Notebook Mapping | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `hmm_vocab_lexicon.txt` | `.txt` | UTF-8 | 192 KB (196,571 B) | 23,776 lines (vocabulary entries) | [`06_pos_tagging_and_viterbi_algorithm.ipynb`](../06_pos_tagging_and_viterbi_algorithm.ipynb) | Pre-computed vocabulary list pairing words with their training corpus frequency. Words occurring < 2 times are mapped to morphological unknown tokens (`--unk_noun--`, `--unk_verb--`, `--unk_digit--`, etc.). |

### 3. `data/features/`

| File Name | Format | Encoding | File Size | Dimensions / Records | Primary Notebook Mapping | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pos_tag_transition_matrix.csv` | `.csv` | UTF-8 | 1.2 KB | $10 \times 10$ transition matrix | [`05_pos_tag_matrices_and_transitions.ipynb`](../05_pos_tag_matrices_and_transitions.ipynb), [`utils.py`](../utils.py) | Stochastic transition probability matrix demonstrating row-normalized transition dynamics between major grammatical tag states (`NN`, `VB`, `JJ`, `IN`, `DT`, etc.). |

### 4. `data/embeddings/`

| File Name | Format | Encoding | File Size | Dimensions / Records | Primary Notebook Mapping | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cbow_word_embeddings.pickle` | `.pickle` | Binary | 3.4 KB | $V = 10$, $N = 50$ embedding vectors | [`14_word_embeddings_extraction.ipynb`](../14_word_embeddings_extraction.ipynb), [`16_word_embeddings_continuous_bag_of_words.ipynb`](../16_word_embeddings_continuous_bag_of_words.ipynb) | Serialized dictionary containing normalized 50-dimensional word vectors, bidirectional vocabulary indices (`word2idx`, `idx2word`), and semantic evaluation vectors. |

---

## Schema Formats & Sample Records

### Penn Treebank POS Annotation (`.pos`)
Each line consists of a token and its assigned Penn Treebank Part-of-Speech tag, separated by a horizontal tab (`\t`). Blank lines represent sentence boundaries (`--s--` / `--n--`).
```tsv
In	IN
an	DT
Oct.	NNP
19	CD
review	NN
of	IN
``	``
The	DT
Misanthrope	NN
''	''
at	IN
Chicago	NNP
...
```

### HMM Lexicon Table (`hmm_vocab_lexicon.txt`)
Whitespace-separated entries listing the vocabulary word and its integer frequency:
```tsv
! 2
# 1
$ 2307
% 1802
& 489
' 138
...
```

---

## Offline Data Loading via `utils.py`

All data files can be loaded using intelligent path resolution:

```python
from utils import resolve_path, safe_open, preprocess, get_dict

# Automatically resolves to data/corpora/shakespeare_corpus.txt
with safe_open("shakespeare_corpus.txt") as f:
    text = f.read()

# Automatically resolves to data/corpora/wsj_corpus_train.pos
orig, prep = preprocess(vocab, "wsj_corpus_train.pos")
```
