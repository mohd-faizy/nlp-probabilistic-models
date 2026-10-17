# Utility Package (`utils/`)

This directory contains simple, beginner-friendly Python helper files used across the notebooks in this repository.

---

## 📁 Files in This Folder

| File | Primary Purpose | Used In | Key Functions & Variables |
| :--- | :--- | :--- | :--- |
| **`utils.py`** | Path resolution & dataset loading | All notebooks | `resolve_path()`, `safe_open()`, `REPO_ROOT` |
| **`utils_pos.py`** | Part-of-Speech (POS) text preprocessing | Week 2 notebooks (POS & Viterbi) | `get_word_tag()`, `preprocess()`, `assign_unk()`, morphology lists (`noun_suffix`, etc.) |
| **`utils2.py`** | Neural network math, CBOW batches & PCA plotting | Week 4 notebooks (Word2Vec CBOW) | `sigmoid()`, `softmax()`, `get_dict()`, `get_batches()`, `compute_pca()`, `plot_word_vectors()` |
| **`utils_nb.py`** | Quick notebook imports & environment helpers | Notebook sessions | Combines common helpers for easy notebook access |
| **`__init__.py`** | Python package initialization | Package import system | Allows `import utils` and `from utils import ...` |

---

## 🔍 Detailed Explanation of Each File

### 1. `utils.py` (Path Resolution & File Helpers)
When running Jupyter notebooks, file paths can sometimes break depending on where you launched the notebook from. `utils.py` provides:
- **`resolve_path(filepath)`**: Takes a filename like `"shakespeare.txt"` and automatically finds its full path inside `data/corpora/` or wherever it is located in the repo.
- **`safe_open(filepath)`**: Opens a file directly without worrying about relative folder paths.

### 2. `utils_pos.py` (Part-of-Speech Tagging Helpers)
Used specifically for Part-of-Speech tagging and Hidden Markov Model implementations:
- **`assign_unk(word)`**: If a word isn't in our vocabulary, this function inspects its ending (e.g., `-ly`, `-tion`, `-ing`) to assign a helpful unknown token like `--unk_noun--`, `--unk_verb--`, or `--unk_digit--`.
- **`get_word_tag(line, vocab)`**: Reads a line containing `word tag` and returns them as a tuple.
- **`preprocess(vocab, data_fp)`**: Reads an entire tagged corpus file and returns two lists: original words and cleaned/preprocessed words with unknown tokens.

### 3. `utils2.py` (Continuous Bag-of-Words & Embeddings)
Used for training the Continuous Bag-of-Words (CBOW) neural network from scratch in NumPy:
- **`sigmoid(z)`** & **`softmax(z)`**: Standard neural network activation functions implemented with numerical stability.
- **`get_dict(data)`**: Creates two lookup dictionaries from your text: `word2Ind` (word to index) and `Ind2word` (index to word).
- **`get_vectors(data, word2Ind, V, C)`**: Sliding window generator that yields `(x, y)` training pairs (context words $\bar{x}$ and center target word $y$).
- **`get_batches(data, word2Ind, V, C, batch_size)`**: Batches multiple `(x, y)` pairs into matrices for mini-batch gradient descent.
- **`compute_pca(data, n_components=2)`**: Principal Component Analysis that reduces 50-dimensional or 300-dimensional word vectors down to 2 dimensions.
- **`plot_word_vectors(...)`**: Plots a 2D scatter plot of word vectors with labels using matplotlib.

### 4. `utils_nb.py` (Notebook Shortcuts)
A simple helper module that imports common functions so you can quickly use them inside any interactive notebook session.

---

## 💡 How to Import in Your Notebooks

You can import functions in any of the following standard ways:

```python
# Option 1: Direct imports from the utils package
from utils import resolve_path, safe_open
from utils import get_word_tag, preprocess
from utils import sigmoid, softmax, get_dict, compute_pca

# Option 2: Importing specific files
from utils.utils_pos import get_word_tag, preprocess
from utils.utils2 import get_dict, compute_pca, plot_word_vectors

# Option 3: Standard legacy imports (supported via root shims)
from utils_pos import get_word_tag, preprocess
from utils2 import get_dict, sigmoid, get_batches, compute_pca
```
