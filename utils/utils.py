"""
Core Utilities (utils.py)
========================
Provides simple path resolution, file loaders, and imports for the NLP project.
Kept simple and beginner-friendly with no complex try/except blocks.
"""

import os

# Base repository paths
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(UTILS_DIR)

# Standard mapping for common datasets and asset files
FILE_LOCATIONS = {
    "shakespeare.txt": "data/corpora/shakespeare_corpus.txt",
    "en_US.twitter.txt": "data/corpora/twitter_corpus.txt",
    "WSJ_02-21.pos": "data/corpora/wsj_corpus_train.pos",
    "WSJ_24.pos": "data/corpora/wsj_corpus_test.pos",
    "test.words": "data/corpora/wsj_test_tokens.txt",
    "hmm_vocab.txt": "data/dictionaries/hmm_vocab_lexicon.txt",
}


def resolve_path(filepath):
    """
    Finds the full path of a file in the project.
    Checks the local folder, repo root, and standard data/assets folders.
    """
    # 1. Direct path exists
    if os.path.exists(filepath):
        return filepath

    # 2. Check relative to repository root
    root_path = os.path.join(REPO_ROOT, filepath)
    if os.path.exists(root_path):
        return root_path

    # 3. Check standard filename mapping
    basename = os.path.basename(filepath)
    if basename in FILE_LOCATIONS:
        mapped_path = os.path.join(REPO_ROOT, FILE_LOCATIONS[basename])
        if os.path.exists(mapped_path):
            return mapped_path

    # 4. Search in common folders
    folders_to_check = [
        os.path.join(REPO_ROOT, "data", "corpora"),
        os.path.join(REPO_ROOT, "data", "dictionaries"),
        os.path.join(REPO_ROOT, "data", "features"),
        os.path.join(REPO_ROOT, "data", "embeddings"),
        os.path.join(REPO_ROOT, "assets"),
    ]
    for folder in folders_to_check:
        candidate = os.path.join(folder, basename)
        if os.path.exists(candidate):
            return candidate

    return filepath


def safe_open(filepath, mode="r", encoding="utf-8"):
    """
    Opens a file after finding its correct path in the project.
    """
    resolved_path = resolve_path(filepath)
    return open(resolved_path, mode=mode, encoding=encoding)
