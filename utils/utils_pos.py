"""
Part-of-Speech Tagging Utilities (utils_pos.py)
==============================================
Provides text preprocessing, word-tag extraction, and morphological
classification for unknown words in Part-of-Speech (POS) tagging models.
Kept simple and beginner-friendly with no try/except blocks.
"""

import string
from .utils import resolve_path

punct = set(string.punctuation)

# Common English morphological suffixes
noun_suffix = [
    "action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood",
    "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry",
    "scape", "ship", "ty"
]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = [
    "able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive",
    "less", "ly", "ous"
]
adv_suffix = ["ward", "wards", "wise"]


def assign_unk(tok):
    """
    Assigns a special unknown token based on word morphology rules.
    Helps the model classify words that are not in the vocabulary.
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"
    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"
    # Uppercase
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"
    # Suffix checks
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    # Default unknown
    return "--unk--"


def get_word_tag(line, vocab):
    """
    Extracts the word and tag from a line in an annotated corpus.
    If the word is not in the vocabulary, assigns an unknown word token.
    """
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
        return word, tag


def preprocess(vocab, data_fp):
    """
    Reads a POS tagged dataset and prepares aligned lists of original
    words and preprocessed words (with unknown words replaced).
    """
    filepath = resolve_path(data_fp)
    orig = []
    prep = []

    with open(filepath, "r", encoding="utf-8") as data_file:
        for word in data_file:
            if not word.split():
                orig.append(word.strip())
                prep.append("--n--")
            elif word.strip() not in vocab:
                orig.append(word.strip())
                prep.append(assign_unk(word.strip()))
            else:
                orig.append(word.strip())
                prep.append(word.strip())

    return orig, prep
