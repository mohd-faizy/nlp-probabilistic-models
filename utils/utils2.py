"""
Continuous Bag-of-Words & Embeddings Utilities (utils2.py)
==========================================================
Provides neural network math, sliding window data generators,
batching, dictionary helpers, PCA projection, and plotting for CBOW.
Kept simple and beginner-friendly with no try/except blocks.
"""

import os
from collections import defaultdict
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from .utils import resolve_path, REPO_ROOT


def sigmoid(z):
    """
    Computes the sigmoid activation function: 1 / (1 + exp(-z)).
    """
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    Computes numerically stable softmax activation by subtracting the max value.
    """
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e_z / np.sum(e_z, axis=0, keepdims=True)


def get_idx(words, word2Ind):
    """
    Converts a list of words into their corresponding integer indices.
    """
    return [word2Ind[word] for word in words]


def pack_idx_with_frequency(context_words, word2Ind):
    """
    Counts how many times each context word appears and packs its index and frequency.
    """
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed


def get_vectors(data, word2Ind, V, C):
    """
    Generator that creates (x, y) training pairs from a text sequence
    using a sliding context window of size C around each center word.
    """
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C):i] + data[(i + 1):(i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            i = 0


def get_batches(data, word2Ind, V, C, batch_size):
    """
    Generates batches of (x, y) matrices ready for neural network training.
    """
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []


def compute_pca(data, n_components=2):
    """
    Reduces the dimension of word vectors using Principal Component Analysis (PCA).
    """
    m, n = data.shape
    data_centered = data - data.mean(axis=0)
    R = np.cov(data_centered, rowvar=False)
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :n_components]
    return np.dot(evecs.T, data_centered.T).T


def get_dict(data):
    """
    Builds two lookup dictionaries:
    word2Ind (maps word -> index) and Ind2word (maps index -> word).
    Can take a list of words or a filename.
    """
    if isinstance(data, str) and os.path.exists(resolve_path(data)):
        with open(resolve_path(data), "r", encoding="utf-8") as f:
            tokens = f.read().split()
        words = sorted(list(set(tokens)))
    else:
        words = sorted(list(set(data)))

    word2Ind = {}
    Ind2word = {}
    for idx, word in enumerate(words):
        word2Ind[word] = idx
        Ind2word[idx] = word
    return word2Ind, Ind2word


def plot_word_vectors(words, word_vectors, word2Ind, n_components=2, fname=None, figsize=(10, 8)):
    """
    Plots a 2D scatter plot of word vectors using PCA with word labels.
    """
    indices = [word2Ind[w] for w in words if w in word2Ind]
    selected_words = [w for w in words if w in word2Ind]
    subset_vectors = word_vectors[indices, :]

    reduced = compute_pca(subset_vectors, n_components=n_components)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(reduced[:, 0], reduced[:, 1], color="#2563eb", edgecolors="#1e40af", s=100, alpha=0.85)

    for i, word in enumerate(selected_words):
        ax.annotate(
            word,
            (reduced[i, 0], reduced[i, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title("2D Word Embedding Projections (PCA)", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    if fname:
        save_path = resolve_path(fname)
        if not os.path.isabs(fname) and not os.path.exists(save_path):
            save_path = os.path.join(REPO_ROOT, "assets", os.path.basename(fname))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax
