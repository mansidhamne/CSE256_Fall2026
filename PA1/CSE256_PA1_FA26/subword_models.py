# subword_models.py
# Part 2: Deep Averaging Network with Byte Pair Encoding.
#
# BPE is learned on train.txt only (Sennrich et al., 2016). Dev sentences are
# segmented with those merges and never update the vocabulary. There are no
# pretrained subword vectors, so the embedding layer is randomly initialized
# and trained from scratch, same as the Part 1b word model.
#
# The network itself matches that Part 1b model (300-d embeddings, one hidden
# layer of size 300, ReLU, no dropout, Adam at 0.001, 8 epochs) so the only
# change across runs is the subword vocabulary size.
#
# python main.py --model SUBWORDDAN

import csv
import heapq
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from DANmodels import build_model, collate_batch, make_optimizer, set_seed, train_dan
from sentiment_data import read_sentiment_examples
from utils import Indexer


END = "</w>"
SEED = 42

# Subword vocabulary sizes, not counting PAD and UNK.
REQUESTED_VOCAB_SIZES = [100, 200, 500, 1000, 2000, 5000, 10000]

# Part 1 numbers for the same architecture and training budget.
WORD_RANDOM_DEV = 0.7901376146788991
WORD_GLOVE_DEV = 0.8188073394495413

CONFIG = {
    "name": "subword",
    "embeddings": "random",
    "num_hidden_layers": 1,
    "hidden_size": 300,
    "dropout": 0.0,
    "dropout_on_embedding": False,
    "dropout_on_hidden": False,
    "nonlinearity": "relu",
    "init": "default",
    "optimizer": "adam",
    "lr": 0.001,
    "epochs": 8,
    "batch_size": 32,
    "random_dim": 300,
}

EXAMPLE_WORDS = [
    "movie", "was", "excellent", "and", "the", "acting", "is", "good"
]


class IndexerEmbeddings:
    """Just enough of the WordEmbeddings interface for a random DAN."""

    def __init__(self, indexer):
        self.word_indexer = indexer


def _apply_merge(symbols, left, right):
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def _retally(word, sign, splits, word_freq, pair_counts, pair_words, changed):
    """Add (sign=+1) or remove (sign=-1) this word's adjacent pairs."""
    symbols = splits[word]
    freq = word_freq[word]
    local = Counter(zip(symbols, symbols[1:]))
    for pair, count in local.items():
        pair_counts[pair] += sign * count * freq
        changed.add(pair)
        if pair_counts[pair] <= 0:
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
        elif sign > 0:
            pair_words[pair].add(word)
        else:
            pair_words[pair].discard(word)


def learn_merges(word_freq, num_merges):
    """Learn `num_merges` BPE merges. Earlier merges are the more frequent pairs.

    Ties are broken by the lexicographically smaller pair so the vocabulary is
    reproducible. Merges stop early if every word has already collapsed to one
    symbol.
    """
    splits = {word: list(word) + [END] for word in word_freq}
    pair_counts = Counter()
    pair_words = defaultdict(set)
    for word in splits:
        _retally(word, +1, splits, word_freq, pair_counts, pair_words, set())

    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    while len(merges) < num_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count != -neg_count or count <= 0:
            continue
        merges.append(pair)
        affected = list(pair_words.get(pair, ()))
        changed = set()
        for word in affected:
            _retally(word, -1, splits, word_freq, pair_counts, pair_words, changed)
            splits[word] = _apply_merge(splits[word], pair[0], pair[1])
            _retally(word, +1, splits, word_freq, pair_counts, pair_words, changed)
        for changed_pair in changed:
            changed_count = pair_counts.get(changed_pair, 0)
            if changed_count > 0:
                heapq.heappush(heap, (-changed_count, changed_pair))
        if len(merges) % 2000 == 0:
            print(f"  learned {len(merges)} merges")
    return merges


def base_symbols(word_freq):
    symbols = {END}
    for word in word_freq:
        symbols.update(word)
    return symbols


class BPETokenizer:
    """Applies a prefix of the learned merges and maps pieces to indices.

    Index 0 is PAD and index 1 is UNK, matching the word-level DAN. A character
    that never occurred in training is emitted as UNK.
    """

    def __init__(self, merges, symbols):
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.indexer = Indexer()
        self.indexer.add_and_get_index("PAD")
        self.indexer.add_and_get_index("UNK")
        for symbol in sorted(symbols):
            self.indexer.add_and_get_index(symbol)
        self.unk_idx = self.indexer.index_of("UNK")
        self.cache = {}

    def encode_word(self, word):
        cached = self.cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word) + [END]
        while len(symbols) >= 2:
            best_rank = None
            best_index = None
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index is None:
                break
            left = symbols[best_index]
            right = symbols[best_index + 1]
            symbols = symbols[:best_index] + [left + right] + symbols[best_index + 2:]
        indices = []
        for symbol in symbols:
            idx = self.indexer.index_of(symbol)
            indices.append(self.unk_idx if idx == -1 else idx)
        self.cache[word] = indices
        return indices

    def encode_sentence(self, words):
        pieces = []
        for word in words:
            pieces.extend(self.encode_word(word))
        return pieces

    def segment_word(self, word):
        ids = self.encode_word(word)
        parts = []
        for idx in ids:
            symbol = self.indexer.get_object(idx)
            parts.append("UNK" if symbol is None else symbol)
        return parts


class SubwordDataset(Dataset):
    def __init__(self, infile, tokenizer):
        examples = read_sentiment_examples(infile)
        self.sentences = [tokenizer.encode_sentence(ex.words) for ex in examples]
        self.labels = [ex.label for ex in examples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


def _word_frequencies(examples):
    counts = Counter()
    for ex in examples:
        counts.update(ex.words)
    return counts


def _coverage(word_freq, tokenizer):
    """How much of the corpus has already been glued into a single piece."""
    single_types = 0
    single_tokens = 0
    total_tokens = 0
    total_pieces = 0
    for word, freq in word_freq.items():
        pieces = len(tokenizer.encode_word(word))
        total_pieces += pieces * freq
        total_tokens += freq
        if pieces == 1:
            single_types += 1
            single_tokens += freq
    return {
        "single_type_fraction": single_types / len(word_freq),
        "single_token_fraction": single_tokens / total_tokens,
        "pieces_per_word": total_pieces / total_tokens,
    }


def _make_tokenizer(merges, symbols, vocab_size):
    num_merges = vocab_size - len(symbols)
    if num_merges < 0:
        raise ValueError(f"vocab size {vocab_size} is smaller than the character vocabulary")
    used = merges[:num_merges]
    vocab = set(symbols)
    for left, right in used:
        vocab.add(left + right)
    return BPETokenizer(used, vocab)


def _result_row(vocab_size, tokenizer, word_freq, train_data, best, history, seconds):
    coverage = _coverage(word_freq, tokenizer)
    lengths = [len(sentence) for sentence in train_data.sentences]
    final_train, _, final_dev, _ = history[-1]
    return {
        "vocab_size": vocab_size,
        "embedding_rows": len(tokenizer.indexer),
        "embedding_params": len(tokenizer.indexer) * CONFIG["random_dim"],
        "avg_subwords_per_sentence": sum(lengths) / len(lengths),
        "pieces_per_word": coverage["pieces_per_word"],
        "single_token_fraction": coverage["single_token_fraction"],
        "single_type_fraction": coverage["single_type_fraction"],
        "best_dev_accuracy": best["dev_accuracy"],
        "best_epoch": best["epoch"],
        "train_accuracy_at_best": best["train_accuracy"],
        "dev_loss_at_best": best["dev_loss"],
        "final_train_accuracy": final_train,
        "final_dev_accuracy": final_dev,
        "seconds": seconds,
    }


def _print_examples(tokenizer, words):
    for word in words:
        parts = " ".join(tokenizer.segment_word(word))
        print(f"    {word:16} {parts}")


def _write_results(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_vocab_curve(rows, path):
    sizes = [row["vocab_size"] for row in rows]
    accuracy = [row["best_dev_accuracy"] * 100 for row in rows]
    seconds = [row["seconds"] for row in rows]

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))

    axes[0].plot(sizes, accuracy, marker="o", ms=5, color="#c45911", label="Subword DAN")
    axes[0].axhline(
        WORD_RANDOM_DEV * 100,
        color="#1f4e79",
        ls="--",
        lw=1,
        label=f"Word-level, random ({WORD_RANDOM_DEV * 100:.1f}%)",
    )
    # axes[0].axhline(
    #     WORD_GLOVE_DEV * 100,
    #     color="#548235",
    #     ls="--",
    #     lw=1,
    #     label=f"Word-level, GloVe ({WORD_GLOVE_DEV * 100:.1f}%)",
    # )
    axes[0].set_xscale("log")
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels([str(size) for size in sizes], rotation=45, ha="right")
    for size, acc in zip(sizes, accuracy):
        axes[0].annotate(
            f"{acc:.1f}",
            (size, acc),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=6,
            color="#c45911",
        )
    axes[0].set_xlabel("BPE vocabulary size")
    axes[0].set_ylabel("Best dev accuracy (%)")
    axes[0].set_ylim(58, 86)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("Dev accuracy")

    axes[1].plot(sizes, seconds, marker="o", ms=5, color="#1f4e79")
    axes[1].set_xscale("log")
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels([str(size) for size in sizes], rotation=45, ha="right")
    axes[1].set_xlabel("BPE vocabulary size")
    axes[1].set_ylabel("Training time (seconds)")
    axes[1].set_ylim(0, 50)
    axes[1].set_title("Training time, 8 epochs")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_subword_dan():
    """Train one DAN per vocabulary size and write the comparison plot."""
    print("\nSubword DAN: BPE on train.txt, random embeddings, same network as Part 1b")
    train_examples = read_sentiment_examples("data/train.txt")
    word_freq = _word_frequencies(train_examples)
    symbols = base_symbols(word_freq)
    vocab_sizes = [size for size in REQUESTED_VOCAB_SIZES if size > len(symbols)]
    num_merges = vocab_sizes[-1] - len(symbols)
    print(f"Learning {num_merges} merges (up to vocab size {vocab_sizes[-1]})")
    merges = learn_merges(word_freq, num_merges)
    print(f"Learned {len(merges)} merges")
    if len(merges) < num_merges:
        vocab_sizes = [size for size in vocab_sizes if size <= len(symbols) + len(merges)]

    rows = []
    for vocab_size in vocab_sizes:
        print(f"\n=== vocab size {vocab_size} ===")
        tokenizer = _make_tokenizer(merges, symbols, vocab_size)
        _print_examples(tokenizer, EXAMPLE_WORDS)
        train_data = SubwordDataset("data/train.txt", tokenizer)
        dev_data = SubwordDataset("data/dev.txt", tokenizer)
        coverage = _coverage(word_freq, tokenizer)
        mean_len = sum(len(s) for s in train_data.sentences) / len(train_data)
        print(
            f"  pieces/word {coverage['pieces_per_word']:.2f}, "
            f"subwords/sentence {mean_len:.1f}, "
            f"train tokens that are one piece {coverage['single_token_fraction'] * 100:.1f}%"
        )
        train_loader = DataLoader(
            train_data,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            collate_fn=collate_batch,
        )
        dev_loader = DataLoader(
            dev_data,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            collate_fn=collate_batch,
        )
        set_seed(SEED)
        model = build_model(IndexerEmbeddings(tokenizer.indexer), CONFIG)
        optimizer = make_optimizer(model, CONFIG)
        start = time.time()
        best, history = train_dan(
            model, train_loader, dev_loader, optimizer, CONFIG["epochs"], log_every=1
        )
        seconds = time.time() - start
        row = _result_row(vocab_size, tokenizer, word_freq, train_data, best, history, seconds)
        rows.append(row)
        print(
            f"  best dev {row['best_dev_accuracy']:.3f} at epoch {row['best_epoch']} "
            f"(train {row['train_accuracy_at_best']:.3f}), "
            f"final dev {row['final_dev_accuracy']:.3f}, {seconds:.1f}s"
        )
        del model

    _write_results("subword_sweep_results.csv", rows)
    plot_vocab_curve(rows, "part2_vocab_curve.png")
    print("\nWrote subword_sweep_results.csv and part2_vocab_curve.png")
    best_row = max(rows, key=lambda row: row["best_dev_accuracy"])
    print(
        f"Best subword dev accuracy {best_row['best_dev_accuracy']:.3f} "
        f"at vocab size {best_row['vocab_size']}"
    )
    print(f"Word-level random DAN: {WORD_RANDOM_DEV:.3f}")
    print(f"Word-level GloVe DAN: {WORD_GLOVE_DEV:.3f}")
    return rows


if __name__ == "__main__":
    train_subword_dan()
