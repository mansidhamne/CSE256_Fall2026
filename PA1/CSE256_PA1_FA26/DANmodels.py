# DANmodels.py
# Deep Averaging Network for binary sentiment classification.
#
# The sentence vector is the mean of the word embeddings (PAD is ignored).
# A feedforward network maps that vector to two logits. Training uses
# CrossEntropyLoss, which applies log-softmax internally.
#
# python main.py --model DAN
#     trains FINAL_CONFIG below
# python main.py --model DAN --sweep
#     runs the Phase 1 grid (1/2/3 hidden layers x width x GloVe 50d/300d x dropout)
#     then Phase 2 one-factor follow-ups on the Phase 1 winner, including
#     randomly initialized embeddings (deliverable 1b)

import csv
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sentiment_data import read_sentiment_examples, read_word_embeddings


EMBEDDING_FILES = {
    "50d": "data/glove.6B.50d-relativized.txt",
    "300d": "data/glove.6B.300d-relativized.txt",
}

# Model trained by `python main.py --model DAN`.
# Swap this for the Phase 1 winner after you run --sweep.
FINAL_CONFIG = {
    "name": "final",
    "embeddings": "300d",
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
}

SWEEP_EPOCHS = 8
SWEEP_BATCH_SIZE = 32
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SentimentDatasetDAN(Dataset):
    """Word-index sentences paired with 0/1 labels. Unknown words map to UNK."""

    def __init__(self, infile, word_embeddings):
        self.examples = read_sentiment_examples(infile)
        indexer = word_embeddings.word_indexer
        unk_idx = indexer.index_of("UNK")
        self.sentences = []
        self.labels = []
        for ex in self.examples:
            indices = []
            for word in ex.words:
                idx = indexer.index_of(word)
                indices.append(unk_idx if idx == -1 else idx)
            self.sentences.append(indices)
            self.labels.append(ex.label)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


def collate_batch(batch):
    """Pad word-index sequences with PAD (index 0) so they can be batched."""
    sentences, labels = zip(*batch)
    max_len = max((len(sentence) for sentence in sentences), default=1)
    max_len = max(max_len, 1)
    padded = torch.zeros(len(sentences), max_len, dtype=torch.long)
    for i, sentence in enumerate(sentences):
        if sentence:
            padded[i, :len(sentence)] = torch.tensor(sentence, dtype=torch.long)
    return padded, torch.tensor(labels, dtype=torch.long)


class DAN(nn.Module):
    """
    Average word embeddings, then a feedforward classifier.

    num_hidden_layers is the number of ReLU/Tanh layers before the 2-class
    output. One hidden layer is Linear(embed, hidden) -> activation -> output.
    Two and three hidden layers stack additional Linear(hidden, hidden) blocks.
    """

    def __init__(
        self,
        word_embeddings,
        num_hidden_layers,
        hidden_size,
        dropout=0.3,
        nonlinearity="relu",
        init="default",
        dropout_on_embedding=True,
        dropout_on_hidden=True,
        random_embeddings=False,
        random_dim=300,
    ):
        super().__init__()
        if nonlinearity not in ("relu", "tanh"):
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        if init not in ("default", "xavier"):
            raise ValueError(f"Unsupported init: {init}")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        self.pad_idx = word_embeddings.word_indexer.index_of("PAD")
        if random_embeddings:
            vocab_size = len(word_embeddings.word_indexer)
            self.embedding = nn.Embedding(vocab_size, random_dim, padding_idx=self.pad_idx)
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            with torch.no_grad():
                self.embedding.weight[self.pad_idx].zero_()
            embed_dim = random_dim
        else:
            # frozen=False fine-tunes the GloVe vectors during training.
            self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=False)
            embed_dim = word_embeddings.get_embedding_length()

        drop_p = dropout if dropout_on_embedding else 0.0
        self.embed_dropout = nn.Dropout(drop_p)

        hidden = []
        in_dim = embed_dim
        for _ in range(num_hidden_layers):
            hidden.append(nn.Linear(in_dim, hidden_size))
            hidden.append(nn.ReLU() if nonlinearity == "relu" else nn.Tanh())
            if dropout_on_hidden and dropout > 0:
                hidden.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.hidden = nn.Sequential(*hidden)
        self.output = nn.Linear(in_dim, 2)

        if init == "xavier":
            for module in list(self.hidden) + [self.output]:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, word_indices):
        # word_indices: (batch, seq_len)
        mask = (word_indices != self.pad_idx).unsqueeze(-1).float()
        embedded = self.embedding(word_indices) * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        averaged = embedded.sum(dim=1) / lengths
        hidden = self.hidden(self.embed_dropout(averaged))
        return self.output(hidden)


def _accuracy_and_loss(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for word_indices, labels in data_loader:
            logits = model(word_indices)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / total


def train_dan(model, train_loader, dev_loader, optimizer, epochs, log_every=1):
    """Train with CrossEntropyLoss on logits. Returns the best dev checkpoint stats."""
    loss_fn = nn.CrossEntropyLoss()
    best = {
        "dev_accuracy": -1.0,
        "dev_loss": None,
        "train_accuracy": None,
        "train_loss": None,
        "epoch": 0,
    }
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for word_indices, labels in train_loader:
            optimizer.zero_grad()
            logits = model(word_indices)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        train_accuracy = correct / total
        train_loss = total_loss / total
        dev_accuracy, dev_loss = _accuracy_and_loss(model, dev_loader, loss_fn)
        history.append((train_accuracy, train_loss, dev_accuracy, dev_loss))
        if dev_accuracy > best["dev_accuracy"]:
            best = {
                "dev_accuracy": dev_accuracy,
                "dev_loss": dev_loss,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "epoch": epoch,
            }
        if log_every and epoch % log_every == 0:
            print(
                f"Epoch #{epoch}: train accuracy {train_accuracy:.3f}, "
                f"train loss {train_loss:.3f}, dev accuracy {dev_accuracy:.3f}, "
                f"dev loss {dev_loss:.3f}"
            )
    return best, history


def make_optimizer(model, config):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"])
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def build_model(word_embeddings, config):
    random_embeddings = config["embeddings"] == "random"
    return DAN(
        word_embeddings,
        num_hidden_layers=config["num_hidden_layers"],
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        nonlinearity=config["nonlinearity"],
        init=config["init"],
        dropout_on_embedding=config["dropout_on_embedding"],
        dropout_on_hidden=config["dropout_on_hidden"],
        random_embeddings=random_embeddings,
        random_dim=config.get("random_dim", 300),
    )


def load_split(word_embeddings):
    train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
    dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
    return train_data, dev_data


def run_config(config, prepared, seed=SEED, log_every=0):
    """Train one configuration from scratch and return a result row."""
    set_seed(seed)
    source = config["embeddings"]
    if source == "random":
        source = "300d" if config.get("random_dim", 300) == 300 else "50d"
    word_embeddings, train_data, dev_data = prepared[source]
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_batch,
    )
    model = build_model(word_embeddings, config)
    optimizer = make_optimizer(model, config)
    start = time.time()
    best, history = train_dan(
        model, train_loader, dev_loader, optimizer, config["epochs"], log_every=log_every
    )
    elapsed = time.time() - start
    final_train, final_train_loss, final_dev, final_dev_loss = history[-1]
    result = dict(config)
    result.update(best)
    result["final_train_accuracy"] = final_train
    result["final_dev_accuracy"] = final_dev
    result["final_train_loss"] = final_train_loss
    result["final_dev_loss"] = final_dev_loss
    result["seconds"] = elapsed
    return result


def phase1_configs():
    """Grid: depth x width x embedding size x dropout on/off."""
    configs = []
    for embeddings in ("50d", "300d"):
        for num_hidden_layers in (1, 2, 3):
            for hidden_size in (100, 300):
                for dropout in (0.0, 0.3):
                    configs.append({
                        "name": "phase1",
                        "embeddings": embeddings,
                        "num_hidden_layers": num_hidden_layers,
                        "hidden_size": hidden_size,
                        "dropout": dropout,
                        "dropout_on_embedding": True,
                        "dropout_on_hidden": True,
                        "nonlinearity": "relu",
                        "init": "default",
                        "optimizer": "adam",
                        "lr": 0.001,
                        "epochs": SWEEP_EPOCHS,
                        "batch_size": SWEEP_BATCH_SIZE,
                    })
    return configs


def phase2_configs(best):
    """One-factor changes on the Phase 1 winner, plus random embeddings (1b)."""
    base = {
        "name": "phase2",
        "embeddings": best["embeddings"],
        "num_hidden_layers": best["num_hidden_layers"],
        "hidden_size": best["hidden_size"],
        "dropout": best["dropout"],
        "dropout_on_embedding": best["dropout_on_embedding"],
        "dropout_on_hidden": best["dropout_on_hidden"],
        "nonlinearity": best["nonlinearity"],
        "init": best["init"],
        "optimizer": best["optimizer"],
        "lr": best["lr"],
        "epochs": SWEEP_EPOCHS,
        "batch_size": SWEEP_BATCH_SIZE,
    }
    random_dim = 300 if best["embeddings"] == "300d" else 50
    followups = [
        ("tanh", {"nonlinearity": "tanh"}),
        ("dropout-embed-only", {
            "dropout": 0.3,
            "dropout_on_embedding": True,
            "dropout_on_hidden": False,
        }),
        ("dropout-hidden-only", {
            "dropout": 0.3,
            "dropout_on_embedding": False,
            "dropout_on_hidden": True,
        }),
        ("xavier", {"init": "xavier"}),
        ("adam-lr-1e-4", {"optimizer": "adam", "lr": 0.0001}),
        ("sgd-lr-0.1", {"optimizer": "sgd", "lr": 0.1}),
        ("random-embeddings", {"embeddings": "random", "random_dim": random_dim}),
    ]
    configs = []
    for name, updates in followups:
        config = dict(base)
        config.update(updates)
        config["name"] = name
        configs.append(config)
    return configs


def _describe(config):
    dropout_where = []
    if config["dropout"] > 0 and config["dropout_on_embedding"]:
        dropout_where.append("emb")
    if config["dropout"] > 0 and config["dropout_on_hidden"]:
        dropout_where.append("hid")
    where = "+".join(dropout_where) if dropout_where else "none"
    return (
        f"{config['embeddings']:6} L={config['num_hidden_layers']} "
        f"H={config['hidden_size']:<3} drop={config['dropout']:.1f}({where}) "
        f"{config['nonlinearity']:4} init={config['init']:7} "
        f"{config['optimizer']} lr={config['lr']}"
    )


def _print_result(index, total, result):
    print(
        f"[{index}/{total}] {_describe(result)} | "
        f"best dev {result['dev_accuracy']:.3f} @ epoch {result['epoch']} "
        f"(train {result['train_accuracy']:.3f}) | "
        f"final dev {result['final_dev_accuracy']:.3f} | "
        f"{result['seconds']:.1f}s"
    )


RESULT_FIELDS = [
    "name",
    "embeddings",
    "num_hidden_layers",
    "hidden_size",
    "dropout",
    "dropout_on_embedding",
    "dropout_on_hidden",
    "nonlinearity",
    "init",
    "optimizer",
    "lr",
    "epochs",
    "epoch",
    "train_accuracy",
    "train_loss",
    "dev_accuracy",
    "dev_loss",
    "final_train_accuracy",
    "final_dev_accuracy",
    "seconds",
]


def _write_results(path, results):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def prepare_data():
    prepared = {}
    for name, path in EMBEDDING_FILES.items():
        word_embeddings = read_word_embeddings(path)
        prepared[name] = (word_embeddings, *load_split(word_embeddings))
    return prepared


def run_sweep():
    prepared = prepare_data()
    phase1 = phase1_configs()
    results = []
    print(f"\nPhase 1: {len(phase1)} runs, {SWEEP_EPOCHS} epochs, CrossEntropyLoss, Adam lr=0.001")
    for index, config in enumerate(phase1, start=1):
        result = run_config(config, prepared)
        results.append(result)
        _print_result(index, len(phase1), result)

    winner = max(results, key=lambda row: row["dev_accuracy"])
    print(f"\nPhase 1 winner: {_describe(winner)} dev {winner['dev_accuracy']:.3f}")

    followups = phase2_configs(winner)
    print(f"\nPhase 2: {len(followups)} one-factor runs on that architecture")
    for index, config in enumerate(followups, start=1):
        result = run_config(config, prepared)
        results.append(result)
        _print_result(index, len(followups), result)

    _write_results("dan_sweep_results.csv", results)
    print("\nSweep results written to dan_sweep_results.csv")
    print("Best dev accuracy overall: "
          f"{max(row['dev_accuracy'] for row in results):.3f}")
    print("Use the Phase 1 winner as FINAL_CONFIG for python main.py --model DAN.")
    return results


def train_final_dan():
    """Train and evaluate the single DAN submitted for part 1a."""
    print("\nDAN config:", _describe(FINAL_CONFIG))
    prepared = prepare_data()
    result = run_config(FINAL_CONFIG, prepared, log_every=1)
    print(
        f"\nBest dev accuracy {result['dev_accuracy']:.3f} at epoch {result['epoch']} "
        f"(train accuracy {result['train_accuracy']:.3f})"
    )
    print(f"Training time: {result['seconds']:.1f} seconds")
    return result
