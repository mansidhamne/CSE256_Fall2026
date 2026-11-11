"""Transformer components built from elementary PyTorch operations (Parts 1–2)."""
import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("Embedding size must be divisible by head count")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x, valid, causal=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.n_head, self.head_size)
                   .transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        # Padding keys cannot receive attention. Encoder attention is bidirectional.
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        if causal:
            future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        attention = scores.softmax(dim=-1)
        context = (attention @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.projection(context), attention


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        # The handout does not specify the encoder FFN width; use 4 * embedding.
        self.feedforward = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                         nn.Linear(4 * n_embd, n_embd))

    def forward(self, x, valid):
        context, attention = self.attention(self.ln1(x), valid)
        x = x + context
        return x + self.feedforward(self.ln2(x)), attention


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, block_size=32, n_embd=64, n_head=2, n_layer=4):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([EncoderBlock(n_embd, n_head) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(n_embd)

    def forward(self, ids):
        if ids.shape[1] > self.block_size:
            raise ValueError("Input exceeds block_size")
        valid = ids.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError("Each input must contain at least one non-padding token")
        x = self.token_embedding(ids) + self.position_embedding(
            torch.arange(ids.shape[1], device=ids.device))
        maps = []
        for block in self.blocks:
            x, attention = block(x, valid)
            # Utilities accepts a list of maps with shape (batch, query, key).
            maps.extend(attention.unbind(dim=1))
        return self.final_norm(x), maps


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, n_embd=64, n_hidden=100, n_output=3):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(n_embd, n_hidden), nn.ReLU(),
                                        nn.Linear(n_hidden, n_output))

    def forward(self, ids):
        vectors, _ = self.encoder(ids)
        valid = ids.ne(0).unsqueeze(-1)
        # Mean over words only: padding contributes neither values nor count.
        mean = (vectors * valid).sum(dim=1) / valid.sum(dim=1)
        return self.classifier(mean)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.feedforward = nn.Sequential(nn.Linear(n_embd, 100), nn.ReLU(),
                                         nn.Linear(100, n_embd))

    def forward(self, x, valid):
        context, attention = self.attention(self.ln1(x), valid, causal=True)
        x = x + context
        return x + self.feedforward(self.ln2(x)), attention


class TransformerDecoder(nn.Module):
    """GPT-like word LM. Return (logits, maps), or scalar loss if targets supplied."""
    def __init__(self, vocab_size, block_size=32, n_embd=64, n_head=2, n_layer=4):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, ids, targets=None):
        if ids.shape[1] > self.block_size:
            raise ValueError("Input exceeds block_size")
        valid = ids.ne(0)
        if not valid[:, 0].all():
            raise ValueError("Decoder inputs must start with a non-padding token")
        x = self.token_embedding(ids) + self.position_embedding(
            torch.arange(ids.shape[1], device=ids.device))
        maps = []
        for block in self.blocks:
            x, attention = block(x, valid)
            maps.extend(attention.unbind(dim=1))
        logits = self.lm_head(self.final_norm(x))
        if targets is not None:
            return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               targets.reshape(-1), ignore_index=-100)
        return logits, maps
