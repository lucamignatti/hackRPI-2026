"""Tiny distilled text -> z model (no downloads).

Features: hashed word unigrams, word bigrams and character trigrams (stable crc32 hashing)
-> mean-pooled EmbeddingBag -> 2-layer MLP -> z on the sqrt(d) sphere. Trained with cosine
loss on (text, z) pairs, where z comes from compile_mission + reward_to_z. It must be
retrained whenever B changes.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens(text: str) -> List[str]:
    t = text.lower()
    words = re.findall(r"[a-z0-9]+", t)
    feats = [f"w:{w}" for w in words]
    feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    for w in words:
        p = f"<{w}>"
        feats += [f"c:{p[i:i + 3]}" for i in range(len(p) - 2)]
    return feats or ["<empty>"]


def featurize(text: str, n_buckets: int) -> List[int]:
    return [zlib.crc32(f.encode()) % n_buckets for f in _tokens(text)]


class TextToZ(nn.Module):
    def __init__(self, d: int, n_buckets: int = 1 << 14, emb_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.d, self.n_buckets, self.emb_dim, self.hidden = d, n_buckets, emb_dim, hidden
        self.emb = nn.EmbeddingBag(n_buckets, emb_dim, mode="mean")
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def _batch(self, texts: Sequence[str]):
        idx, offs = [], []
        for t in texts:
            offs.append(len(idx))
            idx += featurize(t, self.n_buckets)
        dev = self.emb.weight.device
        return torch.tensor(idx, dtype=torch.long, device=dev), torch.tensor(offs, dtype=torch.long, device=dev)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        idx, offs = self._batch(texts)
        z = self.mlp(self.emb(idx, offs))
        return math.sqrt(self.d) * F.normalize(z, dim=-1)

    def fit(self, pairs: Sequence[Tuple[str, torch.Tensor]], epochs: int = 300, lr: float = 3e-3,
            batch_size: int = 64, weight_decay: float = 1e-5, seed: int = 0, verbose: bool = False) -> List[float]:
        """Train on (text, z) pairs with cosine loss. Returns per-epoch mean loss."""
        texts = [t for t, _ in pairs]
        Z = torch.stack([torch.as_tensor(z, dtype=torch.float32).reshape(self.d) for _, z in pairs]).to(
            self.emb.weight.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        g = torch.Generator().manual_seed(seed)
        hist = []
        self.train()
        for ep in range(epochs):
            perm = torch.randperm(len(texts), generator=g)
            tot = 0.0
            for i in range(0, len(texts), batch_size):
                b = perm[i:i + batch_size].tolist()
                pred = self([texts[j] for j in b])
                loss = (1 - F.cosine_similarity(pred, Z[b], dim=-1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(b)
            hist.append(tot / len(texts))
            if verbose and ep % 50 == 0:
                print(f"text2z epoch {ep}: loss {hist[-1]:.4f}")
        self.eval()
        return hist

    @torch.no_grad()
    def predict(self, text: str) -> torch.Tensor:
        """z [d] for one mission text."""
        self.eval()
        return self([text])[0]

    def save(self, path: str) -> None:
        torch.save({"config": {"d": self.d, "n_buckets": self.n_buckets, "emb_dim": self.emb_dim,
                               "hidden": self.hidden}, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TextToZ":
        ck = torch.load(path, map_location=map_location, weights_only=True)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state"])
        m.eval()
        return m


def build_dataset(missions: Iterable[str], compile_fn: Callable[[str], object],
                  reward_to_z_fn: Callable[[object], torch.Tensor],
                  skip_errors: bool = True) -> List[Tuple[str, torch.Tensor]]:
    """(text, z) pairs: z = reward_to_z_fn(compile_fn(text)).

    e.g. compile_fn=lambda t: compile_mission(t, scene, ents),
         reward_to_z_fn=lambda ms: mission_z(ms, agent.encode_g, g_replay).
    """
    out = []
    for t in missions:
        try:
            ms = compile_fn(t)
            z = reward_to_z_fn(ms)
        except Exception:
            if not skip_errors:
                raise
            continue
        out.append((t, torch.as_tensor(z, dtype=torch.float32).detach().cpu()))
    return out
