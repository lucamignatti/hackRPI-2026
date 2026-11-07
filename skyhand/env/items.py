"""Graspable item catalog as tensors (built from spec.ITEM_CATALOG)."""
from __future__ import annotations

import torch

from .. import spec

HANDLE_HEIGHT = 0.20     # rope loop above the item's top, where the gripper grabs


class ItemCatalog:
    def __init__(self, device=None):
        cat = spec.ITEM_CATALOG
        self.n_types = len(cat)
        self.index = {c[0]: i for i, c in enumerate(cat)}
        self.sphere = torch.tensor([c[1] == "sphere" for c in cat], device=device)
        half = []
        for c in cat:
            if c[1] == "sphere":
                r = c[2][0]
                half.append((r, r, r))
            else:
                half.append(tuple(x / 2 for x in c[2]))
        self.half = torch.tensor(half, dtype=torch.float32, device=device)     # [T, 3]
        self.mass = torch.tensor([c[3] for c in cat], dtype=torch.float32, device=device)
        self.is_trash = torch.tensor([c[4] for c in cat], device=device)
        self.trash_ids = torch.tensor([i for i, c in enumerate(cat) if c[4]], device=device)
        self.other_ids = torch.tensor([i for i, c in enumerate(cat) if not c[4] and c[0] != "cooler"],
                                      device=device)

    def id(self, name: str) -> int:
        return self.index[name]
