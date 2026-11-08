"""Data-parallel training: ranks see different worlds but must end up with identical weights."""
from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, out_dir: str) -> None:
    os.environ.update({"WORLD_SIZE": str(world), "RANK": str(rank), "LOCAL_RANK": str(rank),
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    from skyhand.train import dist as D
    from skyhand.train.config import tiny_config
    from skyhand.train.train import Trainer
    D.init_from_env("cpu")
    tr = Trainer(tiny_config(out=os.path.join(out_dir, "run"), steps=6, updates_per_step=2))
    tr.run()
    flat = torch.cat([p.detach().reshape(-1) for p in tr.agent.learner.parameters()])
    torch.save({"params": flat, "updates": tr.agent.learner.num_updates,
                "obs0": tr.streams[0].obs["proprio"].clone()}, os.path.join(out_dir, f"rank{rank}.pt"))
    torch.distributed.destroy_process_group()


def test_two_ranks_stay_in_lockstep(tmp_path):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{i}.pt") for i in range(2))
    assert r0["updates"] == r1["updates"] > 0
    assert not torch.equal(r0["obs0"], r1["obs0"])                       # different worlds per rank
    assert torch.equal(r0["params"], r1["params"])                       # yet the same model
    assert (tmp_path / "run" / "agent.pt").exists()                      # rank 0 saved


def test_single_process_is_unchanged():
    from skyhand.train import dist as D
    assert not D.active() and D.rank() == 0 and D.world_size() == 1 and D.is_main()
