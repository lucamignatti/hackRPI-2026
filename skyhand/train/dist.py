"""Data-parallel training across GPUs and nodes (torch.distributed, gradient averaging).

Every process (rank) runs the full trainer on its own share of the worlds with its own replay
buffer. All ranks start from identical weights (broadcast from rank 0) and, at every optimizer
step, average their gradients before stepping, so every rank holds the same model at all times:
one model trained on world_size x batch_size samples per update. Only rank 0 logs and saves.

Launch: one process per GPU (torchrun, mp.spawn, or any launcher) with the standard
torch.distributed env vars WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR and MASTER_PORT.
With a single process nothing changes.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist


def _env_int(*names: str, default: int) -> int:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return int(v)
    return default


def init_from_env(device: str = "cuda") -> Tuple[int, int, int]:
    """Start the process group when more than one process was launched. Returns (rank, world, local)."""
    world = _env_int("WORLD_SIZE", default=1)
    if world <= 1:
        return 0, 1, 0
    rank = _env_int("RANK", default=0)
    local = _env_int("LOCAL_RANK", default=0)
    if not dist.is_initialized():
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local % torch.cuda.device_count())
        dist.init_process_group("nccl" if use_cuda else "gloo", init_method="env://",
                                rank=rank, world_size=world)
    return rank, world, local


def active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if active() else 0


def world_size() -> int:
    return dist.get_world_size() if active() else 1


def is_main() -> bool:
    return rank() == 0


@torch.no_grad()
def broadcast_module(module: torch.nn.Module) -> None:
    """Copy rank 0's parameters and buffers to every rank."""
    if not active():
        return
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=0)


@torch.no_grad()
def allreduce_grads(params: Iterable[torch.nn.Parameter]) -> None:
    """Average the gradients of `params` across ranks in one flat all-reduce. A parameter without a
    gradient on this rank counts as zero, so every rank reduces the same layout."""
    if not active():
        return
    ps: List[torch.nn.Parameter] = [p for p in params]
    if not ps:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())
    i = 0
    for p in ps:
        n = p.numel()
        g = flat[i:i + n].view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        i += n


def barrier() -> None:
    if active():
        dist.barrier()
