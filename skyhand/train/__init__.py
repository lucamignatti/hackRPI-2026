from __future__ import annotations


def load(path, device="cpu"):
    """Load a trained Agent checkpoint (shortcut for skyhand.train.agent.Agent.load)."""
    from .agent import Agent
    return Agent.load(path, device=device)
