"""Minimal-dependency training.

The training path must import and run with only torch and numpy installed:
no Genesis, google-genai, fastapi, or pydantic. This test hides them and runs
a few training iterations on the pure-torch backend in a subprocess.
"""
import subprocess
import sys
import textwrap


def test_training_runs_without_genesis_or_web_deps(tmp_path):
    code = textwrap.dedent(f"""
        import importlib.abc, sys
        BLOCK = ("genesis", "quadrants", "google", "fastapi", "uvicorn", "pydantic")
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ImportError("blocked for minimal-deps test: " + name)
                return None
        sys.meta_path.insert(0, Blocker())
        import torch
        from skyhand.env.core import make_env
        from skyhand.train import train as T
        env = make_env("ocean", 2, lidar=True)
        env.step(torch.zeros(2, env.num_agents, 5))
        sys.argv = ["train", "--scene", "both", "--backend", "torch", "--num-envs", "2", "--device", "cpu",
                    "--steps", "3", "--out", r"{tmp_path}/run", "--tiny", "--warm-steps", "4", "--batch-size", "16"]
        T.main()
        import os
        assert os.path.exists(r"{tmp_path}/run/agent.pt")
        print("MINIMAL_OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert "MINIMAL_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
