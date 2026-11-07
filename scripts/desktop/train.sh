#!/bin/bash
# Desktop (Ryzen 7900X + RTX 5080). Torch backend is the fast path; Genesis is for
# checking the sim-to-sim gap and for the demo.
set -euo pipefail
BACKEND=${BACKEND:-torch}
SCENE=${SCENE:-both}
python -m skyhand.train.train --scene "$SCENE" --backend "$BACKEND" --device cuda \
  --num-envs ${NUM_ENVS:-2048} --steps ${STEPS:-200000} --out runs/desktop-$(date +%m%d-%H%M) "$@"
