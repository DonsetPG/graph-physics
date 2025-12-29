#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="training_config/entropy_sweep.json"
OUT_DIR="training_config/entropy_sweep_generated"
PROJECT_NAME="entropy_sweep"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DISABLE_PERSISTENT_WORKERS="${DISABLE_PERSISTENT_WORKERS:-1}"
EP_NUM_TRAJ="${EP_NUM_TRAJ:-8}"
EP_MAX_ITER="${EP_MAX_ITER:-200}"
EP_ORDERS="${EP_ORDERS:-node,edge,1hop,2hop}"
EP_MAX_NODES="${EP_MAX_NODES:-1024}"
EP_MAX_EDGES="${EP_MAX_EDGES:-2048}"

mkdir -p "$OUT_DIR"

layers=(1 2 3 4 5)
widths=(8 16 32 64)

for L in "${layers[@]}"; do
  for W in "${widths[@]}"; do
    cfg="${OUT_DIR}/entropy_L${L}_W${W}.json"
    run_id="layers${L}_hidden${W}"

    EP_NUM_TRAJ="$EP_NUM_TRAJ" \
    EP_MAX_ITER="$EP_MAX_ITER" \
    EP_ORDERS="$EP_ORDERS" \
    EP_MAX_NODES="$EP_MAX_NODES" \
    EP_MAX_EDGES="$EP_MAX_EDGES" \
    python - "$BASE_CONFIG" "$cfg" "$L" "$W" <<'PY'
import json
import sys
import os
from pathlib import Path

base, out, layers, width = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(base, "r") as f:
    cfg = json.load(f)

cfg["model"]["message_passing_num"] = layers
cfg["model"]["hidden_size"] = width
cfg.setdefault("experiment", {})
cfg["experiment"]["tag"] = f"layers{layers}_hidden{width}"

ep = cfg.setdefault("diagnostics", {}).setdefault("entropy_production", {})
ep["num_trajectories"] = int(os.environ.get("EP_NUM_TRAJ", ep.get("num_trajectories", 8)))
ep["max_iter"] = int(os.environ.get("EP_MAX_ITER", ep.get("max_iter", 200)))
orders = os.environ.get("EP_ORDERS", "")
if orders:
    ep["orders"] = [o.strip() for o in orders.split(",") if o.strip()]
ep["max_nodes_per_traj"] = int(os.environ.get("EP_MAX_NODES", ep.get("max_nodes_per_traj", 1024)))
ep["max_edges_per_traj"] = int(os.environ.get("EP_MAX_EDGES", ep.get("max_edges_per_traj", 2048)))

Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
PY

    echo "==> Training ${run_id} (config: ${cfg})"
    if ! GRAPH_PHYSICS_DISABLE_PERSISTENT_WORKERS="$DISABLE_PERSISTENT_WORKERS" \
      WANDB_NAME="${run_id}" \
      python -m graphphysics.train \
      --training_parameters_path="$cfg" \
      --num_epochs=1 \
      --init_lr=0.001 \
      --batch_size=2 \
      --warmup=500 \
      --num_workers="$NUM_WORKERS" \
      --prefetch_factor="$PREFETCH_FACTOR" \
      --model_save_name="ep_L${L}_W${W}" \
      --project_name="$PROJECT_NAME" \
      --no_edge_feature; then
        echo "!! FAILED: ${run_id} (continuing)"
    else
        echo "<== Done ${run_id}"
    fi
  done
done
