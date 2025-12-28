#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="training_config/entropy_sweep.json"
OUT_DIR="training_config/entropy_sweep_generated"
PROJECT_NAME="entropy_sweep"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DISABLE_PERSISTENT_WORKERS="${DISABLE_PERSISTENT_WORKERS:-1}"

mkdir -p "$OUT_DIR"

layers=(1 2 3)
widths=(16 32 64 128)

for L in "${layers[@]}"; do
  for W in "${widths[@]}"; do
    cfg="${OUT_DIR}/entropy_L${L}_W${W}.json"
    run_id="L${L}_W${W}"

    python - "$BASE_CONFIG" "$cfg" "$L" "$W" <<'PY'
import json
import sys
from pathlib import Path

base, out, layers, width = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(base, "r") as f:
    cfg = json.load(f)

cfg["model"]["message_passing_num"] = layers
cfg["model"]["hidden_size"] = width
cfg.setdefault("experiment", {})
cfg["experiment"]["tag"] = f"L{layers}_W{width}"

Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(cfg, f, indent=2)
PY

    echo "==> Training ${run_id} (config: ${cfg})"
    if ! GRAPH_PHYSICS_DISABLE_PERSISTENT_WORKERS="$DISABLE_PERSISTENT_WORKERS" \
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
