# Looped Transformer Operator Brief

## How To Use This File

Use this same file for both phases of the campaign.

When you launch an autonomous agent, tell it which role it has:

- `phase=ablation_owner`
- `phase=scaling_worker shard_index=0`
- `phase=scaling_worker shard_index=1`
- `phase=scaling_worker shard_index=2`

Do not invent a different workflow. Follow the role exactly.

## Global Rules

- Start from the latest `origin/loop`.
- Use a separate worktree and branch for your local work.
- Exact research claims require a **DGL sparse** runtime.
- If `dgl.sparse` does not import, stop and report the blocker.
- Use:
  - `batch_size=2`
  - `num_workers=2`
  - `init_lr=0.001`
  - `warmup=1000`
- Smoke runs use `1` epoch.
- Ablation and scaling runs use `20` epochs.
- The train-loss guard is:
  - baseline = epoch-10 primary train loss from ablation stage `1`, seed `42`
  - kill if current primary train loss at epoch `10+` is `>= 1.2 * baseline`

## Runtime Check

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
import torch_geometric
print("pyg", torch_geometric.__version__)
import dgl.sparse as dglsp
print("dgl sparse ok", dglsp is not None)
PY
```

## Targeted Validation

```bash
python -m py_compile \
  graphphysics/models/layers.py \
  graphphysics/models/processors.py \
  graphphysics/models/simulator.py \
  graphphysics/training/parse_parameters.py \
  graphphysics/training/lightning_module.py \
  graphphysics/train.py \
  graphphysics/experiments/looped_ablation_ladder.py \
  graphphysics/experiments/run_guarded_training.py
```

```bash
PYTHONPATH=$PWD python -m pytest -q \
  tests/graphphysics/models/test_layers.py \
  tests/graphphysics/models/test_processors.py \
  tests/graphphysics/training/test_parameters.py
```

If these fail:

- only the `phase=ablation_owner` agent may patch code
- scaling workers must stop and wait for a pushed fix

## Phase A: `phase=ablation_owner`

You own:

1. baseline transformer smoke
2. looped-transformer smoke
3. the full 9-stage ablation ladder
4. the baseline epoch-10 loss publication
5. the best-architecture export for scaling

### Smoke

Run the baseline smoke:

```bash
WANDB_NAME=ablation_owner_transformer_smoke \
PYTHONPATH=$PWD python -m graphphysics.train \
  --training_parameters_path=training_config/coarse-aneurysm.json \
  --project_name=graphphysics-looped-smoke \
  --num_epochs=1 \
  --batch_size=2 \
  --init_lr=0.001 \
  --warmup=1000 \
  --num_workers=2
```

Run the looped smoke:

```bash
WANDB_NAME=ablation_owner_looped_smoke \
PYTHONPATH=$PWD python -m graphphysics.train \
  --training_parameters_path=training_config/coarse-aneurysm-looped.json \
  --project_name=graphphysics-looped-smoke \
  --num_epochs=1 \
  --batch_size=2 \
  --init_lr=0.001 \
  --warmup=1000 \
  --num_workers=2
```

Do not start the ablation ladder until both smoke runs are visible in WandB under `graphphysics-looped-smoke`.

### Ablation Ladder

Use only seed `42`.

Stages:

1. `baseline_transformer`
2. `looped_core`
3. `stable_injection`
4. `loop_embedding`
5. `loop_sampling`
6. `moe_ffn`
7. `act_halting`
8. `target_active_adjacency`
9. `edge_gate_adjacency`

Maximum ablation budget: `9` runs.

Rules:

- stage `1` is reference-only
- stage `1` seed `42` defines `BASELINE_EPOCH10_TRAIN_LOSS`
- stage `2` initializes the looped incumbent
- stages `3` to `9` are accepted only if their single-run `val_1step_rmse` is strictly lower than the incumbent metric
- if a stage is killed by the loss guard, reject it immediately
- if `moe_ffn` is rejected, continue from the last accepted incumbent and do not carry MoE forward

Build a stage config:

```bash
python -m graphphysics.experiments.looped_ablation_ladder build-stage \
  --stage-index <STAGE_INDEX> \
  --seed-index 0 \
  --out .codex/experiments/ablation/stage<STAGE_INDEX>_seed42.json
```

Run stage `1` without the loss guard:

```bash
python -m graphphysics.experiments.run_guarded_training \
  --training-parameters-path .codex/experiments/ablation/stage01_seed42.json \
  --project-name graphphysics-looped-ablation \
  --num-epochs 20 \
  --batch-size 2 \
  --init-lr 0.001 \
  --warmup 1000 \
  --num-workers 2 \
  --seed 42 \
  --wandb-name s01_baseline_transformer_seed42 \
  --epoch-metrics-path .codex/experiments/ablation/stage01_seed42.epochs.jsonl \
  --log-path .codex/experiments/ablation/stage01_seed42.log \
  --result-json .codex/experiments/ablation/stage01_seed42.result.json
```

After stage `1`:

- read epoch `10` from `.codex/experiments/ablation/stage01_seed42.epochs.jsonl`
- take `primary_train_metric` as `BASELINE_EPOCH10_TRAIN_LOSS`
- record it in `experiment_state/looped_transformer_ablation_state.json`
- append it to `.codex/notes/research-log.md`

Record stage `1`:

```bash
python -m graphphysics.experiments.looped_ablation_ladder record-decision \
  --stage-index 1 \
  --decision reference \
  --metric-mean <STAGE1_VAL_1STEP_RMSE> \
  --baseline-epoch10-train-loss <BASELINE_EPOCH10_TRAIN_LOSS> \
  --notes "Baseline transformer reference recorded from stage 1 seed 42"
```

Run later stages with the loss guard:

```bash
python -m graphphysics.experiments.run_guarded_training \
  --training-parameters-path .codex/experiments/ablation/stage<STAGE_INDEX>_seed42.json \
  --project-name graphphysics-looped-ablation \
  --num-epochs 20 \
  --batch-size 2 \
  --init-lr 0.001 \
  --warmup 1000 \
  --num-workers 2 \
  --seed 42 \
  --wandb-name s<STAGE_INDEX>_<STAGE_NAME>_seed42 \
  --epoch-metrics-path .codex/experiments/ablation/stage<STAGE_INDEX>_seed42.epochs.jsonl \
  --log-path .codex/experiments/ablation/stage<STAGE_INDEX>_seed42.log \
  --result-json .codex/experiments/ablation/stage<STAGE_INDEX>_seed42.result.json \
  --loss-guard-baseline <BASELINE_EPOCH10_TRAIN_LOSS> \
  --loss-guard-epoch 10 \
  --loss-guard-max-ratio 1.2
```

Record stage `2`:

```bash
python -m graphphysics.experiments.looped_ablation_ladder record-decision \
  --stage-index 2 \
  --decision reset \
  --metric-mean <STAGE2_VAL_1STEP_RMSE> \
  --notes "Looped incumbent initialized"
```

Record later stages:

```bash
python -m graphphysics.experiments.looped_ablation_ladder record-decision \
  --stage-index <STAGE_INDEX> \
  --decision accept \
  --metric-mean <CANDIDATE_VAL_1STEP_RMSE> \
  --notes "Accepted over incumbent"
```

```bash
python -m graphphysics.experiments.looped_ablation_ladder record-decision \
  --stage-index <STAGE_INDEX> \
  --decision reject \
  --metric-mean <CANDIDATE_VAL_1STEP_RMSE> \
  --notes "Rejected; incumbent kept"
```

### Export Best Architecture For Scaling

After the final ablation decision:

```bash
python -m graphphysics.experiments.looped_ablation_ladder build-incumbent \
  --out experiment_state/looped_transformer_best_architecture.json
```

Push `experiment_state/looped_transformer_best_architecture.json` before scaling starts.

## Phase B: `phase=scaling_worker shard_index=<0|1|2>`

You own only one scaling shard.

Before scaling:

1. confirm both smoke runs are visible in WandB under `graphphysics-looped-smoke`
2. fetch latest `origin/loop`
3. confirm `experiment_state/looped_transformer_best_architecture.json` exists on the branch
4. read `BASELINE_EPOCH10_TRAIN_LOSS` from `.codex/notes/research-log.md`

```bash
git fetch origin
git pull --ff-only origin loop
```

Plan the scaling sweep:

```bash
python .codex/scripts/architecture_sweep.py plan \
  --base-config experiment_state/looped_transformer_best_architecture.json \
  --grid .codex/configs/looped_transformer_scaling.json \
  --out-dir .codex/experiments/sweeps/$(date +%Y%m%d_%H%M%S)_looped_scaling_shard<SHARD_INDEX>
```

Run your shard:

```bash
python .codex/scripts/architecture_sweep.py run-shard \
  --manifest <SCALING_MANIFEST> \
  --shard-index <SHARD_INDEX> \
  --num-shards 3 \
  --python python \
  --loss-guard-baseline <BASELINE_EPOCH10_TRAIN_LOSS> \
  --loss-guard-epoch 10 \
  --loss-guard-max-ratio 1.2
```

Do not substitute `training_config/coarse-aneurysm-looped.json` for scaling. Always use:

- `experiment_state/looped_transformer_best_architecture.json`

That is the accepted winner from the ablation ladder and is the only valid scaling base config.
