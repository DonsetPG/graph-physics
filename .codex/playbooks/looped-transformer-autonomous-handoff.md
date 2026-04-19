# Looped Transformer Autonomous Handoff

## Objective

Take the existing PyTorch-first `looped_transformer` implementation from smoke-test stage to evidence-backed experiment stage.

Work from:

- repo: `/Users/paulgarnier/github/phd/graph-physics`
- remote branch: `origin/loop`
- current reference commit: `ffd6fca` (`Add looped graph transformer scaffold`)

The main goal is **not** to redesign the model. The architecture scaffold is already implemented. The main goal is to:

1. validate the current implementation in a compatible runtime,
2. run the missing baseline and ablation experiments in the correct order,
3. write reproducible artifacts under `.codex/experiments/`,
4. append ranked conclusions to `.codex/notes/research-log.md`,
5. only make code changes when a concrete runtime or metric issue is found.

## Current State

### Already implemented

The following work is already in the repo on `origin/loop`:

- `model.type="looped_transformer"` dispatch in `graphphysics/training/parse_parameters.py`
- recurrent primitives in `graphphysics/models/layers.py`
  - `StableInjection`
  - `LoopIndexEmbedding`
  - `NodeMoEFFN`
  - `NodeACTHalting`
  - `AdaptiveAdjacencyPolicy`
  - `RecurrentGraphCore`
  - `LoopedProcessorBlock`
- `LoopedEncodeTransformDecode` in `graphphysics/models/processors.py`
- loop metrics surfaced via `graphphysics/models/simulator.py`
- training instrumentation in `graphphysics/training/lightning_module.py`
  - recurrent diagnostics
  - throughput metrics
  - CUDA memory hooks
  - inference latency logging
- docs and configs
  - `README.md`
  - `training_config/coarse-aneurysm-looped.json`
  - `.codex/playbooks/looped-graph-transformer.md`
  - `.codex/configs/looped_transformer_{ablation,stability,scaling,vram}.json`
- tests added under `tests/graphphysics/...`

### Important constraints

- Exact research conclusions for `looped_transformer` are valid only on the **DGL sparse backend**.
- The local PyG-only fallback is compatibility-only and should not be used for final research claims.
- The environment previously used in this repo has a known `torch_scatter` / `torch_geometric` binary mismatch that can abort pytest during import. Do not treat that environment as authoritative.
- JAX is out of scope for now. Do not start a `jraphphysics` port unless explicitly directed after PyTorch evidence exists.

## Key Files

Architecture and training:

- `graphphysics/models/layers.py`
- `graphphysics/models/processors.py`
- `graphphysics/models/simulator.py`
- `graphphysics/training/parse_parameters.py`
- `graphphysics/training/lightning_module.py`
- `graphphysics/train.py`

Configs and experiment assets:

- `training_config/coarse-aneurysm.json`
- `training_config/coarse-aneurysm-looped.json`
- `.codex/configs/looped_transformer_ablation.json`
- `.codex/configs/looped_transformer_stability.json`
- `.codex/configs/looped_transformer_scaling.json`
- `.codex/configs/looped_transformer_vram.json`

Documentation and notes:

- `.codex/playbooks/looped-graph-transformer.md`
- `.codex/notes/research-log.md`
- `.codex/notes/jax-backlog.md`
- `.codex/templates/subagent-brief.md`
- `.codex/templates/experiment-result.md`

## Experiment Defaults

Use these defaults unless a specific sweep grid overrides them:

- `batch_size=2`
- `num_workers=2`
- `init_lr=0.001`
- `warmup=1000`
- `num_epochs=1` for smoke runs
- `num_epochs=20` for experiment sweeps

Reason:

- these values are already encoded in all current looped-transformer sweep grids
- they minimize loader/runtime variability while the architecture is still being validated
- they keep the baseline and looped runs directly comparable on the aneurysm scaffold

Do not widen these defaults at the start of the campaign. First establish that the looped model is stable and measurable with the current settings, then revisit throughput-oriented tuning if needed.

## Non-Negotiable Workflow

### Branching and workspace

Use a separate worktree for autonomous experiment work. Do not run everything from the main checkout.

Recommended setup:

```bash
git fetch origin
git worktree add ../graph-physics-looped-exp -b codex/looped-transformer-exp origin/loop
cd ../graph-physics-looped-exp
```

Keep any experiment-only fixes on the new `codex/...` branch. Do not push directly to `loop` unless explicitly asked.

### Reporting contract

For every multi-run batch:

1. generate configs/manifest with `.codex/scripts/architecture_sweep.py plan`
2. execute runs via `run-shard`
3. summarize with `summarize`
4. write a concise experiment result file using `.codex/templates/experiment-result.md`
5. append an evidence-backed entry to `.codex/notes/research-log.md`

Do not claim an improvement without one of:

- a log path
- a config path
- a `summary.csv` row
- a W&B summary artifact

## First Checks To Run

Run these before any training claims:

### 1. Runtime sanity

Confirm the target runtime can import the required stack:

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

If `dgl.sparse` fails, stop and report that the runtime is not acceptable for exact looped-transformer experiments.

### 2. Static code sanity

```bash
python -m py_compile \
  graphphysics/models/layers.py \
  graphphysics/models/processors.py \
  graphphysics/models/simulator.py \
  graphphysics/training/parse_parameters.py \
  graphphysics/training/lightning_module.py \
  graphphysics/train.py
```

### 3. Minimal targeted tests

Only run the new targeted graphphysics tests first. Do not start with the full suite.

```bash
PYTHONPATH=$PWD python -m pytest -q \
  tests/graphphysics/models/test_layers.py \
  tests/graphphysics/models/test_processors.py \
  tests/graphphysics/training/test_parameters.py
```

If these fail:

- fix only the concrete failure,
- re-run the same targeted set,
- record the fix in `.codex/notes/research-log.md`.

## Experiment Order

Run the remaining work in this exact order.

### Phase 0: bring-up

Goal: prove the scaffold trains at all in a compatible runtime.

Required runs:

1. Baseline untied transformer smoke run
2. Looped transformer smoke run

Suggested commands:

```bash
PYTHONPATH=$PWD python -m graphphysics.train \
  --training_parameters_path=training_config/coarse-aneurysm.json \
  --project_name=graphphysics-looped-smoke \
  --num_epochs=1 \
  --batch_size=2 \
  --init_lr=0.001 \
  --warmup=1000 \
  --num_workers=2
```

```bash
PYTHONPATH=$PWD python -m graphphysics.train \
  --training_parameters_path=training_config/coarse-aneurysm-looped.json \
  --project_name=graphphysics-looped-smoke \
  --num_epochs=1 \
  --batch_size=2 \
  --init_lr=0.001 \
  --warmup=1000 \
  --num_workers=2
```

Capture for both:

- command
- config path
- log path
- `train_*_epoch`
- `val_1step_rmse`
- `val_all_rollout_rmse`
- `loop/spectral_radius_max`
- `loop/state_norm`
- `loop/residual_jump`
- `perf/step_time`
- `perf/inference_latency`

Only move on if the looped model completes a 1-epoch smoke run.

### Phase 1: core ablation ladder

Goal: isolate which recurrent ingredients matter.

Required comparison order:

1. current untied `transformer`
2. tied recurrent core without stable injection
3. `+ StableInjection`
4. `+ LoopIndexEmbedding`
5. `+ per-graph loop sampling`
6. `+ NodeMoEFFN`
7. `+ ACT halting`
8. `+ target-active adjacency`
9. `+ learned edge gating`

Use:

- base config: `training_config/coarse-aneurysm-looped.json`
- grid: `.codex/configs/looped_transformer_ablation.json`

Planner:

```bash
python .codex/scripts/architecture_sweep.py plan \
  --base-config training_config/coarse-aneurysm-looped.json \
  --grid .codex/configs/looped_transformer_ablation.json \
  --out-dir .codex/experiments/sweeps/<timestamp>_looped_ablation
```

Then shard it with `run-shard` and aggregate with `summarize`.

If the grid is too broad for the current budget, reduce run count by editing a copy of the grid file in the worktree, but preserve the ablation order.

### Phase 2: recurrence stability

Goal: determine whether stable injection and loop sampling improve optimization robustness.

Use:

- grid: `.codex/configs/looped_transformer_stability.json`

Required outputs:

- `summary.csv`
- ranked table by `metric.val_1step_rmse`
- notes on NaNs, crashes, or instability
- observations for `loop/spectral_radius_max`, `loop/state_norm`, `loop/residual_jump`

### Phase 3: operating point and scaling

Goal: identify viable size/depth operating points.

Use:

- grid: `.codex/configs/looped_transformer_scaling.json`

Focus on:

- debug `(hidden=16, prc_depth=1)`
- small `(64, 3)`
- medium `(128, 5)`
- large `(256, 5)`

Do not claim scaling behavior from partial or failed rows. Summarize only completed rows.

### Phase 4: VRAM and throughput

Goal: measure whether the looped architecture and adaptive compute controls change memory/latency in practice.

Use:

- grid: `.codex/configs/looped_transformer_vram.json`

Primary metrics:

- `duration_seconds`
- `perf/step_time`
- `perf/graphs_per_sec`
- `perf/nodes_per_sec`
- `perf/edges_per_sec`
- `perf/peak_allocated`
- `perf/peak_reserved`
- `perf/inference_latency`

This phase is only worth running after the smoke and ablation phases are stable.

## What To Do If You Hit Problems

### If imports or tests fail

- fix the smallest concrete issue
- keep the fix on the autonomous branch
- re-run only the failing targeted check first
- document the exact command and failure mode in `research-log.md`

### If smoke training fails

Debug in this order:

1. config parsing
2. model forward pass on one batch
3. DGL sparse adjacency path
4. recurrent metrics logging
5. validation callback / inference path

Do not jump straight into sweep execution until a 1-epoch looped run succeeds.

### If adaptive adjacency or ACT breaks training

Do not redesign the feature.

Instead:

- disable the newest feature
- recover a stable earlier rung in the ablation ladder
- file the failure as a concrete follow-up in `research-log.md`

## Scope Boundaries

In scope:

- smoke validation of the current looped architecture
- PyTorch-side bug fixes needed to complete the planned experiments
- sweep execution and summary
- note updates and reproducible artifact generation

Out of scope for this handoff:

- JAX port
- `LoopedEncodeProcessDecode`
- edge-conditioned attention redesign
- replacing the existing sweep runner
- full architecture redesign

## Required Artifacts

For each experiment batch, produce:

- sweep directory under `.codex/experiments/sweeps/<timestamp>_<name>/`
- generated configs under `configs/`
- manifest under `manifest.jsonl`
- run logs under `results/*.log`
- shard journals under `results/journal-shard*.jsonl`
- aggregate table under `results/summary.csv`
- short markdown result note using `.codex/templates/experiment-result.md`

Update:

- `.codex/notes/research-log.md`

Only update:

- `.codex/notes/jax-backlog.md`

if PyTorch results materially change the JAX parity priority.

## Done Criteria

This handoff is complete when all of the following are true:

1. A compatible DGL runtime has been confirmed.
2. The targeted looped-transformer tests pass in that runtime.
3. Both 1-epoch smoke runs complete:
   - baseline transformer
   - looped transformer
4. At least the ablation sweep has been run and summarized.
5. `research-log.md` contains evidence-backed conclusions and next actions.
6. Any code fixes made during experimentation are committed on the autonomous branch.

## Short Version For The Agent

Do not re-implement the looped transformer. It already exists. Validate it in a real DGL runtime, run the smoke jobs, then run the ablation, stability, scaling, and VRAM sweeps in that order, and write everything down with reproducible artifact paths.
