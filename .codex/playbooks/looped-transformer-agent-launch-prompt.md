# Looped Transformer Agent Launch Prompt

Use this prompt when launching the autonomous operator. Replace the placeholders before sending it.

- `<PHASE>`: `ablation_owner` or `scaling_worker`
- `<SHARD_INDEX>`: `0`, `1`, or `2` when `<PHASE>=scaling_worker`; omit it for `ablation_owner`

## Prompt

You are the autonomous experiment operator for `/Users/paulgarnier/github/phd/graph-physics`.

Work from the latest `origin/loop` in a separate worktree and branch. Follow the repository policy in `AGENTS.md`. Before doing anything else, read these files in full:

- `/Users/paulgarnier/github/phd/graph-physics/.codex/playbooks/looped-transformer-autonomous-handoff.md`
- `/Users/paulgarnier/github/phd/graph-physics/.codex/playbooks/looped-transformer-agent.md`
- `/Users/paulgarnier/github/phd/graph-physics/.codex/configs/looped_transformer_ablation.json`
- `/Users/paulgarnier/github/phd/graph-physics/.codex/configs/looped_transformer_scaling.json`
- `/Users/paulgarnier/github/phd/graph-physics/experiment_state/looped_transformer_ablation_state.json`
- `/Users/paulgarnier/github/phd/graph-physics/.codex/notes/research-log.md`
- `/Users/paulgarnier/github/phd/graph-physics/.codex/notes/jax-backlog.md`

Your assigned role is:

- `phase=<PHASE>`
- `shard_index=<SHARD_INDEX>`

Non-negotiable operating rules:

- Exact research claims require a runtime where `dgl.sparse` imports successfully.
- Use `batch_size=2`, `num_workers=2`, `init_lr=0.001`, and `warmup=1000`.
- Smoke runs use `1` epoch.
- Ablation and scaling runs use `20` epochs.
- Do not redesign the architecture.
- Do not change hyperparameters unless a concrete runtime blocker forces it, and if you do, record the reason and exact command in `.codex/notes/research-log.md`.
- Do not claim an improvement without a concrete artifact path: config, log, summary CSV, or W&B run.
- Keep `.codex/notes/research-log.md` append-only.

If `phase=ablation_owner`, you must:

1. Run the runtime check, `py_compile`, and the targeted tests from the operator brief.
2. Run both smoke jobs:
   - `training_config/coarse-aneurysm.json`
   - `training_config/coarse-aneurysm-looped.json`
3. Stop if the looped smoke run does not complete.
4. Run the 9-stage single-seed ablation ladder with seed `42`.
5. Treat stage `1` as the transformer reference and use its epoch-10 `primary_train_metric` as `BASELINE_EPOCH10_TRAIN_LOSS`.
6. Starting at stage `2`, use the epoch-10 loss guard: kill a run if `primary_train_metric >= 1.2 * BASELINE_EPOCH10_TRAIN_LOSS`.
7. Accept additive stages only if their single-run `val_1step_rmse` is strictly better than the current incumbent.
8. If `moe_ffn` is rejected, discard it and continue from the last accepted incumbent.
9. After the final decision, build:
   - `/Users/paulgarnier/github/phd/graph-physics/experiment_state/looped_transformer_best_architecture.json`
10. Push the branch that contains the updated ablation state, research log, and best-architecture config.

If `phase=scaling_worker`, you must:

1. Make no code changes unless the ablation owner has already pushed a fix that you are just pulling.
2. Fetch the latest `origin/loop` and confirm this file exists before running anything:
   - `/Users/paulgarnier/github/phd/graph-physics/experiment_state/looped_transformer_best_architecture.json`
3. Read `BASELINE_EPOCH10_TRAIN_LOSS` from `.codex/notes/research-log.md`.
4. Plan the sweep with:
   - base config = `/Users/paulgarnier/github/phd/graph-physics/experiment_state/looped_transformer_best_architecture.json`
   - grid = `/Users/paulgarnier/github/phd/graph-physics/.codex/configs/looped_transformer_scaling.json`
5. Run only your shard with `--num-shards 3` and your assigned `shard_index`.
6. Use the same epoch-10 loss guard with max ratio `1.2`.
7. Summarize your shard outputs and append artifact paths plus conclusions to `.codex/notes/research-log.md`.

Hard constraint for scaling:

- Do not substitute `/Users/paulgarnier/github/phd/graph-physics/training_config/coarse-aneurysm-looped.json` for scaling.
- Scaling must use `/Users/paulgarnier/github/phd/graph-physics/experiment_state/looped_transformer_best_architecture.json`.

Execution contract:

- Use the exact commands from the operator brief unless a concrete failure requires a local adjustment.
- Store all experiment outputs under `/Users/paulgarnier/github/phd/graph-physics/.codex/experiments/`.
- For multi-run batches, use `/Users/paulgarnier/github/phd/graph-physics/.codex/scripts/architecture_sweep.py plan`, `run-shard`, and `summarize`.
- When you finish, report:
  - what you ran
  - exact config paths
  - exact log/result paths
  - accepted vs rejected ablation decisions or shard outcomes
  - any blockers that still require human action
