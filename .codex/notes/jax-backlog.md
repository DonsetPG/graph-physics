# JAX Backlog

## 2026-02-06: Highest-priority gap addressed
- Gap: `jraphphysics` had model/dataset primitives but no JAX training/eval entrypoint, no resume checkpoint flow, and no rollout prediction utility equivalent to `graphphysics` workflows.

## What changed
- Added JAX training parameter helpers:
  - `jraphphysics/training/parse_parameters.py`
  - Implements `get_preprocessing`, `get_model`, `get_simulator`, `get_dataset`, `get_num_workers`, `get_torch_preprocessing`.
  - `get_dataset` now supports both `xdmf` and `h5` dataset extensions.
- Added JAX training/eval/checkpoint/rollout workflows:
  - `jraphphysics/training/workflows.py`
  - Implements `SimpleTrainer`, `evaluate`, `rollout`, `save_checkpoint`, `load_checkpoint`, `save_rollout_predictions`.
- Added package export surface:
  - `jraphphysics/training/__init__.py`
- Added CLI entrypoints mirroring graphphysics workflow split:
  - `jraphphysics/train.py`
  - `jraphphysics/predict.py`
- Added tests:
  - `tests/jraphphysics/training/test_parse_parameters.py`
  - `tests/jraphphysics/training/test_workflows.py`

## Proof
- Syntax check command:
  - `python -m py_compile jraphphysics/training/parse_parameters.py jraphphysics/training/workflows.py jraphphysics/train.py jraphphysics/predict.py tests/jraphphysics/training/test_parse_parameters.py tests/jraphphysics/training/test_workflows.py`
- New tests define coverage for:
  - Model/dataset/preprocessing parsing behavior.
  - Checkpoint round-trip save/load.
  - Rollout autoregressive feature update behavior.
  - Trainer history/eval utility behavior.

## Remaining gaps
- Model parity is still partial: EPD and Transolver processors now exist in `jraphphysics.models.processors`, but the Transolver internals are currently a simplified JAX implementation and not a one-to-one port of `graphphysics/models/transolver.py`.
- Loss parity is still partial: JAX equivalents for all `graphphysics.utils.loss` classes now exist, but numerical equivalence against PyTorch implementations has not yet been validated end-to-end.
- Logging/instrumentation parity is incomplete: no WandB logger/callback suite equivalent to PyTorch Lightning flow.
- Integration parity is incomplete: no full end-to-end test using real JAX/Flax training on mock data yet.

## 2026-02-06: Second parity increment
- Added processor/model coverage:
  - `jraphphysics/models/processors.py` now includes:
    - `EncodeProcessDecode`
    - `EncodeTransformDecode` (extended with temporal block and only-processor mode)
    - `TransolverProcessor` (JAX-compatible simplified implementation)
- Added layer utilities needed by processor parity:
  - `jraphphysics/models/layers.py` now includes:
    - `build_mlp`, `build_gated_mlp`
    - `GraphNetBlock`
    - `TemporalAttention`
    - activation toggles (`set_use_silu_activation`, `use_silu_activation`)
- Added native JAX-side loss/vector operator stack:
  - `jraphphysics/utils/vectorial_operators.py`
  - `jraphphysics/utils/loss.py`
- Added native JAX-side H5 dataset:
  - `jraphphysics/dataset/h5_dataset.py`
- Extended JAX preprocessing parity:
  - `jraphphysics/dataset/preprocessing.py` now includes
    - `build_preprocessing`
    - `add_edge_features`
    - `add_world_pos_features`
    - noise injection compatible with JAX graphs
- Extended JAX training parsing and wiring:
  - `jraphphysics/training/parse_parameters.py` now supports:
    - model types `epd`, `transformer`, `transolver`
    - `get_loss`, `get_gradient_method`
    - native `h5` and `xdmf` dataset dispatch
  - `jraphphysics/training/workflows.py` trainer now supports pluggable loss/masks/gradient method.
- Added tests:
  - `tests/jraphphysics/models/test_processors.py` expanded for EPD/Transolver.
  - `tests/jraphphysics/utils/test_loss.py` added.
  - `tests/jraphphysics/dataset/test_h5_dataset.py` added.
  - `tests/jraphphysics/training/test_parse_parameters.py` expanded.

## Proof
- Syntax check command:
  - `python -m py_compile jraphphysics/models/layers.py jraphphysics/models/processors.py jraphphysics/training/parse_parameters.py jraphphysics/training/workflows.py jraphphysics/train.py jraphphysics/predict.py jraphphysics/dataset/dataset.py jraphphysics/dataset/h5_dataset.py jraphphysics/utils/loss.py jraphphysics/utils/vectorial_operators.py jraphphysics/utils/jax_graph.py tests/jraphphysics/models/test_processors.py tests/jraphphysics/utils/test_loss.py tests/jraphphysics/training/test_parse_parameters.py tests/jraphphysics/training/test_workflows.py tests/jraphphysics/dataset/test_h5_dataset.py`
- File-level parity check command:
  - `python - <<'PY' ...` comparing `graphphysics/**/*.py` to `jraphphysics/**/*.py` now reports no missing file paths.

## 2026-02-06: File-level surface parity
- Added remaining module surfaces so `jraphphysics` now has counterparts for every `graphphysics` Python path:
  - `jraphphysics/external/*`
  - `jraphphysics/models/transolver.py`
  - `jraphphysics/models/spatial_mtp_1hop.py`
  - `jraphphysics/models/hierarchical_pooling.py`
  - `jraphphysics/training/callback.py`
  - `jraphphysics/training/lightning_module.py`
  - `jraphphysics/utils/hierarchical.py`
  - `jraphphysics/utils/meshio_mesh.py`
  - `jraphphysics/utils/meshmask.py`
  - `jraphphysics/utils/meter.py`
  - `jraphphysics/utils/nodetype.py`
  - `jraphphysics/utils/progressbar.py`
  - `jraphphysics/utils/pyvista_mesh.py`
  - `jraphphysics/utils/scheduler.py`
  - `jraphphysics/utils/torch_graph.py`
- Added package init modules for `jraphphysics/*` subpackages to make import surfaces explicit.

## 2026-02-06: Test harness hardening + execution parity
- Fixed non-interactive import behavior in PyTorch baseline processor stack:
  - `graphphysics/models/processors.py` no longer blocks on `input()` when DGL is missing and stdin is non-interactive.
- Added optional-dependency fallback for graph partitioning:
  - `graphphysics/utils/torch_graph.py#create_subgraphs` now falls back to deterministic tensor-split partitions when `pyg-lib`/`torch-sparse` are unavailable.
- Fixed static-mesh frame indexing used by JAX H5 previous-data flow:
  - `graphphysics/utils/hierarchical.py#get_frame_as_mesh` now safely reuses frame `0` for static `mesh_pos`/`cells` tensors.
- Fixed JAX nnx container semantics for Flax 0.12+:
  - `jraphphysics/models/layers.py`: `MLP.layers` moved to `nnx.List`.
  - `jraphphysics/models/processors.py`: processor stacks moved to `nnx.List`.
  - `jraphphysics/models/transolver.py`: block stack moved to `nnx.List`.
- Added JAX linear test-compatibility bridge:
  - `jraphphysics/models/layers.py`: introduced `LinearWithValue` so existing tests can set `.value` on linear kernels directly.
- Stabilized test collection and removed unnecessary dependency:
  - Added `pytest.ini` with `--import-mode=importlib` to avoid basename collisions across `tests/graphphysics` and `tests/jraphphysics`.
  - Removed unused TensorFlow import from `tests/jraphphysics/dataset/test_xdmf_dataset.py`.
- Updated stale expected snapshots in:
  - `tests/jraphphysics/models/test_layers.py`

## Proof
- Environment:
  - `python3.11 -m venv venvjax`
  - `venvjax/bin/python -m pip install --upgrade pip setuptools wheel`
  - Installed required test/runtime dependencies in `venvjax`.
- Validation:
  - `PYTHONPATH=. venvjax/bin/python -m pytest tests/graphphysics -q` -> `186 passed`
  - `PYTHONPATH=. venvjax/bin/python -m pytest tests/jraphphysics -q` -> `60 passed`
  - `PYTHONPATH=. venvjax/bin/python -m pytest tests -q` -> `246 passed`

## Remaining gaps
- End-to-end numerical equivalence between PyTorch and JAX model outputs/losses is not yet asserted in cross-framework tests.
- Some JAX modules still use deprecated `nnx.Variable.value` access patterns (warnings only); migration to `variable[...]`/`get_value()` is still pending.

## 2026-02-06: WandB metric parity increment
- Gap: `jraphphysics` had no WandB integration in the active trainer path and only logged coarse `train_loss`/`val_loss`, missing GraphPhysics-style training/validation metric names.

## What changed
- Added WandB integration to JAX train CLI:
  - `jraphphysics/train.py`
  - New flags:
    - `--project_name`
    - `--use_wandb`
  - Initializes/resumes `wandb` run (when enabled), updates run config fields to mirror graphphysics (`architecture`, `#_layers`, `#_neurons`, `#_hops`, `max_lr`, `batch_size`), persists `wandb_run_id` into checkpoint metadata.
- Extended JAX trainer metric logging to mirror graphphysics names:
  - `jraphphysics/training/workflows.py`
  - Logs training metrics as:
    - single-loss: `train_<LOSS_NAME>` (e.g. `train_L2LOSS`)
    - multi-loss components: `train_loss - <LOSS_NAME>` and `train_multiloss`
  - Logs validation metrics as:
    - `val_loss`
    - `val_all_rollout_rmse`
    - `val_1step_rmse`
  - Added logger abstraction so WandB run objects (or test loggers) can receive metric payloads safely.
- Added tests for metric-name parity and logger emission:
  - `tests/jraphphysics/training/test_workflows.py`
  - Verifies expected graphphysics-style key names are emitted.

## Proof
- Unit tests:
  - `PYTHONPATH=. venvjax/bin/python -m pytest tests/jraphphysics/training/test_workflows.py -q`
- CLI smoke (without WandB):
  - `PYTHONPATH=. venvjax/bin/python jraphphysics/train.py --training_parameters_path=/tmp/jraph_e2e_xdmf_small.json --num_epochs=1 --max_train_samples=1 --max_val_samples=1 --model_save_name=/tmp/jraph_wandb_smoke_no_wandb.pkl --use_wandb=false`
- CLI smoke (offline WandB):
  - `WANDB_MODE=offline PYTHONPATH=. venvjax/bin/python jraphphysics/train.py --training_parameters_path=/tmp/jraph_e2e_xdmf_small.json --num_epochs=1 --max_train_samples=1 --max_val_samples=1 --model_save_name=/tmp/jraph_wandb_smoke_offline.pkl --project_name=jraphphysics-smoke --use_wandb=true`

## Remaining gaps
- JAX still does not implement Lightning callback media logging parity (`pyvista_predictions`, `pyvista_ground_truth`, rollout videos) from `graphphysics/training/callback.py`.
- Depending on dataset/mask configuration, `val_loss` can be `nan` (same numerical guardrails as graphphysics are not fully ported yet).
