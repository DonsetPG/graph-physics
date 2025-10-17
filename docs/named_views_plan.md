# Named Views over `graph.x` – Implementation Plan

## 1. Objectives
- Deliver wrapper-free, name-addressable access to `torch_geometric.data.Data.x`, matching the RFC requirements (selection, assignment, schema editing, compat helpers).
- Preserve contiguity and PyG-native behaviour while layering deterministic feature layouts and a `NamedData` ergonomics layer.
- Maintain backwards compatibility for existing index-based configs during migration, with opt-in support for semantic configs.
- Ship comprehensive tests, documentation, and example workflows to validate the new API.

## 2. Current State Inventory (index-based touch points)

### Models & Training
- `graphphysics/models/simulator.py:24` – constructor wired to `feature_index_*`, `output_index_*`, `node_type_index`.
- `graphphysics/models/simulator.py:90` – `_get_pre_target` slices by numeric windows.
- `graphphysics/models/simulator.py:122` – `_get_one_hot_type` pulls node type via raw column index.
- `graphphysics/models/simulator.py:140` – `_build_node_features` concatenates slices via numeric offsets.
- `graphphysics/training/lightning_module.py:23` – `build_mask` reads node type indices directly from `param["index"]`.
- `graphphysics/training/lightning_module.py:124` – `training_step` collects node type via numeric index.
- `graphphysics/training/lightning_module.py:221` – `_make_prediction` rewrites `batch.x[:, start:end]` windows for rollouts.
- `graphphysics/training/lightning_module.py:231` – `_make_prediction` caches current outputs via numeric ranges.
- `graphphysics/training/lightning_module.py:271` – `validation_step` fetches node type via positional lookup.
- `graphphysics/training/callback.py:97` – prediction visualisation swaps numeric slices into/out of `graph.x`.
- `graphphysics/training/parse_parameters.py:136` – `get_simulator` forwards numeric windows from config.

### Dataset & Preprocessing
- `graphphysics/dataset/h5_dataset.py:129` – `_build_node_features` concatenates arrays into `x` without schema metadata.
- `graphphysics/dataset/h5_dataset.py:223` – `__getitem__` attaches `previous_data` as raw tensor without layout.
- `graphphysics/dataset/preprocessing.py:49` – `add_obstacles_next_pos` expects explicit `world_pos_index_*` and manual adjustments.
- `graphphysics/dataset/preprocessing.py:92` – `add_world_edges` slices `graph.x` via numeric spans.
- `graphphysics/dataset/preprocessing.py:143` – `add_world_pos_features` depends on contiguous (start, end) windows.
- `graphphysics/dataset/preprocessing.py:177` – `add_noise` requires paired `[start, end)` ranges in config.
- `graphphysics/dataset/preprocessing.py:404` – `build_preprocessing` threads numeric indices through partials for each transform.
- `graphphysics/utils/hierarchical.py:116` – `get_frame_as_mesh`/`get_frame_as_graph` assemble feature tensors but drop semantic names.
- `graphphysics/utils/torch_graph.py:128` – `meshdata_to_graph` concatenates point data in dictionary order without exposing layout metadata.
- `graphphysics/utils/meshmask.py:63` – masking copies `graph.x[selected_indexes]` without carrying layout attributes forward.

### External Feature Builders & Utilities
- `graphphysics/external/panels.py:9` – hard-coded column windows for velocity, pressure, masks.
- `graphphysics/external/aneurysm.py:10` – assumes velocity at columns 0:3 and wall mask at 3.
- `graphphysics/external/bezier.py:8` – column-wise boundary feature detection by positional indices.
- `graphphysics/utils/meshio_mesh.py:55` – exports features as `x{i}` without semantic labels or layout awareness.
- `graphphysics/utils/torch_graph.py:205` – `mesh_to_graph` returns standard `Data` without layout metadata.

### Configuration & Docs
- `training_config/*.json` – `index` sections define all feature slices; no semantic schema.
- `mock_training.json:19` – mirrors legacy index-based structure used in tests and examples.
- `README.md:193` – documents numeric `feature_index_*` / `output_index_*` fields and usage.

### Tests
- `tests/graphphysics/models/test_simulator.py:24` – simulator fixtures rely on raw indices.
- `tests/graphphysics/training/test_lightningmodule.py:56` – parameters fixture injects `index` spans.
- `tests/graphphysics/dataset/test_preprocessing.py:64` – preprocessing tests assert behaviour via numeric slices.
- `tests/jraphphysics/models/test_simulator.py:91` – JAX simulator mirrors index-based assumptions.
- `tests/jraphphysics/dataset/test_preprocessing.py:56` – JAX preprocessing uses numeric ranges identical to PyG pipeline.

## 3. Target Architecture Overview
- Introduce `named_features` package (`named_features/__init__.py`, `x_layout.py`, `named_data.py`, `compat.py`, `exceptions.py`) implementing `XFeatureLayout`, `NamedData`, and compat helpers as defined in the RFC.
- Persist layout metadata on every `NamedData` instance, ensuring PyG `Batch` collation preserves `x_layout` and optional `x_coords`.
- Generators (datasets, loaders, preprocessing transforms) operate on semantic names, delegating slice resolution to the layout.
- Configuration supplies ordered feature names and sizes; meta JSON remains the source of truth for base fields.
- Back-compat: derive numeric windows via `old_indices_from_layout` and expose shim functions so existing downstream modules can temporarily consume index-based APIs.

## 4. Implementation Phases & Tasks

### Phase 0 – Groundwork & Schema Inputs
- Define canonical feature naming for existing datasets (e.g., velocity components, derived channels) by inspecting sample metadata and preprocessing outputs.
- Draft semantic config schema (`features.node`, `sizes`, `targets`, optional `coords`) and map to legacy configs for each training profile.

### Phase 1 – Core `named_features` Package
- Create `named_features/x_layout.py` implementing `XBlock`, `XFeatureLayout`, serialization, validation, and builders (`make_x_layout`, `x_layout_from_meta_and_spec`).
- Add `named_features/named_data.py` with `NamedData` subclass, ensuring:
  - Override `__cat_dim__`/`__inc__` if needed to keep PyG batching intact.
  - `x_sel`, `x_assign`, `x_drop`, `x_to_dict`, `x_from_dict`, `x_rename`, `x_reorder`, `validate_x` per RFC.
- Implement `named_features/compat.py` providing `old_indices_from_layout` and any small helpers for migration (e.g., deriving node type slice start).
- Define `named_features/exceptions.py` for consistent error messages.
- Export public API via `named_features/__init__.py`.

### Phase 2 – Layout Construction & Config Parsing
- Extend parameter parsing in `graphphysics/training/parse_parameters.py:136` to:
  - Read semantic config (`features`, `sizes`, `targets`) with validation.
  - Build `XFeatureLayout` using `meta` + overrides.
  - Populate legacy `param["index"]` via `old_indices_from_layout` for downstream compatibility during rollout.
- Support fallback: if only legacy `index` block is present, auto-create a `XFeatureLayout` with generated placeholder names to ease incremental migration.
- Update CLI entry points (`graphphysics/train.py`, `graphphysics/predict.py`) if they materialise params to ensure layout objects propagate.

### Phase 3 – Dataset & DataLoader Integration
- In `graphphysics/utils/hierarchical.py:116`, ensure `point_data` retains deterministic order aligned with semantic config (likely an `OrderedDict` keyed by config order).
- Update `graphphysics/utils/torch_graph.py:128` to return `NamedData` (or attach layout post-creation) and embed layout metadata on the graph.
- Modify `graphphysics/dataset/h5_dataset.py:129` and `graphphysics/dataset/dataset.py` to:
  - Attach `x_layout` (and optional `x_coords`) when emitting samples.
  - Ensure cached graphs preserve layout across cloning and masking.
- Adjust masking helpers (`graphphysics/utils/meshmask.py:63`) to carry `x_layout` through slicing operations (clone + manual attribute copy if needed).
- Verify PyG batching keeps `x_layout` on `Batch` objects; override `collate` or register `NamedData` collate function if necessary.

### Phase 4 – Preprocessing & Transform Rewrite
- Refactor `graphphysics/dataset/preprocessing.py:49` (`add_obstacles_next_pos`) to accept feature names (e.g., `"mesh_pos"`, `"target_mesh_pos"`, `"node_type"`) and resolve slices via layout.
- Update `add_world_edges` (`graphphysics/dataset/preprocessing.py:92`) and `add_world_pos_features` (`graphphysics/dataset/preprocessing.py:143`) to operate on named selectors.
- Replace `add_noise` (`graphphysics/dataset/preprocessing.py:177`) signature with name-based arguments (`noise_features: list[str]`), leveraging `x_sel`/`x_assign`.
- Update `build_preprocessing` (`graphphysics/dataset/preprocessing.py:404`) to parse semantic transform configs, call new helpers, and maintain backwards compatibility by translating numeric configs through the compat layer.
- Mirror equivalent changes in `jraphphysics/dataset/preprocessing.py:7` for JAX workflows (name → index resolution via shared layout metadata).

### Phase 5 – Model & Training Adaptation
- Update `graphphysics/models/simulator.py:24` to accept an `XFeatureLayout` (or a `NamedData` instance) rather than raw indices; store frequently used slice handles for efficiency.
- Rewrite internal helpers (`graphphysics/models/simulator.py:90`, `graphphysics/models/simulator.py:122`, `graphphysics/models/simulator.py:140`) to call `x_sel("...")`/`x_slice("...")`.
- Modify `graphphysics/training/lightning_module.py` call sites (`build_mask`, `training_step`, `_make_prediction`, `validation_step`) to use layout-driven access instead of manual slicing.
- Adjust `graphphysics/training/callback.py:97` to read/write named selections when updating `graph.x` during rollouts.
- Ensure `NamedData` flows seamlessly through training loops (check `.clone()`, `.to()` behaviours).
- Apply analogous changes to `jraphphysics/models/simulator.py` to keep parity (using layout metadata stored in the GraphsTuple globals, or bridging helper).

### Phase 6 – External Feature Builders & Utilities
- Revise `graphphysics/external/panels.py:9`, `graphphysics/external/aneurysm.py:10`, and `graphphysics/external/bezier.py:8` to reference feature names, using `NamedData.x_sel` and `x_assign`.
- Enhance `graphphysics/utils/meshio_mesh.py:55` to export semantic names (e.g., using layout `x_names()`); provide option to map to legacy `x{i}` if layout missing.
- Update `graphphysics/utils/torch_graph.py:205` and dependent mesh conversion utilities to accept/return layout-aware data.

### Phase 7 – Backwards Compatibility & Migration Hooks
- Introduce a lightweight adapter that exposes derived `feature_index_start/end` etc. for legacy code paths until deprecated.
- Provide CLI flags or config switches to opt into semantic mode while still generating numeric indices for downstream scripts.
- Add validation checks warning when numeric configs and semantic configs disagree.
- Document transitional workflow for existing experiments (e.g., auto-generating semantic config from current index settings).

### Phase 8 – Testing & Tooling
- Add unit tests mirroring RFC §9:
  - `tests/named_features/test_x_layout.py` covering construction, rename, reorder, serialization, meta builder.
  - `tests/named_features/test_named_data_select.py`, `test_named_data_assign.py`, `test_named_data_editing.py`, `test_named_data_batching.py`, `test_named_data_compat.py`.
- Update existing tests to use semantic configs:
  - `tests/graphphysics/models/test_simulator.py:24` – instantiate layout and assert API usage.
  - `tests/graphphysics/training/test_lightningmodule.py:56` – update fixtures to supply layout-aware parameters.
  - `tests/graphphysics/dataset/test_preprocessing.py:64` – rewrite assertions around name-based transforms.
  - `tests/jraphphysics/...` counterparts for dataset and simulator.
- Add integration smoke test exercising dataset → preprocessing → model pipeline with named selections.
- Ensure coverage for `Batch` collation, compat helpers, serialization, and error messaging.

### Phase 9 – Documentation, Configs, and Examples
- Replace `index` blocks in sample configs (`training_config/*.json`, `mock_training.json:19`) with semantic `features`/`sizes` sections, keeping legacy fields available behind a migration flag.
- Update `README.md:193` and surrounding sections to explain named layouts, new helpers, and migration steps.
- Provide an example script (e.g., `examples/named_features_demo.py`) showing layout creation, `NamedData` operations, reorder/drop/export.
- Document CLI instructions for generating layout summaries and exporting compat indices.

## 5. Validation Strategy
- Unit + integration tests outlined above.
- Benchmark selection/assignment overhead vs. baseline to confirm negligible impact (<1%).
- Run representative training smoke tests comparing pre/post change metrics to ensure numerical parity.
- Validate batching with large graphs to ensure no regressions in memory layout or performance.
- Confirm serialization (JSON export/import) works for checkpoints and reproducibility.

## 6. Migration Plan & Rollout
- Step 1: Ship core package + tests while keeping legacy index configs functional via compat adapters.
- Step 2: Migrate internal configs to semantic format; provide script to auto-generate names from existing indices.
- Step 3: Deprecate legacy index usage (warnings), schedule removal once downstream consumers adopt named API.
- Step 4: Update external datasets/plugins gradually, leveraging compat helpers during transition.

## 7. Risks & Open Questions
- Ensuring PyG `Batch` retains `x_layout` without manual intervention; may require custom collate logic.
- Coordinating naming conventions for derived features (current preprocessing appends new columns whose semantics may vary per project).
- Keeping JAX (`jraphphysics`) parity if layout metadata must live outside PyG `Data`.
- Handling legacy checkpoints expecting numeric indices; need conversion utilities.
- Potential third-party scripts importing `graphphysics` and slicing `.x` directly—need communication plan or helper wrappers.

## 8. Deliverables Checklist
- [ ] `named_features` package implemented with docs and tests.
- [ ] Dataset + preprocessing pipeline emitting `NamedData`.
- [ ] Training, callbacks, and simulators consuming named selections.
- [ ] Updated configs, README, and examples.
- [ ] Migration utilities and compatibility shims.
- [ ] Comprehensive automated tests covering new API surface.

