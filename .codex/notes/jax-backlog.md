# JAX Backlog (Living)

Last updated: 2026-02-06

## Open Gaps
- [ ] Add JAX training entrypoint comparable to `graphphysics/train.py`.
- [ ] Define checkpoint save/load flow for JAX models.
- [ ] Add at least one end-to-end training integration test in `tests/jraphphysics/`.
- [ ] Audit dataset output contract: currently `XDMFDataset.__getitem__` returns dict-like mesh payload; evaluate standardizing toward direct `jraph.GraphsTuple` path.
- [ ] Port `StableInjection`, `RecurrentGraphCore`, and `LoopedEncodeTransformDecode` to `jraphphysics` after PyTorch looped-transformer experiments establish the operating point.

## Known Evidence
- `jraphphysics` has dataset/model/simulator components and unit coverage in `tests/jraphphysics/`.
- `jraphphysics/dataset/dataset.py` contains abstract methods and no concrete training orchestrator.
- No `jraphphysics/train.py` exists yet.

## Session Log
- 2026-02-06: Scaffolded codex workflows, skills, and sweep tooling for JAX completion + architecture search.
- 2026-04-19: Added the PyTorch `looped_transformer` architecture program (layers, processor, config surface, sweep grids). JAX parity is intentionally deferred until PyTorch smoke runs and ablations produce evidence.
