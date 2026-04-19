#!/usr/bin/env python3
"""Helpers for the adaptive looped-transformer ablation ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _load_spec(path: Path) -> Dict[str, Any]:
    spec = _read_json(path)
    stages = spec.get("stages", [])
    if not stages:
        raise ValueError(f"No stages found in {path}")
    return spec


def _load_state(path: Path) -> Dict[str, Any]:
    return _read_json(path)


def _stage_by_index(spec: Dict[str, Any], stage_index: int) -> Dict[str, Any]:
    for stage in spec["stages"]:
        if int(stage["index"]) == stage_index:
            return stage
    raise KeyError(f"Stage {stage_index} not found")


def _stage_names_before(spec: Dict[str, Any], stage_index: int) -> List[str]:
    names = []
    for stage in spec["stages"]:
        idx = int(stage["index"])
        if idx >= stage_index:
            break
        if stage["stage_type"] == "additive":
            names.append(stage["name"])
    return names


def _base_looped_reset_stage(spec: Dict[str, Any]) -> Dict[str, Any]:
    for stage in spec["stages"]:
        if stage["stage_type"] == "reset_incumbent":
            return stage
    raise KeyError("No reset_incumbent stage found")


def _build_incumbent_config(spec: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    reset_stage = _base_looped_reset_stage(spec)
    config = _read_json(REPO_ROOT / reset_stage["base_config"])
    for key, value in reset_stage.get("set", {}).items():
        _set_nested(config, key, value)

    accepted = set(state.get("accepted_stage_names", []))
    for stage in spec["stages"]:
        if stage["stage_type"] != "additive":
            continue
        if stage["name"] not in accepted:
            continue
        for key, value in stage.get("set", {}).items():
            _set_nested(config, key, value)

    return config


def _build_stage_config(
    spec: Dict[str, Any],
    state: Dict[str, Any],
    stage_index: int,
) -> Dict[str, Any]:
    stage = _stage_by_index(spec, stage_index)
    stage_type = stage["stage_type"]
    base_config_path = REPO_ROOT / stage["base_config"]
    config = _read_json(base_config_path)

    if stage_type == "reference":
        for key, value in stage.get("set", {}).items():
            _set_nested(config, key, value)
        return config

    reset_stage = _base_looped_reset_stage(spec)
    config = _read_json(REPO_ROOT / reset_stage["base_config"])
    for key, value in reset_stage.get("set", {}).items():
        _set_nested(config, key, value)

    accepted = set(state.get("accepted_stage_names", []))
    for prior_name in _stage_names_before(spec, stage_index):
        if prior_name not in accepted:
            continue
        prior_stage = next(
            candidate for candidate in spec["stages"] if candidate["name"] == prior_name
        )
        for key, value in prior_stage.get("set", {}).items():
            _set_nested(config, key, value)

    if stage_type in {"reset_incumbent", "additive"}:
        for key, value in stage.get("set", {}).items():
            _set_nested(config, key, value)
        return config

    raise ValueError(f"Unsupported stage_type={stage_type}")


def cmd_show(args: argparse.Namespace) -> int:
    spec = _load_spec(Path(args.spec).resolve())
    print(json.dumps(spec, indent=2))
    return 0


def cmd_build_stage(args: argparse.Namespace) -> int:
    spec = _load_spec(Path(args.spec).resolve())
    state = _load_state(Path(args.state_file).resolve())
    stage = _stage_by_index(spec, args.stage_index)
    seed_values = spec.get("seed_values", [])
    if args.seed_index < 0 or args.seed_index >= len(seed_values):
        raise SystemExit(f"seed-index must be in [0, {len(seed_values) - 1}]")

    config = _build_stage_config(spec, state, args.stage_index)
    seed_value = int(seed_values[args.seed_index])
    out_path = Path(args.out).resolve()
    _write_json(out_path, config)

    payload = {
        "stage_index": args.stage_index,
        "stage_name": stage["name"],
        "seed_index": args.seed_index,
        "seed": seed_value,
        "config_path": str(out_path),
        "train_flags": spec.get("train_flags", {}),
        "run_name": f"s{args.stage_index:02d}_{stage['name']}_seed{seed_value}",
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_record_decision(args: argparse.Namespace) -> int:
    spec = _load_spec(Path(args.spec).resolve())
    state_path = Path(args.state_file).resolve()
    state = _load_state(state_path)
    stage = _stage_by_index(spec, args.stage_index)
    decision = args.decision
    metric_mean = args.metric_mean

    history_entry = {
        "stage_index": args.stage_index,
        "stage_name": stage["name"],
        "decision": decision,
        "metric_mean": metric_mean,
        "notes": args.notes,
    }

    if args.baseline_epoch10_train_loss is not None:
        state["baseline_epoch10_train_loss"] = args.baseline_epoch10_train_loss
        state["baseline_guard_threshold"] = (
            args.baseline_epoch10_train_loss * args.loss_guard_max_ratio
        )

    if stage["stage_type"] == "reference":
        state["baseline_reference_stage"] = stage["name"]
    elif stage["stage_type"] == "reset_incumbent":
        state["looped_incumbent_stage"] = stage["name"]
        state["looped_incumbent_metric_mean"] = metric_mean
        state["accepted_stage_names"] = []
    elif stage["stage_type"] == "additive":
        accepted = list(state.get("accepted_stage_names", []))
        if decision == "accept":
            if stage["name"] not in accepted:
                accepted.append(stage["name"])
            state["accepted_stage_names"] = accepted
            state["looped_incumbent_stage"] = stage["name"]
            state["looped_incumbent_metric_mean"] = metric_mean
        elif decision == "reject":
            state["accepted_stage_names"] = accepted
        else:
            raise SystemExit("Additive stages only support accept or reject decisions")
    else:
        raise SystemExit(f"Unsupported stage_type={stage['stage_type']}")

    state.setdefault("stage_history", []).append(history_entry)
    _write_json(state_path, state)
    print(json.dumps(state, indent=2))
    return 0


def cmd_build_incumbent(args: argparse.Namespace) -> int:
    spec = _load_spec(Path(args.spec).resolve())
    state = _load_state(Path(args.state_file).resolve())
    out_path = Path(args.out).resolve()
    config = _build_incumbent_config(spec, state)
    _write_json(out_path, config)
    payload = {
        "config_path": str(out_path),
        "looped_incumbent_stage": state.get("looped_incumbent_stage"),
        "accepted_stage_names": state.get("accepted_stage_names", []),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive looped ablation ladder helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Print the ladder spec")
    show_parser.add_argument(
        "--spec",
        default=".codex/configs/looped_transformer_ablation.json",
        help="Path to the ladder spec JSON",
    )
    show_parser.set_defaults(func=cmd_show)

    build_parser = subparsers.add_parser(
        "build-stage", help="Build a config for one stage and one seed"
    )
    build_parser.add_argument(
        "--spec",
        default=".codex/configs/looped_transformer_ablation.json",
        help="Path to the ladder spec JSON",
    )
    build_parser.add_argument(
        "--state-file",
        default="experiment_state/looped_transformer_ablation_state.json",
        help="Path to the tracked ladder state JSON",
    )
    build_parser.add_argument("--stage-index", type=int, required=True)
    build_parser.add_argument("--seed-index", type=int, required=True)
    build_parser.add_argument("--out", required=True, help="Output config path")
    build_parser.set_defaults(func=cmd_build_stage)

    record_parser = subparsers.add_parser(
        "record-decision", help="Update the tracked ladder state after a stage decision"
    )
    record_parser.add_argument(
        "--spec",
        default=".codex/configs/looped_transformer_ablation.json",
        help="Path to the ladder spec JSON",
    )
    record_parser.add_argument(
        "--state-file",
        default="experiment_state/looped_transformer_ablation_state.json",
        help="Path to the tracked ladder state JSON",
    )
    record_parser.add_argument("--stage-index", type=int, required=True)
    record_parser.add_argument(
        "--decision",
        required=True,
        choices=["reference", "accept", "reject", "reset"],
        help="Decision recorded for the stage",
    )
    record_parser.add_argument("--metric-mean", type=float, required=True)
    record_parser.add_argument("--notes", default="", help="Short decision note")
    record_parser.add_argument(
        "--baseline-epoch10-train-loss",
        type=float,
        help="Baseline epoch-10 primary train loss used by the train-loss guard",
    )
    record_parser.add_argument(
        "--loss-guard-max-ratio",
        type=float,
        default=1.2,
        help="Guard ratio used to derive the kill threshold from the baseline",
    )
    record_parser.set_defaults(func=cmd_record_decision)

    incumbent_parser = subparsers.add_parser(
        "build-incumbent",
        help="Build the current accepted looped incumbent config for downstream scaling",
    )
    incumbent_parser.add_argument(
        "--spec",
        default=".codex/configs/looped_transformer_ablation.json",
        help="Path to the ladder spec JSON",
    )
    incumbent_parser.add_argument(
        "--state-file",
        default="experiment_state/looped_transformer_ablation_state.json",
        help="Path to the tracked ladder state JSON",
    )
    incumbent_parser.add_argument(
        "--out",
        required=True,
        help="Output path for the best-architecture config JSON",
    )
    incumbent_parser.set_defaults(func=cmd_build_incumbent)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
