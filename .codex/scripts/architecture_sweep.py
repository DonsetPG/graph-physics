#!/usr/bin/env python3
"""Plan, execute, and summarize architecture sweeps for graphphysics training."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_RUN_RE = re.compile(r"wandb/run-[^\s/]+")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def _latest_epoch_record(path: Path) -> Optional[Dict[str, Any]]:
    latest: Optional[Dict[str, Any]] = None
    if not path.exists():
        return None
    for record in _iter_jsonl(path):
        latest = record
    return latest


def _loss_guard_trigger(
    epoch_metrics_path: Path,
    baseline: float,
    kill_epoch: int,
    max_ratio: float,
) -> Optional[Dict[str, Any]]:
    latest = _latest_epoch_record(epoch_metrics_path)
    if latest is None:
        return None

    epoch = latest.get("epoch")
    metric_value = latest.get("primary_train_metric")
    metric_name = latest.get("primary_train_metric_name")
    if not isinstance(epoch, int) or epoch < kill_epoch:
        return None
    if not isinstance(metric_value, (int, float)):
        return None

    threshold = baseline * max_ratio
    if float(metric_value) < threshold:
        return None

    return {
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "baseline": float(baseline),
        "threshold": float(threshold),
        "max_ratio": float(max_ratio),
    }


def _terminate_process(process: subprocess.Popen, timeout_seconds: float = 30.0) -> int:
    process.terminate()
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def _safe_slug(value: Any) -> str:
    normalized = str(value).lower().replace(" ", "-")
    normalized = re.sub(r"[^a-z0-9_-]+", "-", normalized)
    return re.sub(r"-+", "-", normalized).strip("-")


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _match_subset(candidate: Dict[str, Any], subset: Dict[str, Any]) -> bool:
    for key, value in subset.items():
        if key not in candidate or candidate[key] != value:
            return False
    return True


def _parse_kv(values: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid key=value pair: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in pair: {item}")
        value: Any = raw_value
        lowered = raw_value.lower()
        if lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        else:
            try:
                if raw_value.isdigit() or (raw_value.startswith("-") and raw_value[1:].isdigit()):
                    value = int(raw_value)
                else:
                    value = float(raw_value)
            except ValueError:
                value = raw_value
        parsed[key] = value
    return parsed


def _build_combinations(matrix: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not matrix:
        yield {}
        return

    keys = list(matrix.keys())
    values = [matrix[key] for key in keys]
    for combo_values in itertools.product(*values):
        yield dict(zip(keys, combo_values))


def _run_id(index: int, combo: Dict[str, Any]) -> str:
    parts = [f"run_{index:04d}"]
    for key, value in combo.items():
        short_key = key.split(".")[-1]
        parts.append(f"{_safe_slug(short_key)}-{_safe_slug(value)}")
        if len("_".join(parts)) > 120:
            break
    return "_".join(parts)


def _build_train_command(
    python_bin: str,
    config_path: str,
    train_flags: Dict[str, Any],
) -> List[str]:
    cmd = [python_bin, "-m", "graphphysics.train", f"--training_parameters_path={config_path}"]
    for key in sorted(train_flags):
        value = train_flags[key]
        if isinstance(value, bool):
            cmd.append(f"--{key}" if value else f"--no{key}")
        else:
            cmd.append(f"--{key}={value}")
    return cmd


def cmd_plan(args: argparse.Namespace) -> int:
    base_config_path = Path(args.base_config).resolve()
    grid_path = Path(args.grid).resolve()

    base_config = _read_json(base_config_path)
    grid = _read_json(grid_path)

    sweep_name = grid.get("sweep_name", f"sweep_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    matrix = grid.get("matrix", {})
    constants = grid.get("constants", {})
    excludes = grid.get("excludes", [])
    train_flags = dict(grid.get("train_flags", {}))
    train_flags.update(_parse_kv(args.train_flag))

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (REPO_ROOT / ".codex" / "experiments" / "sweeps" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_safe_slug(sweep_name)}")
    configs_dir = out_dir / "configs"
    results_dir = out_dir / "results"
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    skipped = 0
    for combo in _build_combinations(matrix):
        if any(_match_subset(combo, exclude) for exclude in excludes):
            skipped += 1
            continue

        run_index = len(entries)
        run_id = _run_id(run_index, combo)
        run_config = json.loads(json.dumps(base_config))

        for key, value in constants.items():
            _set_nested(run_config, key, value)
        for key, value in combo.items():
            _set_nested(run_config, key, value)

        config_path = configs_dir / f"{run_id}.json"
        _write_json(config_path, run_config)

        entry = {
            "index": run_index,
            "run_id": run_id,
            "combo": combo,
            "config_path": str(config_path),
            "train_flags": train_flags,
            "sweep_name": sweep_name,
        }
        entries.append(entry)

    manifest_path = out_dir / "manifest.jsonl"
    for entry in entries:
        _append_jsonl(manifest_path, entry)

    csv_path = out_dir / "manifest.csv"
    combo_keys = sorted(matrix.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "run_id", "config_path", *combo_keys])
        for entry in entries:
            writer.writerow([
                entry["index"],
                entry["run_id"],
                entry["config_path"],
                *[entry["combo"].get(key, "") for key in combo_keys],
            ])

    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "sweep_name": sweep_name,
        "base_config": str(base_config_path),
        "grid": str(grid_path),
        "out_dir": str(out_dir),
        "num_runs": len(entries),
        "num_skipped": skipped,
        "manifest": str(manifest_path),
        "results_dir": str(results_dir),
    }
    _write_json(out_dir / "plan-summary.json", summary)

    print(json.dumps(summary, indent=2))
    print("\nNext steps:")
    print(f"  python .codex/scripts/architecture_sweep.py run-shard --manifest {manifest_path} --shard-index 0 --num-shards 4")
    print(f"  python .codex/scripts/architecture_sweep.py summarize --manifest {manifest_path}")
    return 0


def cmd_run_shard(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    entries = list(_iter_jsonl(manifest_path))
    if not entries:
        print(f"No entries found in {manifest_path}", file=sys.stderr)
        return 1

    shard_index = args.shard_index
    num_shards = args.num_shards
    if shard_index < 0 or shard_index >= num_shards:
        print("Invalid shard index.", file=sys.stderr)
        return 1

    selected = [entry for entry in entries if entry["index"] % num_shards == shard_index]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]

    if not selected:
        print("No runs assigned to this shard.")
        return 0

    results_dir = Path(args.results_dir).resolve() if args.results_dir else manifest_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    journal_path = results_dir / f"journal-shard{shard_index:02d}-of-{num_shards:02d}.jsonl"

    extra_flags = _parse_kv(args.train_flag)

    success = 0
    failed = 0
    for entry in selected:
        run_id = entry["run_id"]
        run_log_path = results_dir / f"{run_id}.log"
        epoch_metrics_path = results_dir / f"{run_id}.epochs.jsonl"
        started = dt.datetime.now()

        train_flags = dict(entry.get("train_flags", {}))
        train_flags.setdefault("model_save_name", run_id)
        train_flags.update(extra_flags)

        cmd = _build_train_command(
            python_bin=args.python,
            config_path=entry["config_path"],
            train_flags=train_flags,
        )

        env = os.environ.copy()
        env.setdefault("WANDB_NAME", run_id)
        env.setdefault("WANDB_RUN_GROUP", entry.get("sweep_name", "architecture-sweep"))

        result_code = 0
        duration = 0.0
        wandb_run_dir = None
        guard_trigger = None

        if args.dry_run:
            run_log_path.write_text("[DRY RUN] " + shlex.join(cmd) + "\n", encoding="utf-8")
        else:
            start_ts = time.time()
            if epoch_metrics_path.exists():
                epoch_metrics_path.unlink()
            with run_log_path.open("w", encoding="utf-8") as log_f:
                log_f.write(shlex.join(cmd) + "\n\n")
                env["GRAPH_PHYSICS_EPOCH_METRICS_PATH"] = str(epoch_metrics_path)
                process = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
                while True:
                    poll_code = process.poll()
                    if args.loss_guard_baseline is not None:
                        guard_trigger = _loss_guard_trigger(
                            epoch_metrics_path=epoch_metrics_path,
                            baseline=args.loss_guard_baseline,
                            kill_epoch=args.loss_guard_epoch,
                            max_ratio=args.loss_guard_max_ratio,
                        )
                        if guard_trigger is not None:
                            log_f.write(
                                "\n[LOSS GUARD] "
                                f"epoch={guard_trigger['epoch']} "
                                f"metric={guard_trigger['metric_name']} "
                                f"value={guard_trigger['metric_value']:.6f} "
                                f"threshold={guard_trigger['threshold']:.6f}\n"
                            )
                            log_f.flush()
                            result_code = _terminate_process(
                                process, timeout_seconds=args.loss_guard_terminate_timeout
                            )
                            break
                    if poll_code is not None:
                        result_code = poll_code
                        break
                    time.sleep(args.loss_guard_poll_seconds)
            duration = time.time() - start_ts

            log_content = run_log_path.read_text(encoding="utf-8", errors="ignore")
            match = WANDB_RUN_RE.search(log_content)
            if match:
                wandb_run_dir = match.group(0)

        failure_reason = "train_loss_guard" if guard_trigger is not None else None
        status = "success" if (args.dry_run or (result_code == 0 and guard_trigger is None)) else "failed"
        if status == "success":
            success += 1
        else:
            failed += 1

        finished = dt.datetime.now()
        record = {
            "run_id": run_id,
            "index": entry["index"],
            "shard_index": shard_index,
            "num_shards": num_shards,
            "status": status,
            "return_code": result_code,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": round(duration, 3),
            "command": cmd,
            "log_path": str(run_log_path),
            "epoch_metrics_path": str(epoch_metrics_path),
            "wandb_run_dir": wandb_run_dir,
            "config_path": entry["config_path"],
            "combo": entry.get("combo", {}),
            "failure_reason": failure_reason,
        }
        if guard_trigger is not None:
            record["guard_epoch"] = guard_trigger["epoch"]
            record["guard_metric_name"] = guard_trigger["metric_name"]
            record["guard_metric_value"] = guard_trigger["metric_value"]
            record["guard_baseline"] = guard_trigger["baseline"]
            record["guard_threshold"] = guard_trigger["threshold"]
            record["guard_max_ratio"] = guard_trigger["max_ratio"]
        _append_jsonl(journal_path, record)

        print(f"[{status}] {run_id} ({entry['index']})")

        if status == "failed" and args.fail_fast:
            print("Stopping early due to --fail-fast")
            break

    print(
        json.dumps(
            {
                "journal": str(journal_path),
                "runs_total": len(selected),
                "success": success,
                "failed": failed,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0 if (failed == 0 or args.dry_run) else 1


def _extract_metrics(summary_path: Path, preferred_metrics: List[str]) -> Dict[str, Any]:
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    metrics: Dict[str, Any] = {}
    for metric in preferred_metrics:
        if metric in payload:
            metrics[metric] = payload[metric]

    if not metrics:
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
    return metrics


def cmd_summarize(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    entries = {entry["run_id"]: entry for entry in _iter_jsonl(manifest_path)}
    if not entries:
        print(f"No entries found in {manifest_path}", file=sys.stderr)
        return 1

    results_dir = Path(args.results_dir).resolve() if args.results_dir else manifest_path.parent / "results"
    journal_files = sorted(results_dir.glob("journal-shard*-of-*.jsonl"))
    if not journal_files:
        print(f"No journal files found in {results_dir}", file=sys.stderr)
        return 1

    latest_records: Dict[str, Dict[str, Any]] = {}
    for journal in journal_files:
        for record in _iter_jsonl(journal):
            run_id = record["run_id"]
            prev = latest_records.get(run_id)
            if prev is None or record.get("finished_at", "") >= prev.get("finished_at", ""):
                latest_records[run_id] = record

    rows: List[Dict[str, Any]] = []
    combo_keys: List[str] = sorted({k for entry in entries.values() for k in entry.get("combo", {}).keys()})

    for run_id, entry in sorted(entries.items(), key=lambda kv: kv[1]["index"]):
        record = latest_records.get(run_id, {})
        status = record.get("status", "not_run")
        wandb_run_dir = record.get("wandb_run_dir")
        metrics: Dict[str, Any] = {}
        if wandb_run_dir:
            summary_path = REPO_ROOT / wandb_run_dir / "files" / "wandb-summary.json"
            metrics = _extract_metrics(summary_path, args.metric)

        row: Dict[str, Any] = {
            "index": entry["index"],
            "run_id": run_id,
            "status": status,
            "return_code": record.get("return_code"),
            "duration_seconds": record.get("duration_seconds"),
            "config_path": entry["config_path"],
            "log_path": record.get("log_path"),
            "epoch_metrics_path": record.get("epoch_metrics_path"),
            "wandb_run_dir": wandb_run_dir,
            "failure_reason": record.get("failure_reason"),
            "guard_epoch": record.get("guard_epoch"),
            "guard_metric_name": record.get("guard_metric_name"),
            "guard_metric_value": record.get("guard_metric_value"),
            "guard_baseline": record.get("guard_baseline"),
            "guard_threshold": record.get("guard_threshold"),
        }
        for key in combo_keys:
            row[key] = entry.get("combo", {}).get(key)
        for key, value in metrics.items():
            row[f"metric.{key}"] = value

        rows.append(row)

    summary_csv = results_dir / "summary.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_key = f"metric.{args.primary_metric}" if args.primary_metric else None
    ranked = rows
    if metric_key and any(metric_key in row and isinstance(row[metric_key], (int, float)) for row in rows):
        ranked = [
            row
            for row in rows
            if row.get("status") == "success" and isinstance(row.get(metric_key), (int, float))
        ]
        ranked.sort(key=lambda row: row[metric_key], reverse=args.maximize)

    out = {
        "summary_csv": str(summary_csv),
        "num_runs": len(rows),
        "num_success": sum(1 for row in rows if row.get("status") == "success"),
        "num_failed": sum(1 for row in rows if row.get("status") == "failed"),
        "num_not_run": sum(1 for row in rows if row.get("status") == "not_run"),
    }
    _write_json(results_dir / "summary.json", out)

    print(json.dumps(out, indent=2))
    if ranked and metric_key:
        print("\nTop runs:")
        for row in ranked[: args.top_k]:
            print(f"  {row['run_id']}: {metric_key}={row[metric_key]} status={row.get('status')}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Architecture sweep helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Generate sweep manifest and configs")
    plan_parser.add_argument("--base-config", required=True, help="Base JSON config file")
    plan_parser.add_argument("--grid", required=True, help="Grid JSON file")
    plan_parser.add_argument("--out-dir", help="Output directory for sweep artifacts")
    plan_parser.add_argument(
        "--train-flag",
        action="append",
        default=[],
        help="Override training flag key=value (repeatable)",
    )
    plan_parser.set_defaults(func=cmd_plan)

    run_parser = subparsers.add_parser("run-shard", help="Execute one shard of a sweep")
    run_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    run_parser.add_argument("--shard-index", type=int, required=True, help="Current shard index")
    run_parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    run_parser.add_argument("--results-dir", help="Directory for logs and journals")
    run_parser.add_argument("--python", default=sys.executable, help="Python binary to use")
    run_parser.add_argument("--dry-run", action="store_true", help="Only emit commands")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop after first failure")
    run_parser.add_argument("--max-runs", type=int, help="Only execute first N runs for this shard")
    run_parser.add_argument(
        "--loss-guard-baseline",
        type=float,
        help="Kill a run once its primary train loss at/after --loss-guard-epoch exceeds baseline * --loss-guard-max-ratio",
    )
    run_parser.add_argument(
        "--loss-guard-epoch",
        type=int,
        default=10,
        help="Epoch at which the train-loss guard becomes active",
    )
    run_parser.add_argument(
        "--loss-guard-max-ratio",
        type=float,
        default=1.2,
        help="Maximum allowed ratio vs baseline train loss before killing the run",
    )
    run_parser.add_argument(
        "--loss-guard-poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval for epoch-metric guard checks",
    )
    run_parser.add_argument(
        "--loss-guard-terminate-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait after terminate() before kill() when the guard fires",
    )
    run_parser.add_argument(
        "--train-flag",
        action="append",
        default=[],
        help="Extra training flag key=value (repeatable)",
    )
    run_parser.set_defaults(func=cmd_run_shard)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize sweep outputs")
    summarize_parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    summarize_parser.add_argument("--results-dir", help="Directory with journal-shard files")
    summarize_parser.add_argument(
        "--metric",
        action="append",
        default=["val_loss", "val_l2loss", "train_l2loss"],
        help="Preferred metric key in wandb-summary.json (repeatable)",
    )
    summarize_parser.add_argument("--primary-metric", help="Metric key to rank by")
    summarize_parser.add_argument("--maximize", action="store_true", help="Rank metric descending")
    summarize_parser.add_argument("--top-k", type=int, default=5, help="Number of top runs to print")
    summarize_parser.set_defaults(func=cmd_summarize)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
