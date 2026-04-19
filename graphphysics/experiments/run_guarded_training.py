#!/usr/bin/env python3
"""Run one graphphysics training job with an optional epoch-based train-loss guard."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_RUN_RE = re.compile(r"wandb/run-[^\s/]+")


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _latest_epoch_record(path: Path) -> Optional[Dict[str, Any]]:
    latest = None
    for record in _iter_jsonl(path) or ():
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


def _build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "graphphysics.train",
        f"--training_parameters_path={args.training_parameters_path}",
        f"--project_name={args.project_name}",
        f"--num_epochs={args.num_epochs}",
        f"--batch_size={args.batch_size}",
        f"--init_lr={args.init_lr}",
        f"--warmup={args.warmup}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
    ]
    if args.no_edge_feature:
        cmd.append("--no_edge_feature")
    if args.model_save_name:
        cmd.append(f"--model_save_name={args.model_save_name}")
    for flag in args.extra_flag:
        cmd.append(flag)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one guarded training job")
    parser.add_argument("--training-parameters-path", required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--init-lr", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--epoch-metrics-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--result-json", help="Optional JSON output with run status")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-save-name")
    parser.add_argument("--no-edge-feature", action="store_true")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--loss-guard-baseline", type=float)
    parser.add_argument("--loss-guard-epoch", type=int, default=10)
    parser.add_argument("--loss-guard-max-ratio", type=float, default=1.2)
    parser.add_argument("--loss-guard-poll-seconds", type=float, default=30.0)
    parser.add_argument("--loss-guard-terminate-timeout", type=float, default=30.0)
    args = parser.parse_args()

    epoch_metrics_path = Path(args.epoch_metrics_path).resolve()
    log_path = Path(args.log_path).resolve()
    result_json = Path(args.result_json).resolve() if args.result_json else None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if epoch_metrics_path.exists():
        epoch_metrics_path.unlink()

    cmd = _build_command(args)
    env = os.environ.copy()
    env["WANDB_NAME"] = args.wandb_name
    env["GRAPH_PHYSICS_EPOCH_METRICS_PATH"] = str(epoch_metrics_path)

    guard_trigger = None
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(shlex.join(cmd) + "\n\n")
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
                    return_code = _terminate_process(
                        process, timeout_seconds=args.loss_guard_terminate_timeout
                    )
                    break
            if poll_code is not None:
                return_code = poll_code
                break
            time.sleep(args.loss_guard_poll_seconds)

    duration_seconds = time.time() - started_at
    log_content = log_path.read_text(encoding="utf-8", errors="ignore")
    wandb_run_dir = None
    match = WANDB_RUN_RE.search(log_content)
    if match:
        wandb_run_dir = match.group(0)

    status = "success" if (return_code == 0 and guard_trigger is None) else "failed"
    payload: Dict[str, Any] = {
        "status": status,
        "return_code": return_code,
        "duration_seconds": round(duration_seconds, 3),
        "command": cmd,
        "log_path": str(log_path),
        "epoch_metrics_path": str(epoch_metrics_path),
        "wandb_run_dir": wandb_run_dir,
        "failure_reason": "train_loss_guard" if guard_trigger is not None else None,
    }
    if guard_trigger is not None:
        payload.update(
            {
                "guard_epoch": guard_trigger["epoch"],
                "guard_metric_name": guard_trigger["metric_name"],
                "guard_metric_value": guard_trigger["metric_value"],
                "guard_baseline": guard_trigger["baseline"],
                "guard_threshold": guard_trigger["threshold"],
            }
        )

    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
