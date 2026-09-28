"""실험 버전 관리: run 이름, 디렉터리, git 정보, metrics.json · summary.md.

한 번 실행할 때마다 <experiment_name>_v001, _v002 ... 처럼 새 run이 생긴다.
  <artifacts_root>/<run>/     Git 제외. adapter/, checkpoints/, logs/, predictions/, run_metadata.json
  <experiments_root>/<run>/   Git 관리. config.yaml, metrics.json, summary.md, loss_history.csv
artifacts_root는 설정 · CLI로 바꿀 수 있어서 저장소 밖(예: D:/tripfit_artifacts)에 둘 수도 있다.
"""

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT, resolve_path

METRICS_SCHEMA_VERSION = 1
ADAPTER_WEIGHT_SUFFIXES = (".safetensors", ".bin")


@dataclass(frozen=True)
class RunPaths:
    name: str
    artifact_dir: Path
    experiment_dir: Path

    @property
    def adapter_dir(self) -> Path:
        return self.artifact_dir / "adapter"

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifact_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.artifact_dir / "logs"

    @property
    def prediction_dir(self) -> Path:
        return self.artifact_dir / "predictions"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_info() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, encoding="utf-8"
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status)}


def next_run_name(experiment_name: str, roots: list[Path]) -> str:
    pattern = re.compile(rf"^{re.escape(experiment_name)}_v(\d+)$")
    used = [int(m[1]) for root in roots if root.is_dir() for p in root.iterdir() if (m := pattern.match(p.name))]
    return f"{experiment_name}_v{max(used, default=0) + 1:03d}"


def run_roots(config: dict) -> tuple[Path, Path]:
    return resolve_path(config["output"]["artifacts_root"]), resolve_path(config["output"]["experiments_root"])


def create_run(config: dict, run_name: str | None = None) -> RunPaths:
    """새 run 디렉터리를 만든다. run_name을 주면 그 이름을 쓴다 (이미 있으면 멈춘다)."""
    artifacts_root, experiments_root = run_roots(config)
    name = run_name or next_run_name(config["experiment_name"], [artifacts_root, experiments_root])
    paths = RunPaths(name, artifacts_root / name, experiments_root / name)
    if paths.experiment_dir.exists() or paths.artifact_dir.exists():
        raise FileExistsError(f"run {name}이 이미 있습니다")
    for directory in (paths.artifact_dir, paths.log_dir, paths.experiment_dir):
        directory.mkdir(parents=True)
    return paths


def open_run(config: dict, run_name: str) -> RunPaths:
    """이미 있는 run (예: 학습한 run에 평가 결과를 더할 때)."""
    artifacts_root, experiments_root = run_roots(config)
    paths = RunPaths(run_name, artifacts_root / run_name, experiments_root / run_name)
    if not paths.experiment_dir.exists():
        raise FileNotFoundError(f"run이 없습니다: {paths.experiment_dir}")
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(config: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def adapter_size_mb(adapter_dir: Path) -> float | None:
    """어댑터 weight 파일 크기 합 (tokenizer · 설정 파일 제외)."""
    files = [p for p in adapter_dir.rglob("*") if p.is_file() and p.suffix in ADAPTER_WEIGHT_SUFFIXES]
    return round(sum(p.stat().st_size for p in files) / 2**20, 3) if files else None


def write_loss_history(log_history: list[dict], path: Path) -> None:
    """Trainer log_history에서 loss 관련 값만 CSV로. Git에 올릴 만큼 작다."""
    keys = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    rows = [{k: entry.get(k) for k in keys} for entry in log_history if "loss" in entry or "eval_loss" in entry]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def base_metrics(config: dict, run: RunPaths, *, data_info: dict, git: dict) -> dict:
    """모든 run(Base · LoRA · QLoRA)이 같은 모양으로 쓰는 metrics.json의 뼈대."""
    method = config["method"]
    hyper = None
    if method != "base":
        hyper = {"lora": config["lora"], "training": config["training"], "quantization": config.get("quantization") if method == "qlora" else None}
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "run_name": run.name,
        "timestamp": now_iso(),
        "git": git,
        "model_id": config["model"]["name_or_path"],
        "method": method,
        "seed": config["seed"],
        "precision": None,
        "data": data_info,
        "hyperparameters": hyper,
        "training": None,
        "evaluation": None,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(metrics: dict, path: Path) -> None:
    lines = [f"# {metrics['run_name']}", "", "| 항목 | 값 |", "| --- | --- |"]
    rows = [
        ("method", metrics["method"]),
        ("model_id", metrics["model_id"]),
        ("git commit", (metrics["git"].get("commit") or "")[:12] + (" (dirty)" if metrics["git"].get("dirty") else "")),
        ("dataset", f"{metrics['data'].get('dataset_version')} / split {metrics['data'].get('split_version')}"),
        ("train / val / test", " / ".join(str(metrics["data"]["counts"].get(s)) for s in ("train", "validation", "test"))),
    ]
    training = metrics.get("training") or {}
    rows += [
        ("final train loss", _fmt(training.get("final_train_loss"))),
        ("eval loss", _fmt(training.get("eval_loss"))),
        ("train time (min)", _fmt(training.get("duration_sec") and training["duration_sec"] / 60, 1)),
        ("peak VRAM (GB)", _fmt(training.get("peak_vram_reserved_gb"), 2)),
        ("adapter size (MB)", _fmt(training.get("adapter_size_mb"), 1)),
    ]
    overall = (metrics.get("evaluation") or {}).get("overall") or {}
    rows += [
        (f"[{(metrics.get('evaluation') or {}).get('split', '-')}] {key}", _fmt(overall.get(key)))
        for key in (
            "json_valid_rate", "aspect_f1", "aspect_macro_f1", "attribute_accuracy", "sentiment_accuracy",
            "evidence_in_source_rate", "evidence_exact_match", "evidence_overlap_f1", "traveler_context_f1",
        )
    ]  # fmt: skip
    lines += [f"| {k} | {v} |" for k, v in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
