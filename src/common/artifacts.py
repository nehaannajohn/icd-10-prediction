"""Run folders under artifacts/: <stage>_<model>_<YYYYMMDD_HHMMSS>/ with config snapshot, meta, raw/."""
import datetime
import json
import subprocess
from pathlib import Path

from src.common.config import PROJECT_ROOT, load_config


def git_commit() -> str | None:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return out.stdout.strip() or None


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def new_run_dir(stage: str, model: str, meta: dict) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = load_config()["paths"]["artifacts_dir"] / f"{stage}_{model}_{ts}"
    (run_dir / "raw").mkdir(parents=True)
    write_json(run_dir / "config.json", load_config())
    write_json(run_dir / "meta.json", {"stage": stage, "model": model, "timestamp": ts,
                                       "git_commit": git_commit(), **meta})
    return run_dir
