"""Load config.yaml once and expose it with paths resolved against the project root."""
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/common/config.py -> project root


@lru_cache(maxsize=1)
def load_config(path: Path = PROJECT_ROOT / "config.yaml") -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    config["paths"] = {name: PROJECT_ROOT / value for name, value in config["paths"].items()}
    return config
