from src.common import artifacts


def test_new_run_dir_creates_layout(tmp_path, monkeypatch):
    real = artifacts.load_config()
    monkeypatch.setattr(artifacts, "load_config", lambda: {**real, "paths": {**real["paths"], "artifacts_dir": tmp_path}})
    d = artifacts.new_run_dir("smoke", "claude-sonnet-5-5", {"encounter_ids": ["e1"]})
    assert d.parent == tmp_path and d.name.startswith("smoke_claude-sonnet-5-5_")
    assert (d / "raw").is_dir() and (d / "config.json").exists()
    assert '"encounter_ids"' in (d / "meta.json").read_text()
