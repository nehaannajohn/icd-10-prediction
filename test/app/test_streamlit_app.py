import pytest
from streamlit.testing.v1 import AppTest

from src.common.config import PROJECT_ROOT, load_config

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
needs_examples = pytest.mark.skipif(
    not all((PROJECT_ROOT / load_config()["app"]["examples_dir"] / f"{e}.json").exists()
            for e in load_config()["app"]["examples"]), reason="cached examples not built")


@needs_examples
def test_app_renders_first_example_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.button[0].disabled                                    # empty transcript -> Run disabled
    assert len(at.expander) >= 2                                    # per-code evidence + transcript + method


@needs_examples
def test_run_without_aws_env_shows_error(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].input("Patient has hypertension.").run()
    at.button[0].click().run()
    assert any("AWS_PROFILE" in e.value for e in at.error)


def test_app_starts_without_cached_examples(tmp_path, monkeypatch):
    import streamlit as st

    from src.common import config as cfgmod
    st.cache_data.clear()                      # load_examples is cached process-wide by earlier tests
    real = cfgmod.load_config()
    monkeypatch.setattr(cfgmod, "load_config",
                        lambda: {**real, "app": {**real["app"], "examples_dir": str(tmp_path / "none")}})
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("No cached examples" in i.value for i in at.info)
