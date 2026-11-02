from src.common.config import PROJECT_ROOT, load_config


def test_paths_resolved_against_project_root():
    paths = load_config()["paths"]
    assert all(p.is_absolute() and p.is_relative_to(PROJECT_ROOT) for p in paths.values())
    assert paths["icd10_dir"] == PROJECT_ROOT / "data/corpus/fy2025"


def test_split_sizes_cover_all_33_encounters():
    split = load_config()["split"]
    assert split["n_icl"] + split["n_tune"] + split["n_heldout"] == 33


def test_models_are_name_to_bedrock_id():
    models = load_config()["models"]
    assert len(models) == 4 and all(isinstance(v, str) and v for v in models.values())
