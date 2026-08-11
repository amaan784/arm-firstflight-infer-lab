"""Work-dir resolution: a non-editable install must never write inside site-packages."""

from pathlib import Path

from firstflight import util


def test_work_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRSTFLIGHT_WORK_DIR", str(tmp_path))
    assert util.repo_root() == tmp_path


def test_repo_root_finds_checkout(monkeypatch):
    monkeypatch.delenv("FIRSTFLIGHT_WORK_DIR", raising=False)
    root = util.repo_root()
    assert (root / "configs").is_dir() and (root / "pyproject.toml").exists()


def test_dirs_follow_work_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRSTFLIGHT_WORK_DIR", str(tmp_path))
    monkeypatch.delenv("FIRSTFLIGHT_MODEL_DIR", raising=False)
    for d in (util.model_dir(), util.results_dir(), util.reports_dir()):
        assert Path(d).is_relative_to(tmp_path)
