import os

from ml.inference.predict import get_default_checkpoint, predict_video


def test_no_experiment_checkpoint_is_selected_by_default(monkeypatch):
    monkeypatch.delenv("SIGNSPEAK_CHECKPOINT", raising=False)
    assert get_default_checkpoint() is None


def test_explicit_relative_checkpoint_is_resolved_from_project_root(monkeypatch, tmp_path):
    checkpoint = tmp_path / "production.pt"
    checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setenv("SIGNSPEAK_CHECKPOINT", str(checkpoint))
    assert get_default_checkpoint() == str(checkpoint)


def test_predict_fails_closed_without_production_checkpoint(monkeypatch):
    monkeypatch.delenv("SIGNSPEAK_CHECKPOINT", raising=False)
    try:
        predict_video("unused.webm")
        assert False, "predict_video should reject missing validated checkpoint"
    except RuntimeError as exc:
        assert "No validated SignSpeak production checkpoint" in str(exc)
