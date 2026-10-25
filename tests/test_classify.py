"""classify.py with a stand-in model: results as the run goes, resume after an interruption, dry run."""
import csv

import numpy as np
import pytest
import torch
from PIL import Image

import classify


class FakeModel:
    """Blur score = the image's mean grey level / 255 (a stand-in; no weights needed)."""

    def __init__(self, fail_on=None):
        self.fail_on, self.calls = fail_on, 0

    def __call__(self, x):
        self.calls += 1
        if self.fail_on is not None and self.calls == self.fail_on:
            raise KeyboardInterrupt
        p = x.float().mean() / 255
        return {"image": torch.logit(p.clamp(1e-6, 1 - 1e-6)).view(1)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, grey in enumerate([10, 200, 30, 220, 40]):            # threshold 0.5: 1 and 3 blurred
        Image.fromarray(np.full((16, 16, 3), grey, np.uint8)).save(frames / f"f{i}.png")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"stand-in")
    state = {"model": FakeModel()}
    monkeypatch.setattr(classify, "load_detector", lambda path, device: (
        state["model"], {"thresholds": {"r90": 0.5, "r95": 0.5, "r98": 0.5}, "input_description": "test", "epoch": 0}))
    args = ["--weights", str(weights), "--input", str(frames), "--output", str(tmp_path / "out"), "--device", "cpu"]
    return {"frames": frames, "out": tmp_path / "out", "args": args, "state": state}


def rows(out, name="results.csv"):
    with open(out / name, newline="", encoding="utf-8") as fh:
        return {r["file"]: r["decision"] for r in csv.DictReader(fh)}


def test_copy_run_writes_every_row(setup):
    classify.main(setup["args"])
    assert rows(setup["out"]) == {"f0.png": "sharp", "f1.png": "blur", "f2.png": "sharp", "f3.png": "blur", "f4.png": "sharp"}
    assert sorted(p.name for p in (setup["out"] / "blur").iterdir()) == ["f1.png", "f3.png"]
    assert len(list(setup["frames"].iterdir())) == 5


def test_dry_run_copies_nothing_and_a_real_run_can_follow(setup):
    classify.main(setup["args"] + ["--dry-run"])
    assert len(rows(setup["out"], "results_dry_run.csv")) == 5
    assert not (setup["out"] / "sharp").exists() and not (setup["out"] / "blur").exists()
    classify.main(setup["args"])
    assert rows(setup["out"]) == rows(setup["out"], "results_dry_run.csv")
    assert len(list((setup["out"] / "sharp").iterdir())) == 3


def test_interrupted_move_run_resumes_without_losses_or_duplicates(setup):
    setup["state"]["model"] = FakeModel(fail_on=3)
    with pytest.raises(KeyboardInterrupt):
        classify.main(setup["args"] + ["--move"])
    assert len(rows(setup["out"])) == 2                          # the rows so far are on disk
    # an image moved just before the interruption, without its row
    (setup["frames"] / "f2.png").rename(setup["out"] / "sharp" / "f2.png")
    setup["state"]["model"] = FakeModel()
    with pytest.raises(SystemExit, match="--resume"):
        classify.main(setup["args"] + ["--move"])
    classify.main(setup["args"] + ["--move", "--resume"])
    with open(setup["out"] / "results.csv", newline="", encoding="utf-8") as fh:
        names = [r["file"] for r in csv.DictReader(fh)]
    assert sorted(names) == [f"f{i}.png" for i in range(5)] and len(names) == 5
    assert not list(setup["frames"].iterdir())
    assert sorted(p.name for p in (setup["out"] / "sharp").iterdir()) == ["f0.png", "f2.png", "f4.png"]
