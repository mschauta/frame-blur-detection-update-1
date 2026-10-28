"""The evidence files are strict JSON, and the numbers quoted in README.md / METHOD.md are the ones in them."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = sorted((ROOT / "reproduction").glob("*.json"))


def strict(path: Path):
    def refuse(token):
        raise ValueError(f"{path.name}: {token} is not JSON")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


@pytest.mark.parametrize("path", EVIDENCE, ids=lambda p: p.name)
def test_evidence_is_strict_json(path):
    strict(path)


def test_readme_quotes_the_evidence():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = strict(ROOT / "reproduction" / "validation_epoch29.json")
    heldout = strict(ROOT / "reproduction" / "heldout_films.json")
    corpus = strict(ROOT / "reproduction" / "corpus_checks.json")
    assert f"| frames | {validation['frames']['auc']:.3f} |" in readme
    small = strict(ROOT / "reproduction" / "validation_small_epoch29.json")
    kept = " / ".join(f"{100 * small['frames_at_thresholds'][f'sharp_kept_r{r}']:.1f}" for r in (90, 95, 98))
    assert f"| frames, ConvNeXt-Small | {small['frames']['auc']:.3f} | {kept}% |" in readme
    assert small["photos"]["auc"] == validation["photos"]["auc"] == 1.0
    th = " / ".join(f"{small['thresholds'][f'r{r}']:.4f}" for r in (90, 95, 98))
    assert f"| 29 (last of 30) | {th} | {small['frames']['auc']:.3f} |" in readme
    groups = {g["group"]: g for g in heldout["groups"]}
    for name, row in (("validation films (3)", "validation (3)"), ("test films (3)", "test (3)"),
                      ("validation + test (6 films)", "all six")):
        g = groups[name]
        assert f"| {row} | {g['n_sharp']:,} | {g['n_blur']:,} | {g['auc']:.3f} |" in readme
        low, high = g["auc_ci95"]
        assert f"{low:.3f}–{high:.3f}" in readme
    films = [f["auc"] for f in heldout["films"]]
    assert f"Per film the AUC is {min(films):.3f}–{max(films):.3f}" in readme
    assert f"All {corpus['scored_frames']:,} frames of the {corpus['videos']} videos" in readme
    assert f"For {corpus['two_releases']['films']} films released twice" in readme
    sc = corpus["scene_consistency"]
    assert f"median AUC of {sc['median_auc_model']:.3f}" in readme and f"reach {sc['median_auc_vlm_labels']:.3f}" in readme


def test_method_explains_its_counts():
    method = (ROOT / "METHOD.md").read_text(encoding="utf-8")
    corpus = strict(ROOT / "reproduction" / "corpus_checks.json")
    labelled = corpus["vlm_labelled_scored"]
    assert f"{labelled:,} when the corpus checks were computed" in method
    assert f"1,019,201 of them labelled" in method and 1_019_201 - labelled == 5_040 and "5,040" in method
    assert "32 films" in method and f"covers {corpus['two_releases']['films']} films" in method


def test_heldout_intervals_contain_their_auc():
    heldout = strict(ROOT / "reproduction" / "heldout_films.json")
    for item in heldout["films"] + [g for g in heldout["groups"] if "auc_ci95" in g]:
        low, high = item["auc_ci95"]
        assert low <= item["auc"] <= high
