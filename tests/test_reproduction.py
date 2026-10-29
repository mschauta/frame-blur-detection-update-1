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


EXPERIMENT = ROOT / "experiments" / "r4s1_small_base"


@pytest.mark.parametrize("path", sorted((EXPERIMENT / "data").glob("*.json")), ids=lambda p: p.name)
def test_experiment_results_are_strict_json(path):
    strict(path)


def test_experiment_manifest_matches_the_files():
    import hashlib
    manifest = strict(EXPERIMENT / "MANIFEST.json")
    on_disk = {p.relative_to(EXPERIMENT).as_posix() for p in EXPERIMENT.rglob("*")
               if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts}
    assert set(manifest["files"]) == on_disk
    for name, digest in manifest["files"].items():
        data = (EXPERIMENT / name).read_bytes()
        if name.endswith((".md", ".py", ".json", ".log", ".sha256")):
            # git may check text files out with CRLF line ends (core.autocrlf); the hashes are of the LF files
            data = data.replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_experiment_protocol_is_the_registered_one():
    import hashlib
    digest = hashlib.sha256((EXPERIMENT / "PROTOCOL_hu.md").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest in (EXPERIMENT / "PROTOCOL.sha256").read_text(encoding="utf-8")


def test_experiment_controls_match_the_published_evidence():
    heldout = strict(EXPERIMENT / "data" / "results_heldout.json")
    published = {f["film"]: f["auc"] for f in strict(ROOT / "reproduction" / "heldout_films.json")["films"]}
    for c in heldout["control_rescored_base_vs_published"]:
        assert c["published_auc"] == published[c["film"]]
        assert abs(c["rescored_base_auc"] - c["published_auc"]) < 5e-5
    ladder = strict(EXPERIMENT / "data" / "results_ladder.json")["control_base_vs_published"]
    assert ladder["max_abs_share_difference"] == 0 and ladder["max_abs_median_logit_difference"] == 0


def test_readme_and_report_quote_the_experiment():
    flat = lambda p: " ".join(p.read_text(encoding="utf-8").split())                    # noqa: E731
    readme, report = flat(ROOT / "README.md"), flat(EXPERIMENT / "REPORT.md")
    num = lambda v, d=4: f"{v:+.{d}f}".replace("-", "−")                                 # noqa: E731
    six = {g["group"]: g for g in strict(EXPERIMENT / "data" / "results_heldout.json")["groups"]}["validation + test (6 films)"]
    d = six["auc_difference_small_minus_base"]
    quoted = f"{num(d['value'])} [{num(d['ci95'][0])}, {num(d['ci95'][1])}]"
    assert quoted in readme and quoted in report
    assert f"the AUC is {six['auc']['small']:.3f} for the Small and {six['auc']['base']:.3f} for the Base" in readme
    op = six["operating_points"]
    assert f"keeps {100 * op['small_r95']['sharp_kept']:.1f}% of the sharp frames, the Base {100 * op['base_r95']['sharp_kept']:.1f}%" in readme
    rho = six["motion"]["spearman_logit_region_px"]
    assert f"Spearman {rho['small']:.3f} against {rho['base']:.3f}" in readme
    flips = {m: v["flip_L_50pct"]["r95"] for m, v in strict(EXPERIMENT / "data" / "results_ladder.json")["models"].items()}
    assert f"{flips['small']:.2f} px against {flips['base']:.2f} px" in readme
    speed = strict(EXPERIMENT / "data" / "speed.json")
    ms = {m: speed["models"][m]["batch_1"]["ms_per_image"] for m in ("small", "base")}
    assert f"{speed['small_over_base_batch_1']:.2f}× faster: {ms['small']:.1f} against {ms['base']:.1f} ms" in readme
