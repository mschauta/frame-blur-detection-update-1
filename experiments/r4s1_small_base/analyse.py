"""R4S-1 analysis: data/heldout_frames.csv.gz and data/ladder_logits.npz -> data/results_heldout.json,
data/results_ladder.json (PROTOCOL.md, steps 1-3).

The two inputs are the scores as `score.py` wrote them, without frame ids, film names or photo ids. The analysis is the
one that produced the reported numbers; from these files it reproduces both result files byte for byte.

usage: python experiments/r4s1_small_base/analyse.py heldout [--reps 2000]
       python experiments/r4s1_small_base/analyse.py ladder
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
MODELS = ("base", "small")
STATS = ("logit", "m_inner", "max_inner", "mean_inner")
SEED = 20261028
SCORING = ("device cuda (114,687 frames), the published Update 1 loader, ConvNeXt trunk under bfloat16 autocast, "
           "fingerprint float32, batch 1")


def auc(score: np.ndarray, blurred: np.ndarray) -> float:
    ranks = rankdata(score)
    n1 = int(blurred.sum())
    n0 = len(score) - n1
    return float((ranks[blurred].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x, y) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def kappa(a_, b_) -> float:
    po = np.mean(a_ == b_)
    pe = np.mean(a_) * np.mean(b_) + np.mean(~a_) * np.mean(~b_)
    return float((po - pe) / (1 - pe))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def thresholds() -> dict:
    return {m: v["thresholds"] for m, v in json.loads((DATA / "thresholds.json").read_text(encoding="utf-8")).items()}


# ------------------------------------------------------------------------------------------------- held-out
def load_films() -> tuple[dict, dict]:
    """{film label: arrays}, {film label: role}; the films in the order of the file."""
    cols = [f"{m}_{k}" for m in MODELS for k in STATS]
    raw: dict[str, list] = {}
    role = {}
    with gzip.open(DATA / "heldout_frames.csv.gz", "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw.setdefault(r["film"], []).append(r)
            role[r["film"]] = r["role"]
    films = {}
    for film, rows in raw.items():
        f = {"blurred": np.array([r["blurred"] == "1" for r in rows], bool),
             "base_corpus": np.array([float(r["base_corpus"]) for r in rows]),
             "shots": np.array([int(r["shot"]) for r in rows]),
             "region": np.array([float(r["region_px"]) if r["region_px"] else np.nan for r in rows]),
             "eval": np.array([r["eval_sample"] == "1" for r in rows])}
        for c in cols:
            f[c] = np.array([float(r[c]) for r in rows])
        films[film] = f
    return films, role


def run_heldout(a):
    published = json.loads((REPO / "reproduction" / "heldout_films.json").read_text(encoding="utf-8"))
    films, role = load_films()
    th = thresholds()
    cols = [f"{m}_{k}" for m in MODELS for k in STATS]
    pub = {p["film"]: p for p in published["films"]}
    check = []
    for film, f in films.items():
        p = pub[film]
        assert (p["n_sharp"], p["n_blur"]) == (int((~f["blurred"]).sum()), int(f["blurred"].sum())), film
        assert abs(auc(f["base_corpus"], f["blurred"]) - p["auc"]) < 1e-9, film
        check.append({"film": film, "published_auc": p["auc"], "rescored_base_auc": auc(f["base_logit"], f["blurred"]),
                      "pearson_rescored_vs_corpus_logit": float(np.corrcoef(
                          f["base_logit"], logit(f["base_corpus"]))[0, 1])})
    readouts = {"small": "small_logit", "base": "base_logit", "base_corpus": "base_corpus",
                "small_m_inner": "small_m_inner", "base_m_inner": "base_m_inner"}
    rng = np.random.default_rng(SEED)

    def group(members, motion_ci):
        keys = ("blurred", "region", "eval", "base_corpus", *cols)
        cat = {k: np.concatenate([films[v][k] for v in members]) for k in keys}
        b = cat["blurred"]
        out = {"n_sharp": int((~b).sum()), "n_blur": int(b.sum())}
        out["auc"] = {name: auc(cat[col], b) for name, col in readouts.items()}
        groups = [[np.flatnonzero(films[v]["shots"] == s) for s in range(films[v]["shots"].max() + 1)] for v in members]
        boot = {name: [] for name in readouts}
        rho = {"small": [], "base": []}
        for _ in range(a.reps):
            parts = []
            for v, g in zip(members, groups):
                pick = rng.integers(len(g), size=len(g))
                parts.append((v, np.concatenate([g[j] for j in pick])))
            bb = np.concatenate([films[v]["blurred"][ix] for v, ix in parts])
            if 0 < bb.sum() < len(bb):
                for name, col in readouts.items():
                    boot[name].append(auc(np.concatenate([films[v][col][ix] for v, ix in parts]), bb))
            if motion_ci:
                reg = np.concatenate([films[v]["region"][ix] for v, ix in parts])
                ok = np.isfinite(reg)
                for m in rho:
                    rho[m].append(spearman(np.concatenate([films[v][f"{m}_logit"][ix] for v, ix in parts])[ok], reg[ok]))
        out["auc_ci95"] = {name: np.percentile(v, [2.5, 97.5]).tolist() for name, v in boot.items()}
        d = np.array(boot["small"]) - np.array(boot["base"])
        out["auc_difference_small_minus_base"] = {"value": out["auc"]["small"] - out["auc"]["base"],
                                                  "ci95": np.percentile(d, [2.5, 97.5]).tolist()}
        out["auc_difference_rescored_minus_corpus_base"] = out["auc"]["base"] - out["auc"]["base_corpus"]
        ok = np.isfinite(cat["region"])
        out["motion"] = {"frames_with_measurement": int(ok.sum()),
                         "spearman_logit_region_px": {m: spearman(cat[f"{m}_logit"][ok], cat["region"][ok])
                                                      for m in ("small", "base")},
                         "spearman_m_inner_region_px": {m: spearman(cat[f"{m}_m_inner"][ok], cat["region"][ok])
                                                        for m in ("small", "base")}}
        ev = ok & cat["eval"]
        out["motion"]["eval_sample"] = {"frames": int(ev.sum()), "spearman_m_inner_region_px": {
            m: spearman(cat[f"{m}_m_inner"][ev], cat["region"][ev]) for m in ("small", "base")}}
        if motion_ci:
            dr = np.array(rho["small"]) - np.array(rho["base"])
            out["motion"]["spearman_logit_ci95"] = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in rho.items()}
            out["motion"]["spearman_logit_difference_small_minus_base"] = {
                "value": out["motion"]["spearman_logit_region_px"]["small"] - out["motion"]["spearman_logit_region_px"]["base"],
                "ci95": np.percentile(dr, [2.5, 97.5]).tolist()}
        out["operating_points"] = {}
        calls = {}
        for m in MODELS:
            p = 1 / (1 + np.exp(-cat[f"{m}_logit"]))
            for r, t in th[m].items():
                call = p >= t
                calls[(m, r)] = call
                out["operating_points"][f"{m}_{r}"] = {"threshold": t, "sharp_kept": float(np.mean(~call[~b])),
                                                       "blur_caught": float(np.mean(call[b]))}
        cs, cb = calls[("small", "r95")], calls[("base", "r95")]
        dis = cs != cb
        out["agreement_r95"] = {"same_decision": float(np.mean(~dis)), "cohen_kappa": kappa(cs, cb),
                                "spearman_logits": spearman(cat["small_logit"], cat["base_logit"]),
                                "disagreements": int(dis.sum()),
                                "of_which_the_vlm_label_sides_with_small": float(np.mean(cs[dis] == b[dis])),
                                "small_blurred_base_sharp": int((cs & ~cb).sum()),
                                "small_sharp_base_blurred": int((~cs & cb).sum())}
        return out

    held = list(films)
    val = [v for v in held if role[v] == "validation"]
    test = [v for v in held if role[v] == "test"]
    G = {"validation films (3)": group(val, False), "test films (3)": group(test, False),
         "validation + test (6 films)": group(held, True)}
    per_film = [{"film": v, "role": role[v], "shots": int(films[v]["shots"].max() + 1),
                 "n_sharp": int((~films[v]["blurred"]).sum()), "n_blur": int(films[v]["blurred"].sum()),
                 "auc": {name: auc(films[v][col], films[v]["blurred"]) for name, col in readouts.items()}}
                for v in held]
    out = {
        "what": ("R4S-1: every VLM-labelled frame of the 6 held-out films, scored once with both published round3 files "
                 "(ConvNeXt-Base v1.0, ConvNeXt-Small v1.1) in the same script; the Base's corpus scores of the same "
                 "frames as the control; Update 1's class definitions, film names withheld"),
        "readouts": {"small": "the Small's image logit", "base": "the Base's image logit, rescored here",
                     "base_corpus": "the Base's corpus score (master.db), as published in Update 1",
                     "small_m_inner / base_m_inner": "M read from the cell map: mean of relu(cell logit) over the inner cells"},
        "auc_ci95": (f"95% percentile interval from a cluster bootstrap over shots ({a.reps} resamples, shots drawn with "
                     "replacement within each film); the same resamples for every readout, so the difference is paired"),
        "control_rescored_base_vs_published": check,
        "groups": [{"group": k, **v} for k, v in G.items()], "films": per_film,
        "scoring": SCORING,
    }
    path = DATA / "results_heldout.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(path)


# ------------------------------------------------------------------------------------------------- ladder
def flip(Ls, share):
    """The length where the share first reaches 50%, interpolated linearly (as for Update 1)."""
    j = np.flatnonzero(share >= 0.5)
    if not len(j):
        return None
    if j[0] == 0:
        return float(Ls[0])
    x0, x1, y0, y1 = Ls[j[0] - 1], Ls[j[0]], share[j[0] - 1], share[j[0]]
    return float(x0 + (0.5 - y0) * (x1 - x0) / max(y1 - y0, 1e-9))


def pairs(v: np.ndarray, Ls: np.ndarray, mask: np.ndarray) -> dict:
    """Pairs of lengths (L_i < L_j) of the same photo: ordered (more blur scores higher), tied, reversed."""
    idx = np.flatnonzero(mask)
    up = tie = tot = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            d = v[:, idx[j]] - v[:, idx[i]]
            up += int((d > 0).sum()); tie += int((d == 0).sum()); tot += len(d)
    return {"ordered": up / tot, "tied": tie / tot, "reversed": (tot - up - tie) / tot}


def run_ladder(a):
    d = np.load(DATA / "ladder_logits.npz")
    Ls = d["L"]
    published = json.loads((REPO / "reproduction" / "photo_ladder_epoch29.json").read_text(encoding="utf-8"))
    th = thresholds()
    low, mid = Ls <= 3, (Ls >= 3) & (Ls <= 16)
    out = {"what": ("R4S-1 photo ladder: the Update 1 ladder (the same 300 unseen etalon photos, the same direction per "
                    "photo, L = 0 ... 30 px, H.264 CRF 23), scored with both published round3 files; blurred = sigmoid "
                    "score >= the file's own threshold"),
           "photos": int(d["base"].shape[0]), "L_px": [float(x) for x in Ls], "models": {}}
    for m in MODELS:
        lg = d[m]
        p = 1 / (1 + np.exp(-lg))
        per_L = [{"L": float(L), "median_logit": float(np.median(lg[:, j])),
                  **{f"blur_{k}": float(np.mean(p[:, j] >= t)) for k, t in th[m].items()}}
                 for j, L in enumerate(Ls)]
        flips = {k: flip(Ls, np.array([r[f"blur_{k}"] for r in per_L])) for k in th[m]}
        order = {"0_3_px": pairs(lg, Ls, low), "3_16_px": pairs(lg, Ls, mid)}
        mono = float(np.mean(np.all(np.diff(lg[:, [0, 2, 4, 6]], axis=1) > 0, axis=1)))
        out["models"][m] = {"thresholds": th[m], "per_L": per_L, "flip_L_50pct": flips,
                            "pair_order": order, "strictly_increasing_0_1_2_3": mono}
    base = out["models"]["base"]
    out["control_base_vs_published"] = {
        "max_abs_share_difference": max(abs(r[f"blur_{k}"] - q[f"blur_{k}"]) for r, q in zip(base["per_L"], published["per_L"])
                                        for k in ("r90", "r95", "r98")),
        "max_abs_median_logit_difference": max(abs(r["median_logit"] - q["median_logit"])
                                               for r, q in zip(base["per_L"], published["per_L"])),
        "published_flip": published["flip_L_50pct"], "rescored_flip": base["flip_L_50pct"],
        "published_pair_order_0_3": published["pair_order_accuracy_0_3_px"],
        "published_pair_order_3_16": published["pair_order_accuracy_3_16_px"],
        "published_strictly_increasing": published["strictly_increasing_0_1_2_3"]}
    lg_s, lg_b = d["small"], d["base"]
    out["same_photo_same_length"] = {"spearman_logits": spearman(lg_s.ravel(), lg_b.ravel()),
                                     "small_minus_base_median_logit_per_L": [float(np.median(lg_s[:, j] - lg_b[:, j]))
                                                                             for j in range(len(Ls))]}
    path = DATA / "results_ladder.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("heldout", "ladder"))
    ap.add_argument("--reps", type=int, default=2000, help="heldout: bootstrap resamples")
    a = ap.parse_args()
    {"heldout": run_heldout, "ladder": run_ladder}[a.cmd](a)


if __name__ == "__main__":
    main()
