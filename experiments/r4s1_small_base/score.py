"""R4S-1 scoring: the round3 ConvNeXt-Small against the round3 ConvNeXt-Base (PROTOCOL.md).

As run on 2026-10-28/29 (blurtrainer/tools/r4s1_small_base.py), with the publication edits listed in MANIFEST.json:
the two report steps are in analyse.py, and the etalon albums' names are withheld. It needs the non-public corpus
(master.db, the v9 frame table, the frames and the photos) and two unpublished modules of the training code
(eval_v9: the held-out frame set and its VLM codes; probe_round3: the Update 1 photo ladder), so it does not rerun
outside the author's machine; analyse.py reruns from the published tables.

Both published weight files, loaded with the published Update 1 loader (blurdetect), so the numbers belong to the
published files. Read-only on master.db, the v9 frame table and the dataset. Each image is loaded once and both models
score the same array: GPU, ConvNeXt trunk under bfloat16 autocast, fingerprint float32.

usage: python score.py speed
       python score.py heldout           (resumable)
       python score.py ladder
Output: runs/round3_convnext_small_imagenet_p99_v8_v1/probes/r4s1_*
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import eval_v9 as E                                          # noqa: E402  held-out frame set, codes, AUC
import probe_round3 as P                                     # noqa: E402  the Update 1 ladder: photos, angles, lengths

U1 = ROOT.parent / "frame-blur-detection-update-1"
_spec = importlib.util.spec_from_file_location("blurdetect_u1", U1 / "blurdetect" / "model.py")
U1_MODEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U1_MODEL)

WEIGHTS = {"base": (U1 / "weights" / "round3_base_p99_ep029.pt",
                    "9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6"),
           "small": (U1 / "weights" / "round3_small_p99_ep029.pt",
                     "b715a163f9c6927464d97b7002967c8361c6a0de2b87e3103dc72a7ba733d48b")}
OUT = ROOT / "runs" / "round3_convnext_small_imagenet_p99_v8_v1" / "probes"
HELDOUT = OUT / "r4s1_heldout.csv"
DEVICES = OUT / "r4s1_heldout_devices.json"
STATS = ("logit", "m_inner", "max_inner", "mean_inner")
PROBE = Path("F:/RGB Mesh/demo_processed/frames_1080_h264/07_2b_00170.png")      # one 1920 x 1080 frame (not published)
ETALON_ALBUMS: tuple[str, ...] = ()           # publication edit: the two etalon albums' bin names, withheld
log = E.log


def load_models(device="cuda") -> dict:
    models = {}
    for name, (path, sha) in WEIGHTS.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == sha, (name, digest)
        model, meta = U1_MODEL.load_detector(path, device)
        assert meta["epoch"] == 29 and meta["input"] == "p99", meta
        models[name] = (model, meta)
    return models


def stats(out, k=0) -> dict:
    local = out["local"].float()[k, 0]
    inner = local[1:-1, 1:-1]
    return {"logit": float(out["image"].float()[k]), "m_inner": float(inner.clamp(min=0).mean()),
            "max_inner": float(inner.max()), "mean_inner": float(inner.mean())}


def forward(model, x):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x)


# ------------------------------------------------------------------------------------------------- speed
def run_speed(a):
    models = load_models()
    rgb = torch.from_numpy(np.asarray(Image.open(PROBE).convert("RGB")).copy()).permute(2, 0, 1)[None].cuda()
    res = {"frame": f"{rgb.shape[-1]}x{rgb.shape[-2]}", "gpu": torch.cuda.get_device_name(),
           "torch": torch.__version__, "path": "classify.py: the published loader, bfloat16 autocast, no mask",
           "warmup": a.warmup, "iterations": a.iters, "models": {}}
    for name, (model, _) in models.items():
        res["models"][name] = {}
        for bs in (1, 8):
            x = rgb.expand(bs, -1, -1, -1).contiguous()
            for _ in range(a.warmup):
                forward(model, x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(a.iters):
                forward(model, x)
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1) / a.iters
            res["models"][name][f"batch_{bs}"] = {"ms_per_batch": ms, "ms_per_image": ms / bs,
                                                  "images_per_s": 1000 * bs / ms,
                                                  "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20}
            log(f"{name} batch {bs}: {ms / bs:.2f} ms/kép, {1000 * bs / ms:.1f} kép/s, "
                f"csúcs {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    for bs in (1, 8):
        b, s = (res["models"][m][f"batch_{bs}"]["images_per_s"] for m in ("base", "small"))
        res[f"small_over_base_batch_{bs}"] = s / b
    (OUT / "r4s1_speed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"Small / Base: batch 1 {res['small_over_base_batch_1']:.2f}×, batch 8 {res['small_over_base_batch_8']:.2f}×")


# ------------------------------------------------------------------------------------------------- held-out
def heldout_rows():
    """Every frame of the six held-out films with an AUC class (Update 1's heldout_films.json): 114,687."""
    con = sqlite3.connect(f"file:{E.V9_DB}?mode=ro", uri=True)
    rows = con.execute("SELECT i, frame_id, rgb_path, video_id FROM frames WHERE split = 'heldout' ORDER BY i").fetchall()
    master = Path(json.loads(E.V9_DB.with_suffix(".json").read_text(encoding="utf-8"))["master_root"])
    db = sqlite3.connect(f"file:{master / 'master.db'}?mode=ro", uri=True)
    code = E._codes(db, [r[1] for r in rows])
    used = {vid: E.used_codes(db, vid) for vid in {r[3] for r in rows}}
    rows = [r for r in rows if code.get(r[1]) in used[r[3]]]
    assert len(rows) == 114_687, len(rows)
    return rows, master


def run_heldout(a):
    rows, master = heldout_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if HELDOUT.exists():
        with open(HELDOUT, newline="", encoding="utf-8") as fh:
            done = {int(r["i"]) for r in csv.DictReader(fh)}
    todo = [r for r in rows if r[0] not in done]
    log(f"címkés kimaradt frame-ek: {len(rows)}, kész {len(done)}, hátra {len(todo)}; Base + Small ep029, cuda")
    models = load_models()
    load = lambda p: np.asarray(Image.open(master / p).convert("RGB"))            # noqa: E731
    cols = ["i", "frame_id"] + [f"{m}_{k}" for m in WEIGHTS for k in STATS]
    new = not HELDOUT.exists()
    t0 = time.time()
    with open(HELDOUT, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(3) as pool:
        w = csv.DictWriter(fh, cols)
        if new:
            w.writeheader()
        ahead = [pool.submit(load, r[2]) for r in todo[:6]]
        for n, (i, fid, *_) in enumerate(todo):
            rgb = ahead.pop(0).result()
            if n + 6 < len(todo):
                ahead.append(pool.submit(load, todo[n + 6][2]))
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].cuda()
            row = {"i": i, "frame_id": fid}
            for name, (model, _) in models.items():
                row.update({f"{name}_{k}": f"{v:.7g}" for k, v in stats(forward(model, x)).items()})
            w.writerow(row)
            if (n + 1) % 500 == 0:
                fh.flush()
                left = (time.time() - t0) / (n + 1) * (len(todo) - n - 1)
                log(f"  {n + 1} / {len(todo)}, hátra kb. {left / 60:.0f} perc")
    devices = json.loads(DEVICES.read_text(encoding="utf-8")) if DEVICES.exists() else {}
    devices["cuda"] = devices.get("cuda", 0) + len(todo)
    DEVICES.write_text(json.dumps(devices, indent=1), encoding="utf-8")
    log(f"kész: {HELDOUT}")


# ------------------------------------------------------------------------------------------------- ladder
def run_ladder(a):
    from blurtrainer.data import read_index
    rows = [r for r in read_index(ROOT / "data" / "round3_index.csv") if r["source"] == "photo" and r["synth"] == "sharp"
            and r["split"] in ("val", "test") and r["shot_bin"] in ETALON_ALBUMS]
    rows.sort(key=lambda r: P.h("ladder", r["sample_id"]))
    rows = rows[: a.photos]
    jobs = [(r, P.ANGLES[P.h("angle", r["sample_id"]) % len(P.ANGLES)]) for r in rows]
    log(f"fotólétra: {len(rows)} etalonfotó × {len(P.LADDER_L)} elmosási hossz, H.264 CRF 23, Base + Small ep029")
    models = load_models()
    res = {m: np.zeros((len(rows), len(P.LADDER_L))) for m in WEIGHTS}
    t0 = time.time()
    with ProcessPoolExecutor(10) as ex:
        for i, images in enumerate(P.bounded_map(ex, P._ladder_job, jobs)):
            for name, (model, _) in models.items():
                for j0 in range(0, len(images), 8):               # batch 8, as the Base's original ladder
                    x = torch.from_numpy(np.stack(images[j0:j0 + 8])).permute(0, 3, 1, 2).contiguous().cuda()
                    res[name][i, j0:j0 + 8] = forward(model, x)["image"].float().cpu().numpy()
            if (i + 1) % 50 == 0:
                log(f"  {i + 1} / {len(rows)} fotó, hátra kb. {(time.time() - t0) / (i + 1) * (len(rows) - i - 1) / 60:.0f} perc")
    np.savez_compressed(OUT / "r4s1_ladder.npz", L=np.array(P.LADDER_L), angle=np.array([j[1] for j in jobs]),
                        sample_id=np.array([r["sample_id"] for r in rows]), **res)
    log(f"kész: {OUT / 'r4s1_ladder.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("speed", "heldout", "ladder"))
    ap.add_argument("--photos", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"speed": run_speed, "heldout": run_heldout, "ladder": run_ladder}[a.cmd](a)


if __name__ == "__main__":
    main()
