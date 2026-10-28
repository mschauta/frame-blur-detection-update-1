"""Sort images into sharp / blur with a published weight file.

    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted
    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted --recall 98 --move
    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted --dry-run
    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted --resume

(the weight file is a release asset of this repository; put it into weights/, see README.md)

Every image of --input (png, jpg, jpeg, bmp, tif, tiff, webp; not recursive) is scored and copied (or moved) to
<output>/sharp or <output>/blur. The decision threshold is the one stored with the weights: it catches 90, 95 or 98%
of the blurred frames of the validation split (--recall). A higher recall is stricter: fewer blurred frames pass as
sharp, more sharp frames are rejected. <output>/results.csv lists every image with its blur score (the model's sigmoid
output, not a calibrated probability) and decision.

results.csv is written as the run goes: a row is added right after its image was copied or moved, so an interrupted
run loses at most the image in hand. --resume continues such a run: images already in results.csv are skipped, and an
image found in <output>/sharp or <output>/blur without a row (moved just before the interruption) is scored again in
place and gets its row. <output>/run.json records the run's settings (the weight file's sha256, the threshold, the
input folder, copy or move); --resume continues only with the same settings, so one results.csv never mixes two
decision rules. --dry-run only scores, into results_dry_run.csv (settings in run_dry_run.json), and copies or moves
nothing, also with --resume; a real run in the same output folder afterwards starts from scratch.

Images are scored one at a time, as for the published numbers: batching and precision move the scores by a little.

The models were trained for video frames (1080p, decoded from H.264, stored losslessly as PNG); see METHOD.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from blurdetect import load_detector

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FIELDS = ["file", "blur_score", "decision"]
# The release assets of this repository (README.md, "Weights").
PUBLISHED_SHA256 = {"9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6": "round3_base_p99_ep029.pt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(model, path: Path, device: str) -> float:
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)["image"]
        else:
            logit = model(x)["image"]
    return float(torch.sigmoid(logit.float()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--recall", type=int, choices=(90, 95, 98), default=95)
    ap.add_argument("--move", action="store_true", help="move the files instead of copying them")
    ap.add_argument("--dry-run", action="store_true", help="only score, into results_dry_run.csv; copy or move nothing")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run into the same --output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    digest = sha256(a.weights)
    known = PUBLISHED_SHA256.get(digest)
    print(f"weights sha256 {digest}: " + (f"the published {known}" if known else "not a published release asset"))
    model, meta = load_detector(a.weights, a.device)
    thr = meta["thresholds"][f"r{a.recall}"]
    files = sorted(p for p in a.input.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    a.output.mkdir(parents=True, exist_ok=True)
    results = a.output / ("results_dry_run.csv" if a.dry_run else "results.csv")
    run_file = a.output / ("run_dry_run.json" if a.dry_run else "run.json")
    # What a resumed run must share with the run it continues.
    run = {"weights_sha256": digest, "threshold": thr, "input": str(a.input.resolve())}
    if not a.dry_run:
        run["move"] = a.move
    folders = {label: a.output / label for label in ("sharp", "blur")}
    done: set[str] = set()
    if a.resume:
        if results.exists() and not run_file.exists():
            raise SystemExit(f"{results} has no {run_file.name} (an earlier classify.py wrote it), so its settings "
                             "cannot be checked; use a new output folder")
        if run_file.exists():
            saved = json.loads(run_file.read_text(encoding="utf-8"))
            changed = [key for key in run.keys() | saved.keys() if saved.get(key) != run.get(key)]
            if changed:
                raise SystemExit(f"{run_file} records another {', '.join(sorted(changed))}; resume with the same "
                                 "settings, or use a new output folder")
        if results.exists():
            with open(results, newline="", encoding="utf-8") as fh:
                done = {row["file"] for row in csv.DictReader(fh)}
    else:
        if results.exists():
            raise SystemExit(f"{results} exists; use --resume to continue that run, or a new output folder")
        if not a.dry_run:
            for folder in folders.values():
                if folder.exists() and any(folder.iterdir()):
                    raise SystemExit(f"{folder} is not empty; use a new or empty output folder, or --resume")
    if not a.dry_run:
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
    run_file.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    # Moved just before an interruption: in an output folder, but without a row. A dry run leaves the folders alone.
    orphans = [p for folder in folders.values() if folder.exists() for p in sorted(folder.iterdir())
               if a.resume and not a.dry_run and p.is_file() and p.suffix.lower() in EXTS and p.name not in done]
    todo = [f for f in files if f.name not in done and f.name not in {p.name for p in orphans}]
    print(f"{len(todo) + len(orphans)} images to score ({len(done)} already in results.csv), model input: "
          f"{meta['input_description']}, epoch {meta['epoch']}, threshold {thr:.6f} (catches {a.recall}% of the "
          f"blurred validation frames), device {a.device}" + (", dry run" if a.dry_run else ""))
    new_file = not results.exists()
    counts = {"sharp": 0, "blur": 0}
    with open(results, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()
        for i, f in enumerate(orphans + todo, 1):
            p = score(model, f, a.device)
            label = "blur" if p >= thr else "sharp"
            if not a.dry_run:
                if f in orphans:
                    if f.parent != folders[label]:
                        shutil.move(f, folders[label] / f.name)
                else:
                    (shutil.move if a.move else shutil.copy2)(f, folders[label] / f.name)
            writer.writerow({"file": f.name, "blur_score": f"{p:.6f}", "decision": label})
            fh.flush()
            counts[label] += 1
            print(f"[{i}/{len(orphans) + len(todo)}] {f.name}: {label} (score {p:.4f})")
    print(f"sharp {counts['sharp']}, blur {counts['blur']} in this run -> {a.output}")


if __name__ == "__main__":
    main()
