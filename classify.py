"""Sort images into sharp / blur with a published weight file.

    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted
    python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted --recall 98 --move

(the weight file is a release asset of this repository; put it into weights/, see README.md)

Every image of --input (png, jpg, jpeg, bmp, tif, tiff, webp; not recursive) is scored and copied (or moved) to
<output>/sharp or <output>/blur. The decision threshold is the one stored with the weights: it catches 90, 95 or 98%
of the blurred frames of the validation split (--recall). A higher recall is stricter: fewer blurred frames pass as
sharp, more sharp frames are rejected. <output>/results.csv lists every image with its blur score (the model's sigmoid output, not a calibrated
probability) and decision.

The models were trained for video frames (1080p, decoded from H.264, stored losslessly as PNG); see METHOD.md.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from blurdetect import load_detector

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--recall", type=int, choices=(90, 95, 98), default=95)
    ap.add_argument("--move", action="store_true", help="move the files instead of copying them")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, meta = load_detector(a.weights, a.device)
    thr = meta["thresholds"][f"r{a.recall}"]
    files = sorted(p for p in a.input.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    for d in ("sharp", "blur"):
        (a.output / d).mkdir(parents=True, exist_ok=True)
        if any((a.output / d).iterdir()):
            raise SystemExit(f"{a.output / d} is not empty; use a new or empty output folder")
    print(f"{len(files)} images, model input: {meta['input_description']}, epoch {meta['epoch']}, "
          f"threshold {thr:.6f} (catches {a.recall}% of the blurred validation frames), device {a.device}")
    rows = []
    with torch.no_grad():
        for i, f in enumerate(files, 1):
            x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).copy()).permute(2, 0, 1)[None].to(a.device)
            if a.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(x)["image"]
            else:
                logit = model(x)["image"]
            p = float(torch.sigmoid(logit.float()))
            label = "blur" if p >= thr else "sharp"
            (shutil.move if a.move else shutil.copy2)(f, a.output / label / f.name)
            rows.append({"file": f.name, "blur_score": f"{p:.6f}", "decision": label})
            print(f"[{i}/{len(files)}] {f.name}: {label} (score {p:.4f})")
    with open(a.output / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "blur_score", "decision"])
        w.writeheader()
        w.writerows(rows)
    n_sharp = sum(r["decision"] == "sharp" for r in rows)
    print(f"sharp {n_sharp}, blur {len(rows) - n_sharp} -> {a.output}")


if __name__ == "__main__":
    main()
