# R4S-1 report: ConvNeXt-Small against ConvNeXt-Base, the same recipe, epoch 29

The protocol is [PROTOCOL.md](PROTOCOL.md) (English translation) and [PROTOCOL_hu.md](PROTOCOL_hu.md) (the registered
text, SHA-256 in [PROTOCOL.sha256](PROTOCOL.sha256)). Every measure of the protocol is reported below; nothing was
added or dropped. There is no deviation from the protocol.

## Summary

- **On the held-out films no AUC difference is shown.** Small − Base over all six films: −0.0007 [−0.0018, +0.0004].
  The point value is below zero in both groups. Per film the Small is below on three films and above on three.
- **At r95 and r98 the Small keeps fewer sharp frames:** 1.4 and 2.1 points fewer, with the same share of blurred
  frames caught. At r90 the two are equal.
- **The Small follows the motion measurement slightly more closely:** Spearman 0.705 against 0.699, difference
  +0.006 [+0.001, +0.011].
- **On the photo ladder the two are nearly the same.** The Small's 50% flip comes 0.1 px earlier (9.04 against
  9.13 px at r95); its pair order over 0–3 px is 65.9% against 61.7%.
- **The Small is 1.35× faster** (11.5 against 15.4 ms per 1080p frame on an RTX 5090).
- **Both controls match:** the rescored Base reproduces the published held-out AUCs and the published photo ladder.

No acceptance threshold was fixed in advance, so this report states the differences and leaves the choice of model
open.

## 1. What was run

- The published weight files, loaded with the published loader (`blurdetect`, this repository at `62734f4`):
  `round3_base_p99_ep029.pt` (release v1.0, SHA-256 `9840160f…d0e6`) and `round3_small_p99_ep029.pt` (release
  v1.1, `b715a163…d48b`). Both files' thresholds are in [data/thresholds.json](data/thresholds.json).
- One machine (RTX 5090), GPU, ConvNeXt trunk under bfloat16 autocast, fingerprint in float32, no mask. Both models
  scored the same loaded array of every frame and photo.
- Held-out films: every one of the 114,687 labelled frames, batch 1. Photo ladder: 300 photos × 17 lengths, batch 8.

## 2. Controls

- **Held-out films.** The rescored Base reproduces the published per-film AUCs (`reproduction/heldout_films.json`) to
  within 0.00002 (largest difference, test film A: 0.95419 against 0.95417). Its logits correlate with the published
  corpus scores at Pearson ≥ 0.999999 on every film. The analysis also checks that the corpus scores in the table give
  the published AUC exactly (to 1e−9) and the published class counts.
- **Photo ladder.** The rescored Base equals `reproduction/photo_ladder_epoch29.json`: every share, every median logit
  and every flip length is the same (largest difference 0).

## 3. Held-out films (step 1)

AUC with its 95% interval; the difference is paired (the same shot resamples for both models).

| | sharp / blurred frames | Small | Base | Small − Base |
|---|---|---|---|---|
| validation films (3) | 37,133 / 17,886 | 0.9578 [0.9498, 0.9649] | 0.9584 [0.9513, 0.9651] | −0.0007 [−0.0022, +0.0007] |
| test films (3) | 33,877 / 25,791 | 0.9654 [0.9531, 0.9748] | 0.9663 [0.9544, 0.9750] | −0.0009 [−0.0026, +0.0007] |
| all six | 71,010 / 43,677 | 0.9620 [0.9540, 0.9689] | 0.9626 [0.9548, 0.9692] | −0.0007 [−0.0018, +0.0004] |

Per film:

| film | shots | Small | Base | Small − Base |
|---|---|---|---|---|
| test film A | 378 | 0.9544 | 0.9542 | +0.0002 |
| test film B | 446 | 0.9756 | 0.9764 | −0.0008 |
| test film C | 121 | 0.9792 | 0.9804 | −0.0012 |
| validation film A | 396 | 0.9591 | 0.9586 | +0.0006 |
| validation film B | 358 | 0.9513 | 0.9536 | −0.0023 |
| validation film C | 151 | 0.9646 | 0.9645 | +0.0001 |

Read from the cell map instead (M, the relu mean over the inner cells), the six-film AUC is 0.9612 for the Small and
0.9603 for the Base.

**Operating points**, all six films, each model at its own thresholds (sharp kept / blurred caught):

| | Small | Base |
|---|---|---|
| r90 | 88.8% / 90.9% | 88.9% / 91.0% |
| r95 | 79.0% / 95.1% | 80.4% / 95.0% |
| r98 | 65.8% / 97.4% | 67.9% / 97.3% |

The validation and test groups show the same pattern ([data/results_heldout.json](data/results_heldout.json)).

**Agreement at r95**, all six films:
- The two models make the same decision on 96.2% of the frames (Cohen's κ 0.923); the Spearman correlation of their
  logits is 0.976.
- Of the 4,415 frames where they disagree, the Small says blurred and the Base sharp on 2,739, the reverse on 1,676.
- On the disagreements the VLM label sides with the Small on 39.5% and with the Base on 60.5%.

## 4. The motion measurement as a witness (step 2)

The measured regional motion (`region_px`) exists for every labelled frame. Spearman correlation with the image
logit, all six films:

| Small | Base | Small − Base |
|---|---|---|
| 0.705 [0.678, 0.730] | 0.699 [0.673, 0.724] | +0.006 [+0.001, +0.011] |

Read from the cell map (M): 0.715 and 0.716. On Update 2's evaluation sample (11,571 frames): 0.722 and 0.722.

## 5. Photo ladder (step 3)

The protocol fixes no interval for the ladder, so these are point values (300 photos per length):

| | Small | Base |
|---|---|---|
| 50% flip (r90 / r95 / r98) | 9.10 / 9.04 / 8.99 px | 9.20 / 9.13 / 9.07 px |
| called blurred at 8 px, r95 | 11.7% | 7.3% |
| called blurred at 10 px, r95 | 85.3% | 83.0% |
| false alarms at 0–6 px, r95 | 1 image (6 px) | 2 images (one at 5 px, one at 6 px) |
| pair order 0–3 px | 65.9% | 61.7% |
| pair order 3–16 px | 99.2% | 99.2% |
| strictly increasing 0–1–2–3 px | 25.3% | 21.3% |
| median logit at 0 px | −21.5 | −22.6 |
| median logit at 30 px | +17.9 | +21.6 |

On the same image the two models' logits correlate at Spearman 0.963. The Small's logits span a narrower range: the
median per-image difference Small − Base is +1.1 to +1.7 at 0–8 px and −2.1 to −3.6 from 12 px on.

## 6. Speed (step 4)

[data/speed.json](data/speed.json): RTX 5090, one 1920 × 1080 frame, the `classify.py` path, 30 + 200 rounds:

| | batch 1 | batch 8 | peak memory (batch 1) |
|---|---|---|---|
| Base | 15.44 ms/image, 64.8 images/s | 15.57 ms/image, 64.2 images/s | 805 MiB |
| Small | 11.46 ms/image, 87.2 images/s | 11.37 ms/image, 88.0 images/s | 708 MiB |

The Small is 1.35× faster (1.37× at batch 8). Batch 8 gives neither model a gain: one 1080p frame already fills the
GPU.

## 7. Interpretation by the fixed rules

- **AUC:** the paired intervals contain 0 in all three groups, so no difference is shown on this measure. The point
  values (−0.0007 to −0.0009) are on the Base's side, as in the in-training validation (0.9612 against 0.9621).
- **Operating points:** at r95 and r98 the Small keeps 1.4 and 2.1 points fewer sharp frames at the same catch rate.
  The protocol fixes no interval for this measure.
- **Motion witness:** the interval of the difference excludes 0. The Small agrees slightly more with the measurement;
  the size is small (+0.006).
- **Photo ladder:** nearly the same, with a slightly earlier flip and slightly better order at 0–3 px for the Small.
- **Speed:** 1.35×.

The Small is close enough to the Base that these tests do not separate them in accuracy, and it is faster. Whether
that is enough is a decision, not a result.

## 8. Limitations

- **The intervals describe the shots, not the films.** The bootstrap resamples shots within each of the six films.
  The intervals therefore describe the uncertainty from the sample of shots in these six films. They do not promise
  the same difference on other films: six films are too few to estimate how much the difference varies between films.
- **The VLM label is not ground truth.** It is Update 1's training signal (on other films). "The label sides with the
  Base" on disagreements means agreement with the teacher, not correctness.
- **The validation films** were seen by the per-epoch evaluation during training, though not used to pick an epoch;
  the test films are reported separately and show the same pattern.
- **The photo ladder** is an exact signal, but only uniform camera motion with a single H.264 CRF 23 coding, and no
  interval.
- **Not a pure trunk comparison.** The Small was trained with the leak-free mask. No test here uses a mask, so the
  inputs were bit-identical, but the training inputs differed on masked samples.
- **The coded frame type** affects both models: see Update 2's
  [experiments/r4k2_gop_photo](https://github.com/mschauta/frame-blur-detection-update-2/tree/main/experiments/r4k2_gop_photo).
  In that test (not this protocol) the Small gave more false alarms than the Base on sharp photos coded with a group
  of pictures (r95: 2.4% against 1.7% of 115,200 frames).
- **Speed** was measured on one GPU, for the model alone.

## 9. Files and rerunning

| file | what |
|---|---|
| `PROTOCOL_hu.md`, `PROTOCOL.sha256` | the registered protocol (Hungarian) and its SHA-256 |
| `PROTOCOL.md` | its English translation |
| `score.py` | the scoring as run (speed, held-out frames, photo ladder), with the publication edits in `MANIFEST.json`; needs the non-public corpus |
| `analyse.py` | the two reports, from the published tables |
| `data/heldout_frames.csv.gz` | one row per labelled held-out frame: film label, role, shot number, class, Update 2 evaluation-sample flag, `region_px`, the Base's published corpus score, and both models' logit and three cell-map readouts. No frame ids, times, images or film names |
| `data/ladder_logits.npz` | both models' logits per photo (300) and length (17), with each photo's direction; no photo ids |
| `data/thresholds.json` | both weight files' identity and thresholds |
| `data/results_heldout.json`, `data/results_ladder.json` | the reports |
| `data/speed.json` | the speed measurement |
| `data/score_heldout.log`, `data/score_ladder.log` | the scoring logs (Hungarian messages) |
| `MANIFEST.json` | SHA-256 of every file, the publication edits |

```
pip install -r requirements.txt
python experiments/r4s1_small_base/analyse.py heldout
python experiments/r4s1_small_base/analyse.py ladder
```

From the two tables, `analyse.py` reproduces both result files byte for byte. The held-out report checks itself
against `reproduction/heldout_films.json` and stops if the corpus scores do not give the published AUCs.

## 10. Log

Local time (CET), translated from the Hungarian log.

- 2026-10-28 23:15: the protocol's hash recorded (the file up to the "## Napló" heading; the log goes below it).
- 2026-10-28 23:16: step 4 (speed) done.
- 2026-10-28 23:17: step 1 scoring started (both models, 114,687 frames).
- 2026-10-29 01:24: step 1 scoring done. Report: the control matches (see section 2); the paired comparison uses the
  rescored Base.
- 2026-10-29 01:27: step 3 scoring done. Report: the Base matches the published ladder exactly.
- 2026-10-29: results written to the log as above; no deviation.
