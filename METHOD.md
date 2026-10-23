# Method — changes in Update 1

This file lists what differs from the original
[METHOD.md](https://github.com/mschauta/frame-blur-detection/blob/main/METHOD.md); the section numbers follow it.
Everything not mentioned here is as described there.

## 1. Task and scope

Unchanged: a sharp / blurred decision for video frames, with the cost of a blurred frame accepted as sharp counted as
the expensive error. Out of scope, as observed since: captions, overlaid graphics and animation (README, Limitations).

## 2. Material

| | original 100-epoch runs | Update 1 |
|---|---|---|
| videos | 37 processed files (36 groups) | 173 videos, all 1920 × 1080, H.264, 4:2:0 |
| frames | the labelled snapshot of those files | 5,010,602 frames from the dataset-builder pipeline; 1,019,201 of them labelled by the VLM teacher |
| photos | 48,797 originals (33,484 train / 7,656 validation / 7,657 test) | unchanged |

## 4. Labelling

Unchanged teacher and questions (the frozen VLM with three yes / no questions, codes such as `sss`, `ssb`, `bsb`,
`bbb`). The labelling covered more videos.

## 5. Frame selection (v8 rules)

- **Films as units.** Two releases of the same film (32 films, found by matching shot cut times) form one film; both
  releases give all their frames, so the same picture is seen at two bitrates, and both always share a split.
- **Classes.** Sharp = `sss`; blurred = `bbb` + `bsb`. Film-relative sharp: in a film where `sss / (sss + ssb)` is
  below 0.46, `ssb` counts as sharp too — there it measures as sharp as `sss` does in `sss`-dominant films, and without
  it such a film would give almost only blurred frames. `sbb`, `bss` and the rest are left out.
- **Content.** No subtitles or overlay text, visually processable, luminance std ≥ 1, entropy ≥ 1 bit, at most 90%
  over- or under-exposed.
- **The measurement only ranks inside a shot**, never on an absolute scale (close-ups move many pixels but keep the
  order). Per shot, ranked by the measured region motion, the sharp frames lose their largest 20% (ranked together, so
  the drop falls mostly on `ssb`), and each blurred code, ranked on its own, loses its smallest 10%; at least one frame
  per shot and class stays; unmeasured frames are left out.
- **No thinning of the training split**: every remaining labelled frame is a sample (neighbouring frames included);
  the trainer balances by sampling. Validation and test are 1:1 per film.
- **Split, by film, by hand**: 3 validation and 3 test films (complete, `sss`-dominant, varied); every other film is
  train.

| split | sharp | blurred |
|---|---|---|
| train | 349,624 | 240,039 |
| validation | 16,190 | 16,190 |
| test | 20,732 | 20,732 |

## 6. Photos

Unchanged: the same pool, the same split, blur in linear light, H.264 CRF 20 / 23 / 26 / 28, a 192-point grid of
(length, direction, CRF) per class walked differently for every photo and epoch; sharp 0–3 px, blurred 16–30 px. As
before, half of the photo samples and half of the frame samples carry a watermark validity mask.

## 7. Model input

The same p99-normalised signed `m09` fingerprint, but computed in **float32 outside autocast**. The original runs
computed its 3 × 3 convolution inside bf16 autocast, so their fingerprint was bf16-rounded and their weights learned on
that input. The loader reads the precision from the weight file (`fingerprint`: `fp32` or `bf16-rounded`); the
original weights keep their behaviour bit for bit.

## 8. Model, loss and sampling

| | Update 1 |
|---|---|
| trunk | ConvNeXt-Base, ImageNet-pretrained, to stride 16 (60.2 M parameters with the head) |
| head and pooling | as before: one logit per 16 px cell, masked log-sum-exp pooling with a learnable sharpness r (4.0 at the start, 0.86 at the end) |
| loss | image-level binary cross-entropy; on photos also the dense cell term (weight 0.5), every textured cell carrying the photo label |
| strata per epoch | 12,000 each: sharp frames, blurred frames, sharp photos, blurred photos (48,000 samples) |
| optimisation | AdamW, learning rate 1e-4 (trunk) / 1e-3 (head), weight decay 0.05, warmup 0.25 epoch, cosine decay to 1%, gradient clip 1.0, batch 8 × 2 accumulation |
| length | 30 epochs, 90,440 optimizer steps; each sharp training frame seen about once, each blurred about 1.5 times |
| sampling | resume-exact: a chain of permutations per stratum keyed by seed, stratum and pass, so every sample comes once before any comes twice; the last smaller batch of a size bucket is trained too |
| checkpoint | the last epoch (29), as in the original runs; fixed before the corpus evaluation |

## 9. Evaluation

- **Validation** during training on the v8 validation selection; the thresholds stored with the weights catch 90 /
  95 / 98% of its blurred frames.
- **Corpus scoring.** After the checkpoint was fixed, all 5,010,602 frames were scored once. The held-out evaluation
  uses every labelled frame of the six held-out films; the consistency checks (scene consistency, two releases,
  dominant motion) use the whole corpus with the non-learned measurement as a witness.
- **Photo ladder.** 300 unseen photos (validation and test split, no 4K frames), uniform linear camera motion of
  L = 0 … 30 px with one fixed direction per photo, through H.264 CRF 23.
- **Cell map checks.** 3,981 unseen photos without added blur, through H.264 CRF 23 and uncoded; 2,000 held-out frames
  against the measurement's 160 px motion cells.

## 10. Training environment

NVIDIA GeForce RTX 5090 (32 GB), Windows 11, Python 3.13, PyTorch 2.14 with CUDA 13.0, FFmpeg 7.1.1 (x264). Training
took 28.5 hours including the per-epoch evaluation, without interruption.

## 12. Data availability

Unchanged: the training data is not published.
