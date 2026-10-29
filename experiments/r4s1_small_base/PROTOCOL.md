# R4S-1: ConvNeXt-Small and ConvNeXt-Base trained with the same recipe

English translation of the registered protocol, [PROTOCOL_hu.md](PROTOCOL_hu.md). The Hungarian text was fixed on
2026-10-28, before any comparative result existed; its SHA-256 is in [PROTOCOL.sha256](PROTOCOL.sha256). Where the
two differ, the Hungarian text governs. File names refer to the training code as run; here they are `score.py`
(scoring) and `analyse.py` (reports).

## Protocol (fixed on 2026-10-28, before any comparative result existed)

Only the in-training validation was known beforehand (epoch 29, selected frames): AUC 0.9612 (Small) and 0.9621
(Base).

**Question** (the author, 2026-10-28): is the Small large enough to learn what the Base learned; if so, how much
faster is it; perhaps it is better. "The comparison tests will decide."

**Models:** the published files and the published loader (frame-blur-detection-update-1, `62734f4`).
- Base: `round3_base_p99_ep029.pt` (release v1.0, sha `9840160f…`).
- Small: `round3_small_p99_ep029.pt` (release v1.1, sha `b715a163…`).
- Both at the last epoch (29), fixed in advance (the author: "here too, epoch 29 shall be chosen").

**Not a pure trunk comparison:** the Small was trained with the mask fix (`mask: leak-free`). This matters only for
masked samples; the tests below use no mask, so the two models get bit-identical input.

**Scoring:** both models with the same script (`tools/r4s1_small_base.py`), on the same machine, on the GPU, the
ConvNeXt trunk under bfloat16 autocast, the fingerprint in float32, batch 1 (on the photo ladder batch 8, as in the
Base's original ladder). A frame is loaded once and both models get the same array.

### 1. Held-out films, VLM labels

- Material: all 114,687 labelled frames of the six held-out films (3 validation, 3 test). Sharp = `sss` (plus `ssb`
  in the one ssb-dominant film), blur = `bbb` + `bsb`, as in Update 1's `heldout_films.json`.
- Measure: AUC per film and per group (validation, test, all six); 95% interval from a cluster bootstrap over shots
  (2000 resamples, shots drawn with replacement within each film, seed 20261028). The same resamples for both models,
  so the Small − Base difference is paired.
- Operating points: each model at its own stored thresholds (r90 / r95 / r98, from its own validation subset): the
  share of sharp frames kept and of blurred frames caught, per group.
- Agreement of the two models: decision agreement at r95, Cohen's κ, the Spearman correlation of the logits.
- **Control:** the per-film AUCs of the rescored Base shall match the published values (from the corpus scoring). If
  they differ, this is reported, and the paired comparison uses the rescored Base.

### 2. The measurement as a witness

- The Spearman correlation of the image logit with the measured regional motion (`region_px`, v9 frame table) on the
  labelled frames that have a measurement; both models, paired bootstrap difference (the same procedure as in 1.).
- Secondary readout: M from the cell map (relu mean over the inner cells), because the Base's 0.691 reference for v9
  was made this way.

### 3. Photo ladder

- The 300 unseen etalon photos, one fixed direction per photo, L = 0 … 30 px of uniform camera motion, H.264 CRF 23
  (`tools/probe_round3.py`: the same photos, directions, lengths). Both models get the same images.
- Measure: the share called blurred per L at the own thresholds, the place of the 50% flip, the median logit, the pair
  order between 0–3 px and 3–16 px, strict increase over 0–1–2–3 px.
- **Control:** the Base shall match the published `photo_ladder_epoch29.json`.

### 4. Speed

- On the same machine (RTX 5090), with one 1080p frame, through the `classify.py` path: GPU, bfloat16 autocast;
  batch 1 and 8.
- 200 timed rounds after 30 warm-up rounds, with CUDA synchronisation; images/s, ms/image, peak memory.
- The model only (fingerprint, trunk, pooling); image loading and decoding are not included. Training time is not an
  inference measurement.

### Interpretation (in advance)

- No acceptance threshold is fixed in advance: the differences are reported with intervals, the decision is the
  author's.
- If the interval of the paired difference contains 0, no difference is shown on that measure; if not, the direction
  and the size are reported.
- The VLM label is not ground truth: it is Update 1's training signal (on other films). The photo ladder is an exact
  signal (known L), but only uniform camera motion, with a single coding.
- The frames of the validation films were seen by the per-epoch evaluation during training, but were not used for
  choosing an epoch; the test films are also reported separately.

### Not included

- Scoring the whole corpus (agreement of the two releases, scene consistency): about one day of GPU time, a separate
  decision.
- The effect of the coded frame type: a separate protocol (R4K-2, the 4b test of `round4_terv.md`).

## Log

The log follows in [REPORT.md](REPORT.md).
