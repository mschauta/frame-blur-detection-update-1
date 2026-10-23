# Frame Blur Detection — Update 1

The second published model of [Frame Blur Detection](https://github.com/mschauta/frame-blur-detection): a blur-specific
sharp / blurred classifier for video frames, on the same input as before — the signed edge fingerprint of the
[RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling) `m09` reconstruction — and with the same
synthetic photo anchors. What changed is the scale and the model: a **much larger frame set (173 videos instead of 37
files)**, a **ConvNeXt-Base** trunk instead of ConvNeXt-Small, and an **exact float32 fingerprint** instead of the
bf16-rounded one.

The task, the principles, the labelling, the photo construction and the model design are described in the original
repository's [README](https://github.com/mschauta/frame-blur-detection#readme) and
[METHOD.md](https://github.com/mschauta/frame-blur-detection/blob/main/METHOD.md). This repository documents only what
is new: [METHOD.md](METHOD.md) lists the changes section by section, [reproduction/](reproduction/) holds the aggregate
evidence behind every number on this page.

## What changed

| | Frame Blur Detection (100-epoch runs) | Update 1 |
|---|---|---|
| frame material | 37 processed video files (36 groups) | 173 videos, 5,010,602 frames, ~1.0 million labelled by the same frozen VLM teacher |
| frame selection | VLM labels, measurement ranking inside shots | v8 rules: films as units (both releases of a film twice released at different bitrates), a film-relative sharp class, the measurement only ranks inside a shot, no thinning (METHOD §5) |
| training frames | — | 589,663 (349,624 sharp, 240,039 blurred) |
| validation / test | 11 / 7 video groups | 3 / 3 films chosen by hand, none of their frames trained on |
| photo anchors | 33,484 training photos, 0–3 px sharp / 16–30 px blurred, through H.264 | **unchanged** |
| fingerprint | computed inside bf16 autocast (bf16-rounded) | computed in float32, outside autocast |
| model | ConvNeXt-Small (~34 M parameters) to stride 16 | ConvNeXt-Base (60.2 M parameters) to stride 16; the same head and pooling |
| training | 100 epochs | 30 epochs × 48,000 samples, 90,440 optimizer steps; the last epoch is the result |

## Weights

The weight file is attached to the [release](../../releases) (230 MB; not stored in the repository):

| file | sha256 | input | epoch | thresholds r90 / r95 / r98 | validation frame AUC |
|---|---|---|---|---|---|
| `round3_base_p99_ep029.pt` | `9840160f0c337fc1312c2566abf4231be0bd7ee709f65f489fd8f81c65b8d0e6` | edge fingerprint | 29 (last of 30) | 0.2324 / 0.0549 / 0.0164 | 0.962 |

A frame is called blurred when its score (the sigmoid output, not a calibrated probability) reaches the threshold; the
three thresholds catch 90 / 95 / 98% of the blurred validation frames. The file holds the state dict and its metadata;
the metadata keys `arch` and `fingerprint` tell the included loader to build ConvNeXt-Base with the float32
fingerprint. The weights of the original repository load unchanged with this code.

```
pip install -r requirements.txt
# put round3_base_p99_ep029.pt into weights/
python classify.py --weights weights/round3_base_p99_ep029.pt --input frames --output sorted --recall 95
```

The model expects what it was trained on: **video frames decoded from H.264** (1080p in training), stored losslessly.
Uncoded images are outside its training distribution (see the cell map below).

## Results

### Validation (selected frames, 3 films)

16,190 sharp and 16,190 blurred frames selected by the v8 rules, and 4,000 photo variants
([validation_epoch29.json](reproduction/validation_epoch29.json)):

| | AUC | sharp frames kept at 90 / 95 / 98% blurred caught |
|---|---|---|
| frames | 0.962 | 88.5 / 79.5 / 66.5% |
| photos (0–3 px vs 16–30 px) | 1.000 | — |

### Held-out films, every labelled frame

All VLM-labelled frames of the six held-out films, scored once after the checkpoint had been fixed. Sharp = `sss` (plus
`ssb` in films where `ssb` dominates), blurred = `bbb` + `bsb`; the thresholds come from the validation selection, so
the caught share on all labelled frames is close to, but not exactly, 90 / 95 / 98%
([heldout_films.json](reproduction/heldout_films.json)):

| films | sharp | blurred | AUC | kept / caught at r95 | kept / caught at r98 |
|---|---|---|---|---|---|
| validation (3) | 37,133 | 17,886 | 0.958 | 79.9% / 94.4% | 66.1% / 97.7% |
| test (3) | 33,877 | 25,791 | 0.966 | 81.0% / 95.3% | 69.8% / 97.0% |
| all six | 71,010 | 43,677 | 0.963 | 80.4% / 95.0% | 67.9% / 97.3% |

Per film the AUC is 0.954–0.980. These numbers are **not directly comparable** with the original 100-epoch runs: the
frame material, the selection and the split are different.

### Consistency over the whole corpus

All 5,010,602 frames of the 173 videos were scored ([corpus_checks.json](reproduction/corpus_checks.json)).

- **Scene consistency.** Within shots that hold both decisions (95% point), the measured motion separates the model's
  blurred frames from its sharp ones with a median AUC of 0.854; the VLM labels reach 0.863. The measurement is a
  separate, non-learned motion and sharpness measurement that was never a model input.
- **Two releases of the same film.** For 31 films released twice at different bitrates, frames matched by time agree
  at the 95% point with a median of 85.0% (Cohen's κ 0.68; Pearson correlation of the logits 0.80).
- **By dominant motion** (labelled blurred frames caught at r95): camera / global motion 98.5%, subject / local motion
  93.1%, mixed 99.2%. Small local blur is the weakest case.

### Photo ladder

300 unseen photos with uniform camera motion of known length, one direction per photo, through H.264 CRF 23
([photo_ladder_epoch29.json](reproduction/photo_ladder_epoch29.json)):

![Photo ladder: share called blurred and median logit against the added camera motion](figures/photo_ladder.svg)

The decision flips at about **9 px** at all three thresholds (98%: 8 px 8%, 10 px 86%, 12 px 100%): the logit jumps
from −14 to +8 between 8 and 10 px. From 0 to 6 px the logit stays saturated around −22, so **the score does not rank
inside the sharp range** (pairwise order accuracy 0.62 for 0–3 px, 0.99 for 3–16 px). This follows from the training
construction: 0–3 px are taught as sharp, 16–30 px as blurred, and the 4–15 px gap is never trained. A later model will
use a different photo ladder; this one belongs to Update 1.

### Cell map

The model gives one logit per 16 px cell (`out["local"]`); the image score pools them. On 3,981 unseen photos with no
added blur, passed through H.264 CRF 23, no cell exceeds logit 0.06 (one photo reaches 0). The same photos **without**
coding produce a cell at or above 0 in 18.7% of the cases, and some are called blurred: the model relies on the coding
chain it was trained on. Against the motion measurement, the cell map agrees with where the subject moves (median
within-frame Spearman 0.34 over 160 px cells, 2,000 held-out frames). The image score follows region and subject motion
(Spearman 0.71 / 0.73) much more than camera motion (0.20).

## Limitations

- **Labels are VLM labels.** Agreement with them is not ground truth; the consistency checks above are witnesses, not
  proofs.
- **Small local blur** is caught less often than global blur, and the score does not rank within 0–3 px.
- **Captions, overlaid graphics and animation are out of scope.** The watermark training teaches the model to ignore
  overlaid layers (masked on half of the frames, visible on the other half); on captions it is unreliable.
- **Filmic material.** On a professionally produced drama episode the cell map is mostly placed well, but deliberate
  shallow depth of field (an out-of-focus partner in a dialogue) is often called blurred, and motion seen through a car
  window is real blur that a film may accept. The training material comes from handheld-camera, people-centred videos.
- **Uncoded images** are outside the training distribution.

## Data availability

As in the original repository: the training data is **not published**, it was built from copyrighted videos and
photographs, and no images, frames, crops or derived images from it are published. Published are the method, the
selection rules, the reported aggregate metrics, the inference code and the trained weights.

## Related work by the author

- [Frame Blur Detection](https://github.com/mschauta/frame-blur-detection): the method, the RGB / grayscale / edge
  fingerprint comparison, the first published weights and the evidence package.
- [RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling): the continuous RGB-mesh reconstruction from
  which the edge fingerprint is derived.

## License

[MIT](LICENSE) © Dr. Schauta Marcell
