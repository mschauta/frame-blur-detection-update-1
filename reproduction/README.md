# Evidence

Aggregate numbers behind the README; no images, frames or per-frame records of the non-public training data.

| file | what |
|---|---|
| `validation_epoch29.json` | the validation report of the published checkpoint: frames, photos, per VLM code, thresholds and the shares kept / caught at them |
| `heldout_films.json` | every labelled frame of the six held-out films, scored once after the checkpoint was fixed: per group and per film (film names withheld) |
| `corpus_checks.json` | whole-corpus checks: scene consistency, the two releases of the same films, catch rates by dominant motion, agreement with the non-learned motion measurement, cell maps on unblurred photos |
| `photo_ladder_epoch29.json` | the photo ladder: share called blurred and median logit per added blur length |
| `render_photo_ladder.py` | redraws `figures/photo_ladder.svg` from `photo_ladder_epoch29.json` (standard library only) |
