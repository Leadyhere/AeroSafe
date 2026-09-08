# Accuracy changes and remaining evidence

This revision is a fresh experiment. Do not resume previous 7-class or prior-recipe states.
"Fresh" means initialize from general pretrained detector/MAE weights, not random weights.
No new Kaggle accuracy results are claimed. A GPU training run is still required.

## Implemented

- Binary defect training and explicit checkpoint/data/recipe compatibility checks.
- Dataset image SHA-256 identities in prepared annotations; portable grouping manifests.
- Fixed `dataset.split_seed: 42`, independent of optimization seed.
- Validation-default evaluation; final testing requires `--split test --final-test`.
- MMR/PatchCore no longer automatically consume the final test set after fitting.
- Validation-only aircraft model selection; test rankings cannot recommend a winner.
- Tiled validation predictions match the configured tiled deployment path.
- Small-object AP was already present; false positives per image is now reported too.
- PatchCore uses bounded per-image patch quotas so late batches do not erase early domains.
- Removed the ignored MMR inference mask-count setting; actual inference uses zero masking,
  one pass. Engine single-image preprocessing now matches the calibration center crop.
- Cosine learning rate reaches zero after all training epochs, not during the last one.
- Hard-negative mining supports all transformer candidates and writes pending proposals;
  it no longer creates a human-review approval marker automatically.
- Optional training-only slicing retains all full images and every visible annotated fragment.
- Existing quarter archives, atomic state writes, download cells and notebook deadline remain.

## Dataset deliverables

`artifacts/dataset_review_v2.zip` contains a 400-image training-only review gallery,
`review_decisions.csv`, `duplicate_conflicts.json`, and `aerosafe_group_manifest.csv`.
Open `review.html` after extracting the ZIP. Red boxes are original source annotations;
they have NOT been manually corrected or approved. Original dataset files were not edited.

The fresh audit found 195 duplicate groups with class conflicts and 326 with differing box
geometry (these groups overlap). The gallery prioritizes training conflicts. Validation/test
images are not shown for iterative label tuning. All 400 review decisions start unreviewed.

`datasets/aerosafe_group_manifest.csv` is installed locally. It freezes the existing split
by image hash and includes blank aircraft/session/camera fields. It does NOT establish unknown
acquisition identities. Ask the data provider for those IDs. If a newly supplied identity spans
multiple frozen splits, preparation fails; decide a new grouped benchmark version before training.
Do not silently move test images into training to improve a score.

On Kaggle keep the original ASDD, AircraftSurface, AGDD and IMDD datasets attached. Add the
manifest file as a private dataset named `aerosafe-group-manifest-v2`. The updated notebook
helper finds it automatically. Attach the same manifest version to every aircraft notebook
(Accounts 1–3 and Account 4 when it trains Faster R-CNN). Engine notebooks do not need it.
The review ZIP is for review; it is not a replacement training-image dataset.

## Start the first clean experiments

Clone the latest revision in a fresh notebook. Set `PREVIOUS_ARCHIVE = ""` and `PART = 1`.
Use the current KAGGLE_FRIENDS cells. Do NOT pin the earlier aabe32aa revision.
Run configuration and `prepare(MODEL, SESSION_START)` again. For local preparation:

```bash
python scripts/prepare_data.py --scope aircraft --config config.yaml
```

Train RT-DETRv2 and Deformable DETR on identical data; Faster R-CNN is the comparison baseline.
The RT-DETR comparison remains available. The five-account assignment is unchanged.

After training candidates, restore their final archives into an evaluation workspace and run:

```bash
python scripts/evaluate.py --target aircraft-transformers --split validation --config config.yaml
python scripts/evaluate.py --target faster-rcnn --split validation --config config.yaml
python scripts/select_aircraft_model.py --models rt_detr_v2 rt_detr deformable_detr faster_rcnn
```

If training only a subset of transformers, keep only those candidates in the evaluation
config and list only evaluated models for selection. This creates `reports/model_selection.json`
using validation mAP50:95, then recall. Compare AP-small, misses, and FP/image alongside the ranking.
Anomaly validation contains normals only: it cannot select an engine winner using AUROC/AP.
Use a separate labeled engine development set or predeclare the engine configuration; do not
reuse the official test anomalies for tuning.

## Threshold requirements

Choose recall and false-alarm targets appropriate to the demonstration. This example uses
90% validation recall and at most 0.1 false positives per image; it is NOT an aviation standard:

```bash
python scripts/calibrate_aircraft_threshold.py --predictions reports/validation/aircraft_transformers/rt_detr_v2/aircraft_predictions/predictions.json --min-recall 0.90 --max-fp-per-image 0.10 --output reports/validation/operating_point.json
```

If the targets cannot both be met, `feasible` is false and no threshold is recommended.
FP/image differs from false-positive rate on confirmed-normal images. Without confirmed-normal
validation examples, report that limitation. MMR/PatchCore quantile thresholds use only held-out
training normals; those normals alone cannot measure anomaly recall.

## Winner-only slicing and seed experiments

After selecting a candidate on validation, create a fresh ablation notebook (new state):

```bash
python scripts/add_training_slices.py --input data/processed/aircraft/train.json --output data/processed/slices512 --size 512 --max-per-image 4
```

Set `aircraft.training_annotations_override: data/processed/slices512/train.json` in config
after preparation, then start a fresh run for the winning model. Full-image records are retained.
Use 640 instead of 512 for a separate ablation. Crops increase work per epoch; retain the notebook
deadline and expect additional partial sessions. Do not create slices from validation/test.

For three repeats, use separate fresh notebooks with `training.seed` 42, 54 and 77;
leave `dataset.split_seed` and the manifest unchanged. Keep each run's outputs in a separate
notebook/dataset. Report mean and spread of validation metrics before freezing a final recipe.

## Engine experiments and official reference

The official AeBAD-S MMR configuration uses 224-pixel input, batch 16, learning rate .001,
200 epochs, weight decay .05, warmup 50, mask ratio .4 in training and 0 in testing:
https://github.com/zhangzilongc/MMR/blob/master/method_config/AeBAD_S/MMR.yaml

The compact local implementation is an adaptation, not an exact reproduction. Its teacher
weights/FPN details, held-out normal calibration split and hardware differ. The official repo
is the reference for a separate controlled comparison. It has NOT been GPU-trained here.

For resolution ablations, use fresh config copies with `engine.image_size` 224, 320 or 384,
preserve the train/validation identities, and train/calibrate each separately. Never reuse
224-pixel thresholds at another resolution. Higher resolutions cost more memory and time;
do not infer improved accuracy from passing an architecture shape test.

For sliced-run resumes, regenerate the same slices from the frozen train split and reapply
the same annotation override before restoring training progress. The ordinary archives do
not bundle slice image files. The data identity check rejects a full-image-only resume of a
sliced experiment.

## Pending human/source-data work

1. Review the 400-image gallery and correct source bounding boxes where visually justified.
   Keep an edit log and create a new dataset version; do not erase the originals.
2. Supply real aircraft/session/camera identities for grouping.
3. Supply 500–1,000 human-confirmed normal aircraft photos with acquisition metadata.
   Unlabeled existing images and synthetic pictures are not substitutes for confirmed normals.
4. Mine proposals with `scripts/mine_hard_negatives.py --normal-dir YOUR_NORMALS --confirmed-normal
   --model rt_detr_v2 --round 1`. Review the pending folder. Import only individually approved
   images into the reviewed hard-negative dataset and document the review.
5. Obtain unseen external aircraft images, preferably with independent annotations.
6. Run training and compare official/compact MMR on a predeclared protocol.

After architecture, thresholds, resolution and seed protocol are frozen, unlock final testing:

```bash
python scripts/evaluate.py --target aircraft --split test --final-test --config config.yaml
python scripts/evaluate.py --target engine --split test --final-test --config config.yaml
```

Set `aircraft.checkpoint` to the validation-selected candidate first. Use `--target external-aircraft`
with `--split test --final-test` for independently annotated external data. Human inspection remains
part of the workflow; portfolio results do not certify aviation safety.

Research basis: grouped evaluation avoids identity leakage
(https://scikit-learn.org/stable/modules/cross_validation.html); SAHI supports investigating mixed
full-image and sliced training/inference for small objects (https://arxiv.org/abs/2202.06934).
These justify experiments, not a guaranteed accuracy improvement.
