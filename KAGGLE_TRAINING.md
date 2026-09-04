# Kaggle training runbook

This is the complete AeroInspect training plan. It contains seven fair training runs: five
transformer-based runs and two comparison baselines. Do not run all full models in one notebook session.
Run one command at a time, save the Kaggle version, and download the generated archive.

## Models, datasets, and estimated Kaggle T4 time

Times are planning ranges, not guarantees. Kaggle GPU type, image loading speed, and validation time vary.
The code measures real epoch time and exits early when continuing could exceed the configured 11-hour
session budget, leaving 45 minutes to package the result.

| Run | Type | Purpose | Training data | Epoch plan | Approx. T4 time |
|---|---|---|---|---:|---:|
| RT-DETR v2 | Transformer | Primary aircraft localizer | AGDD, then binary ASDD + aircraftsurface1 | 3 + 20, one logical run | 8-16 h |
| RT-DETR | Transformer | Real-time transformer comparison | Same aircraft split | 3 + 20, one logical run | 8-16 h |
| Deformable DETR | Transformer | Multi-scale transformer comparison | Same aircraft split | 3 + 20, one logical run | 10-20 h |
| MMR real | Transformer hybrid | Primary engine anomaly score and heatmap | AeBAD-S normal images | 200; 50/50/50/50 | 6-10 h total |
| MMR + BladeSynth | Transformer-hybrid experiment | Test auxiliary normal-domain pretraining | AeBAD-V normals + BladeSynth Normal, then AeBAD-S | 5 + 200; endpoints 55/105/155/205 | 8-14 h total |
| Faster R-CNN | CNN baseline | Prove whether transformers improve aircraft results | Same aircraft split | 3 + 20, one logical run | 8-16 h |
| PatchCore | CNN baseline | Engine anomaly comparison | AeBAD-S normal images | One fit | 0.5-1.5 h |

The combined estimate is roughly 49-94 T4 GPU-hours. Each individual notebook session remains under 12
hours. A 20-epoch model is attempted in one logical run; if real measured speed makes that unsafe, it stops,
archives the completed epochs, and resumes when the same command is run again.

## Dataset rules

- Aircraft training uses one reliable `defect` class. Original source labels remain in the audit report.
- AGDD is auxiliary box pretraining. ASDD and aircraftsurface1 form the grouped main split.
- IMDD has image labels but no boxes, so it is excluded from localization training.
- MMR and PatchCore fit only normal images. The official AeBAD-S test set never calibrates thresholds.
- BladeSynth anomalies are never treated as normal.
- Exact and near duplicates stay within a single train/validation/test group.

## 1. Upload datasets

Create private Kaggle datasets from these extracted local folders/files:

| Local item | Used by |
|---|---|
| `Dataset2.v2-defect-classification.coco/` | Aircraft models |
| `aircraftsurface1.v1i.coco/` | Aircraft models |
| `AGDD-main/` | Aircraft auxiliary pretraining |
| `0to4_aircraft_skin4000pics/` and `0-4aircraft4000.csv` | Audit only |
| `AeBAD/AeBAD/` | MMR and PatchCore |
| `Dataset_bladesynth/` | Separate MMR + BladeSynth experiment |

Do not upload an unfinished `.crdownload`. Extract RAR archives locally before uploading them.

## 2. Create the GPU notebook

Choose a T4/P100 GPU, enable Internet for the first pretrained-weight download, and attach all private
datasets. Then run:

```python
!git clone https://github.com/Leadyhere/AeroSafe.git /kaggle/working/aeroinspect
%cd /kaggle/working/aeroinspect
!pip install -q -r requirements.txt
!find /kaggle/input -maxdepth 5 -type d | head -200
```

Update the `paths` values in `config.yaml` to the exact mounted paths. Outputs must stay under
`/kaggle/working/aeroinspect`:

```yaml
paths:
  asdd: /kaggle/input/YOUR-ASDD/Dataset2.v2-defect-classification.coco
  aircraftsurface: /kaggle/input/YOUR-AIRCRAFT/aircraftsurface1.v1i.coco
  agdd: /kaggle/input/YOUR-AGDD/AGDD-main
  imdd_aircraft_images: /kaggle/input/YOUR-IMDD/0to4_aircraft_skin4000pics
  imdd_aircraft_csv: /kaggle/input/YOUR-IMDD/0-4aircraft4000.csv
  aebad: /kaggle/input/YOUR-AEBAD/AeBAD/AeBAD
  bladesynth: /kaggle/input/YOUR-BLADESYNTH/Dataset_bladesynth
  processed: /kaggle/working/aeroinspect/data/processed
  checkpoints: /kaggle/working/aeroinspect/checkpoints
  reports: /kaggle/working/aeroinspect/reports
  artifacts: /kaggle/working/aeroinspect/artifacts
```

## 3. Prepare and smoke-test once

```python
!python scripts/kaggle_train_all.py --stage prepare --config config.yaml
!python scripts/kaggle_train_all.py --stage smoke --config config.yaml
!python scripts/kaggle_train_all.py --stage plan --config config.yaml
```

Preparation must report `task_mode: binary`, one `defect` category, and no split leakage. Smoke mode uses
only a tiny real subset; it checks code and GPU compatibility but does not produce meaningful accuracy.

## 4. Train aircraft models

Run each command in its own Kaggle session/version:

```python
!python scripts/kaggle_train_all.py --stage train --model rt_detr_v2 --quarter 1 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model rt_detr --quarter 1 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model deformable_detr --quarter 1 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model faster_rcnn --quarter 1 --config config.yaml
```

These have 20 main epochs and therefore one logical part. If the time guard exits before epoch 23, save the
Kaggle version and rerun the identical command after restoring the archive. It resumes automatically.

## 5. Train engine models

MMR is explicitly divided into four quarters:

```python
!python scripts/kaggle_train_all.py --stage train --model mmr_real --quarter 1 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model mmr_real --quarter 2 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model mmr_real --quarter 3 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model mmr_real --quarter 4 --config config.yaml

!python scripts/kaggle_train_all.py --stage train --model mmr_bladesynth --quarter 1 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model mmr_bladesynth --quarter 2 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model mmr_bladesynth --quarter 3 --config config.yaml
!python scripts/kaggle_train_all.py --stage train --model mmr_bladesynth --quarter 4 --config config.yaml

!python scripts/kaggle_train_all.py --stage train --model patchcore --quarter 1 --config config.yaml
```

Part 2 refuses to start unless part 1 is complete; the same rule applies to later parts. This prevents
silently skipping epochs.

## 6. Preserve and resume every output

Every successful full-training command creates both files under `artifacts/`:

```text
MODEL_part_N_epoch_X_of_Y.tar.gz
MODEL_part_N_epoch_X_of_Y.tar.gz.sha256
```

A time-limited partial run includes `partial` in the name. The archive contains model/optimizer/scheduler
state, the best checkpoint available so far, configuration, reports, and a per-file SHA-256 manifest.

Before ending the notebook, choose **Save Version** with outputs enabled and download both artifact files.
For the next session, attach the previous output as a Kaggle dataset and restore it at the repository root:

```python
!tar -xzf /kaggle/input/YOUR-PREVIOUS-OUTPUT/MODEL_ARCHIVE.tar.gz -C /kaggle/working/aeroinspect
```

Then run the same part again if it was partial, or the next quarter if the earlier quarter completed. The
trainer finds its standard state file automatically. Never start part 2/3/4 from only deployment weights;
the resumable state is required.

## 7. Evaluate after every model is complete

In the evaluation notebook, attach and extract the final archive from all seven runs so every checkpoint
is present together under `checkpoints/`. Then run:

```python
!python scripts/kaggle_train_all.py --stage evaluate --config config.yaml
```

Compare aircraft models using frozen-test mAP@50:95, mAP@50, recall, and missed-defect rate. Compare engine
models using image AUROC, image average precision, pixel AUROC, AUPRO, false-alarm rate, and missed-anomaly
rate. Training loss alone does not select the winner.

`reports/model_selection.json` ranks aircraft and engine models separately and records the recommended
winner and selection rule. It never compares aircraft mAP directly with engine AUROC.

Only after this comparison should `aircraft.checkpoint` point to the winning candidate. Keep the baseline
results in the project because they demonstrate that the transformer choice was measured rather than
assumed.
