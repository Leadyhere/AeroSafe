# Kaggle training runbook

This runbook prepares the downloaded datasets, smoke-tests all four models, trains them, and performs
final evaluation. Keep every dataset separate; the preparation code combines only compatible labels and
preserves source metadata.

## 1. Finish and extract BladeSynth

`Unconfirmed 93855.crdownload` is an incomplete Chrome download. It cannot be uploaded to Kaggle or used
for training. Wait until Chrome finishes and renames it to `Dataset_bladesynth.rar`, then extract the RAR.
The configured extracted directory is `Dataset_bladesynth/`.

Do not rename the `.crdownload` file manually. If Chrome reports that the download failed, resume it in
Chrome or restart the download.

## 2. Create Kaggle datasets

Upload the following directories/files as private Kaggle datasets. Large datasets may be split into more
than one Kaggle input; the code only needs the final paths.

| Local item | Purpose |
|---|---|
| `Dataset2.v2-defect-classification.coco/` | ASDD detector fine-tuning |
| `aircraftsurface1.v1i.coco/` | Aircraft detector fine-tuning |
| `AGDD-main/` | AGDD detector auxiliary pretraining |
| `0to4_aircraft_skin4000pics/` and `0-4aircraft4000.csv` | IMDD integrity audit only; no boxes exist |
| `AeBAD/AeBAD/` | AeBAD-S training/test plus AeBAD-V normal auxiliary frames |
| extracted `Dataset_bladesynth/` | BladeSynth Normal-only MMR auxiliary pretraining |

For each Kaggle dataset: open **Datasets → New Dataset**, upload the corresponding folder (or a ZIP), set
visibility to **Private**, and create it. Kaggle automatically extracts ZIP files, but RAR extraction is
not dependable; upload the already-extracted BladeSynth directory.

## 3. Create the notebook

Create a Kaggle notebook, choose a GPU accelerator, enable Internet for pretrained-weight downloads, and
attach every private Kaggle dataset from the notebook's **Add Input** panel.

Clone and install AeroInspect:

```python
!git clone https://github.com/Leadyhere/AeroSafe.git /kaggle/working/aeroinspect
%cd /kaggle/working/aeroinspect
!pip install -q -r requirements.txt
```

Inspect the exact mounted names before editing the configuration:

```python
!find /kaggle/input -maxdepth 5 -type d | head -200
```

## 4. Set the exact paths

Edit only the `paths` block in `config.yaml`. Replace every `YOUR-...-SLUG` with the name shown under
`/kaggle/input`. The directory may have one additional wrapper level depending on how it was uploaded.

```yaml
paths:
  asdd: /kaggle/input/YOUR-ASDD-SLUG/Dataset2.v2-defect-classification.coco
  aircraftsurface: /kaggle/input/YOUR-AIRCRAFTSURFACE-SLUG/aircraftsurface1.v1i.coco
  agdd: /kaggle/input/YOUR-AGDD-SLUG/AGDD-main
  imdd_aircraft_images: /kaggle/input/YOUR-IMDD-SLUG/0to4_aircraft_skin4000pics
  imdd_aircraft_csv: /kaggle/input/YOUR-IMDD-SLUG/0-4aircraft4000.csv
  imdd_localization: /kaggle/input/YOUR-REVIEWED-IMDD-SLUG/imdd_localization_reviewed
  hard_negatives: /kaggle/working/aeroinspect/data/hard_negatives/reviewed
  group_manifest: /kaggle/input/YOUR-METADATA-SLUG/aerosafe_group_manifest.csv
  aebad: /kaggle/input/YOUR-AEBAD-SLUG/AeBAD/AeBAD
  bladesynth: /kaggle/input/YOUR-BLADESYNTH-SLUG/Dataset_bladesynth
  external_iisc: /kaggle/input/not-used
  processed: /kaggle/working/aeroinspect/data/processed
  checkpoints: /kaggle/working/aeroinspect/checkpoints
  reports: /kaggle/working/aeroinspect/reports
  database: /kaggle/working/aeroinspect/data/aeroinspect.sqlite3
  inspections: /kaggle/working/aeroinspect/data/inspections
```

The AeBAD adapter accepts either the parent that contains `AeBAD_S/` and `AeBAD_V/` or the `AeBAD_S/`
directory itself, but the parent is required here because MMR also uses normal AeBAD-V training frames.

## 5. Prepare and verify

Run preparation first:

```python
!python scripts/kaggle_train_all.py --stage prepare --config config.yaml
```

This command must finish successfully and create `reports/dataset_report.json`. It checks annotation
formats, image decoding, missing masks, class mappings, exact/near duplicates, and minimum BladeSynth
Normal-image count. Do not continue if it raises an error; correct the named path or incomplete dataset.

Scientific data use is intentionally staged:

- Deformable DETR and Faster R-CNN pretrain on mapped AGDD boxes, then fine-tune on the cleaned combined
  ASDD + aircraftsurface1 detector split.
- IMDD's 4,281 images have image-level CSV labels but no bounding boxes, so they are audited and excluded
  from object-detector training. Creating full-image fake boxes would damage localization training.
- The primary MMR is trained only on real AeBAD-S normals. A second, separately checkpointed experiment
  pretrains on sampled real AeBAD-V normals plus BladeSynth's `Normal` class, then fine-tunes on AeBAD-S.
- Both MMR variants calibrate thresholds only with held-out real AeBAD-S normals. BladeSynth anomaly
  images are never treated as normal, and experimental metrics are written under
  `reports/experiments/bladesynth_mmr/` rather than mixed into the four-model comparison.
- PatchCore remains a clean real-data baseline fitted only on the same AeBAD-S normal split.

After the first full Deformable DETR run, create IMDD proposals:

```python
!python scripts/bootstrap_imdd_boxes.py --config config.yaml
```

Download `data/annotation_tasks/imdd_proposals.json`, import it with the IMDD images into CVAT/Roboflow,
correct every selected image, and export COCO. Put that reviewed export in a separate Kaggle Dataset with
an empty file named `REVIEWED`, update `imdd_localization`, rerun preparation, and retrain both detectors.
Unreviewed proposals and whole-image boxes are refused.

If your datasets provide aircraft/tail/session/video/camera identifiers, attach a completed copy of
`docs/group_manifest.example.csv`. If they do not, leave `group_manifest` pointing to a nonexistent path;
the preparer still groups exact, near, and generated derivatives but reports zero explicit metadata.

For post-deployment hard-negative rounds, upload only inspector-confirmed normal images and run round 1:

```python
!python scripts/mine_hard_negatives.py --normal-dir /kaggle/input/CONFIRMED-NORMALS --round 1 --confirmed-normal
!python scripts/prepare_data.py --config config.yaml
!python scripts/train_aircraft.py --mode full --config config.yaml
!python scripts/train_faster_rcnn.py --mode full --config config.yaml
```

Repeat for rounds 2 and 3 after reviewing new pilot images. Unlabeled images are not confirmed normals.

## 6. Monitor training with TensorBoard

Every training entry point writes TensorBoard events by default. The logs are kept below the configured
reports directory, so a Kaggle run using `config.kaggle.yaml` writes to:

```text
/kaggle/working/aeroinspect/reports/tensorboard/
  smoke/
    deformable_detr/
    faster_rcnn/
    mmr_real/
    patchcore/
    mmr_bladesynth/
  full/
    deformable_detr/
    faster_rcnn/
    mmr_real/
    patchcore/
    mmr_bladesynth/
```

After at least one smoke or full training command has started, open the dashboard in a new notebook cell:

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/aeroinspect/reports/tensorboard
```

Use `progress/percent` to see completion, `train/batch_loss` and `train/epoch_loss` for optimization,
and the `validation/*` charts for model quality. Deformable DETR and Faster R-CNN log mAP, precision,
recall, and detector loss components. Both MMR variants log reconstruction loss and held-out-normal
calibration. PatchCore has no gradient epochs, so it logs feature-extraction percentage, retained
patches, memory-bank size, calibration, and final evaluation metrics.

TensorBoard progress is not an accuracy score. Final comparisons still come from
`reports/model_comparison.json` after the frozen test evaluation.

## 7. Smoke-test, train, and evaluate

Run each stage in a separate notebook cell:

```python
!python scripts/kaggle_train_all.py --stage smoke --config config.yaml
!python scripts/kaggle_train_all.py --stage full --config config.yaml
!python scripts/kaggle_train_all.py --stage evaluate --config config.yaml
```

The smoke and full stages run the four primary models and then the separately named BladeSynth MMR
experiment. This means MMR is trained twice by design and requires additional Kaggle GPU time; the two
checkpoints and metric directories never overwrite each other.

Smoke artifacts are isolated under `checkpoints/smoke/` and `reports/smoke/`; they never replace full
weights or metrics. If CUDA runs out of memory, reduce the relevant `batch_size` in `config.yaml`.

If a Kaggle session stops, resume the affected gradient-trained model with its saved state:

```python
!python scripts/train_aircraft.py --mode full --config config.yaml --resume checkpoints/aircraft_training_state.pt
!python scripts/train_faster_rcnn.py --mode full --config config.yaml --resume checkpoints/faster_rcnn_training_state.pt
!python scripts/train_engine.py --mode full --variant real --config config.yaml --resume checkpoints/engine_training_state.pt
!python scripts/train_engine.py --mode full --variant bladesynth --config config.yaml --resume checkpoints/engine_bladesynth_training_state.pt
```

PatchCore fitting is deterministic and should be rerun instead of resumed.

### Reliable short Kaggle runs

Kaggle draft sessions can end before a long detector run finishes. Do not repeat completed work. The
aircraft trainer can intentionally finish a short, resumable chunk after an **absolute** epoch number:

```python
# First saved chunk: epochs 1 through 8 (including the three AGDD epochs).
!python scripts/train_aircraft.py --mode full --config config.kaggle.yaml --stop-after-epoch 8

# After attaching the saved output checkpoint in a later Kaggle session, continue through epoch 16.
!python scripts/train_aircraft.py --mode full --config config.kaggle.yaml \
  --resume /kaggle/input/YOUR-SAVED-CHECKPOINTS/aircraft_training_state.pt \
  --stop-after-epoch 16
```

After every completed chunk, choose **Save Version** with outputs enabled. Attach that version's output
or upload the checkpoint as a private Kaggle dataset before the next chunk. Never close a draft session
until its `checkpoints/aircraft_training_state.pt` has been preserved outside `/kaggle/working`.

The two other gradient-trained model families use the same absolute-epoch flag:

```python
!python scripts/train_faster_rcnn.py --mode full --config config.kaggle.yaml --stop-after-epoch 6
!python scripts/train_engine.py --mode full --variant real --config config.kaggle.yaml --stop-after-epoch 6
!python scripts/train_engine.py --mode full --variant bladesynth --config config.kaggle.yaml --stop-after-epoch 6
```

Resume each model only from its matching state file and variant. PatchCore is a single deterministic fit,
not an epoch-trained model, so it should be run once and does not accept either chunk or resume flags.

## 8. Preserve outputs

Before the Kaggle session ends, choose **Save Version** with outputs enabled or download:

```text
checkpoints/
reports/
```

`reports/model_comparison.json` contains the final four-model comparison. The repository ships with no
fabricated metrics or trained weights.
