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

To promote IMDD into detector training later, annotate true defect boxes for a reviewed subset in
Roboflow or CVAT, export it as COCO, and add that export as a new boxed source. Keep the original CSV-only
copy unchanged for image-level external checks. Do not auto-convert each full image into one box.

## 6. Smoke-test, train, and evaluate

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

## 7. Preserve outputs

Before the Kaggle session ends, choose **Save Version** with outputs enabled or download:

```text
checkpoints/
reports/
```

`reports/model_comparison.json` contains the final four-model comparison. The repository ships with no
fabricated metrics or trained weights.
