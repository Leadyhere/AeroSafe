# Kaggle training runbook

This runbook trains or fits all four declared models on controlled, identical splits:

1. Deformable DETR and Faster R-CNN use the same cleaned ASDD + aircraftsurface1 split.
2. MMR and PatchCore use the same normal AeBAD-S training and threshold-calibration split.
3. All four are evaluated on held-out data. IISc remains external-test-only; BladeSynth is excluded
   from this initial real-data comparison.

## 1. Create the notebook

Create a Kaggle notebook, select a GPU accelerator, enable Internet for pretrained-weight downloads,
and attach ASDD, aircraftsurface1, and AeBAD-S as inputs. Dataset licenses and access remain the user's
responsibility.

Clone the repository:

```python
!git clone https://github.com/Leadyhere/AeroSafe.git /kaggle/working/aeroinspect
%cd /kaggle/working/aeroinspect
!pip install -q -r requirements.txt
```

Inspect the exact input directory names:

```python
!find /kaggle/input -maxdepth 4 -type d | head -100
```

## 2. Set paths

Edit only the `paths` section of `config.yaml`. Kaggle input mounts are read-only; processed data,
checkpoints, and reports must use `/kaggle/working`.

```yaml
paths:
  asdd: /kaggle/input/YOUR-ASDD-SLUG
  aircraftsurface: /kaggle/input/YOUR-AIRCRAFTSURFACE1-SLUG
  aebad: /kaggle/input/YOUR-AEBAD-SLUG/AeBAD_S
  bladesynth: /kaggle/input/not-used
  external_iisc: /kaggle/input/not-used
  processed: /kaggle/working/aeroinspect/data/processed
  checkpoints: /kaggle/working/aeroinspect/checkpoints
  reports: /kaggle/working/aeroinspect/reports
  database: /kaggle/working/aeroinspect/data/aeroinspect.sqlite3
  inspections: /kaggle/working/aeroinspect/data/inspections
```

## 3. Prepare, smoke-test, train, and evaluate

Run stages separately so failures are easy to diagnose:

```python
!python scripts/kaggle_train_all.py --stage prepare --config config.yaml
!python scripts/kaggle_train_all.py --stage smoke --config config.yaml
!python scripts/kaggle_train_all.py --stage full --config config.yaml
!python scripts/kaggle_train_all.py --stage evaluate --config config.yaml
```

Smoke outputs are isolated under `checkpoints/smoke` and `reports/smoke`; they never replace full
checkpoints or final metrics. If a session stops during gradient-based training, resume the affected
model with its own `--resume` option. PatchCore fitting is deterministic and should simply be rerun.

If CUDA runs out of memory, reduce the relevant `batch_size` in `config.yaml`. Do not change dataset
splits independently between a main model and its baseline.

## 4. Preserve outputs

Before ending the Kaggle session, use **Save Version** with outputs enabled or download:

```text
checkpoints/
reports/
```

`reports/model_comparison.json` contains the final four-model comparison. Real metrics appear only after
the final evaluation; the repository intentionally ships with no fabricated scores or model weights.
