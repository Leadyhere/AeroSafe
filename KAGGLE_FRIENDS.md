# Copy-and-paste Kaggle instructions

For the Deformable DETR NaN failure, use [the recovery cells](KAGGLE_DEFORMABLE_RECOVERY.md).
They retain the working dataset paths and use explicit 12/23 cycle targets.

Accuracy-integrity update: read [ACCURACY_PLAN.md](ACCURACY_PLAN.md) before starting new runs.
Use fresh states from this revision; older checkpoint recipes are rejected. Aircraft notebooks
should attach `aerosafe-group-manifest-v2` alongside the original datasets. Final-test evaluation
is now explicitly locked during model development. The older aabe32aa revision lacks these changes.

Each friend uses their own notebook and one model. Use the same dataset versions across
friends. Every code block below is a separate notebook cell. Run cells in order and stop
on errors. Enable a GPU and Internet before starting. Start a fresh session for every part;
do not run all four MMR parts with Run All. The new code has local automated tests, but
the runtime estimates and accuracy have not been verified in a full Kaggle training run.

| Friend | Notebook name | Model | Attach datasets | Main epochs | Estimated total T4 time |
|---|---|---|---|---|---|
| 1 | aerosafe-rt-detr-v2 | rt_detr_v2 | ASDD, AircraftSurface, AGDD, IMDD | 20 + 3 auxiliary | 8–16 h |
| 2 | aerosafe-rt-detr | rt_detr | Same four | 20 + 3 auxiliary | 8–16 h |
| 3 | aerosafe-deformable-detr | deformable_detr | Same four | 20 + 3 auxiliary | 10–20 h |
| 4 | aerosafe-faster-rcnn | faster_rcnn | Same four | 20 + 3 auxiliary | 8–16 h |
| 5 | aerosafe-mmr-real | mmr_real | AeBAD | 200, four parts | 6–10 h |
| 6 | aerosafe-mmr-bladesynth | mmr_bladesynth | AeBAD + BladeSynth | 200 + 5 auxiliary, four parts | 8–14 h |
| 7 | aerosafe-patchcore | patchcore | AeBAD | One fit | 0.5–1.5 h |

Aircraft models draw boxes around defects. MMR and PatchCore produce engine anomaly scores
and heatmaps. Two MMR runs are variants of the same algorithm, not distinct architectures.

Dataset titles and contents:

| Title | Keep this folder/file inside the upload |
|---|---|
| aerosafe-asdd | Dataset2.v2-defect-classification.coco/ |
| aerosafe-aircraftsurface | aircraftsurface1.v1i.coco/ |
| aerosafe-agdd | AGDD-main/ |
| aerosafe-imdd | 0to4_aircraft_skin4000pics/ AND 0-4aircraft4000.csv |
| aerosafe-aebad | AeBAD_S/ and AeBAD_V/ with their original contents |
| aerosafe-bladesynth | Dataset_bladesynth/ |

Upload extracted data. The employee-turnover CSV is unrelated. IMDD is required by the
aircraft audit but is not used for box training because its labels do not supply boxes.
Friend 6 must wait until the complete BladeSynth dataset is available.

## Cell 1 — clock and friend number

Change FRIEND only. Run this immediately in a fresh GPU session. The clock is not reset
by re-running this cell in the same Python kernel. A kernel restart inside an existing
GPU session is not a fresh session; do not use it to reset the clock.

```python
import time
if "SESSION_START" not in globals():
    SESSION_START = time.time()
FRIEND = 1   # Your friend number: 1, 2, 3, 4, 5, 6, or 7
PART = 1     # Start with 1. MMR later uses 2, 3, then 4, in fresh sessions.
```

## Cell 2 — download code and install

```python
from pathlib import Path
import os, subprocess, sys
PROJECT = Path('/kaggle/working/aeroinspect')
if not PROJECT.exists():
    subprocess.run(['git', 'clone', 'https://github.com/Leadyhere/AeroSafe.git',
                    str(PROJECT)], check=True, timeout=600)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
               check=True, timeout=1800)
subprocess.run(['git', 'log', '-1', '--oneline'], check=True)
from scripts.kaggle_session import (
    FRIENDS, configure_data, prepare, run_part, download_links, restore, bounded_command
)
MODEL = FRIENDS[FRIEND]
print('Your model:', MODEL, '| Your part:', PART)
```

Use the same Git commit for all friends and all resumed parts. If an old notebook already
has the project, create a fresh notebook/session using the new code before training.

## Cell 3 — restore previous work (skip for first run)

Upload the downloaded archive AND its .sha256 file as a private Kaggle dataset. Attach it
to the next notebook. Put the exact archive path below. Restore only your own trusted
archives. This cell also applies when repeating a partial part.

```python
# First run: leave empty. Resume: paste the path from the input file browser.
PREVIOUS_ARCHIVE = ''
if PREVIOUS_ARCHIVE:
    restore(PREVIOUS_ARCHIVE, MODEL)
```

## Cell 4 — check datasets and prepare

```python
configure_data(MODEL)
prepare(MODEL, SESSION_START)
```

Wait for PASS before continuing. This validates actual loader requirements, not just dataset
titles. Preparation runs again after restore to rebuild aircraft paths for this notebook.
Do not change source images or dataset versions between parts.

## Cell 5 — small test (first session only)

```python
bounded_command(
    [sys.executable, 'scripts/kaggle_train_all.py', '--stage', 'smoke',
     '--model', MODEL, '--config', 'config.yaml'],
    SESSION_START, 9.5
)
```

This checks compatibility; smoke scores are not meaningful accuracy results.

## Cell 6 — train ONE part

```python
run_part(MODEL, PART, SESSION_START)
```

The helper requests a maximum of nine hours for the training command, reduces that budget
by time already spent on setup, and stops the process group at 10.5 hours from Cell 1 if
necessary. This leaves 1.5 hours before the requested 12-hour window. Training's own guard
reserves 45 minutes for packaging. Time estimates cannot guarantee that a specified number
of epochs will finish. If stopped midway through an epoch, repeat that epoch after resume.
Kaggle interruptions, machine hangs, and download speed are outside this timer's control.

## Cell 7 — clickable downloads (ALL friends, after EVERY part)

```python
download_links(MODEL)
```

Cell 6 also displays these links automatically. Download both the `.tar.gz` and its `.sha256`.
They contain resumable state; intermediate MMR parts are not yet calibrated deployment models.
Links are local to the notebook session, not permanent public links. If the browser download
does not work, use the notebook's output-file browser and save the outputs before ending the
session. Files are in `/kaggle/working/aeroinspect/artifacts/`.

If a hard interruption prevented archive completion, the helper offers the last completed
raw checkpoint and config. Download those immediately. In a fresh notebook restore that
checkpoint explicitly before Cell 4:

```python
import shutil
from scripts.kaggle_session import state_path
# Only for emergency raw-checkpoint recovery; use your exact attached file path:
RAW_STATE = ''
if RAW_STATE:
    destination = state_path(MODEL)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW_STATE, destination)
```

A `.pending` checkpoint or archive without its checksum is unfinished. Do not use it.
An emergency raw aircraft checkpoint may not include the earlier best-validation weights;
prefer a complete archive whenever available.

## Which part next?

- Aircraft models (RT-DETRv2, RT-DETR, Deformable DETR, Faster R-CNN): use PART 1–4.
  Cumulative targets are 8, 13, 18, 23. Part 1 has 3 auxiliary + 5 main epochs;
  each later part has 5 main epochs. Repeat the current part if it stops early.
- Friend 5: completed targets are 50, 100, 150, 200 for PART 1, 2, 3, 4 respectively.
- Friend 6: targets are 55, 105, 155, 205 (five auxiliary epochs are in the first part).
- Friend 7: one fit, PART=1. PatchCore does not support resuming a half-finished fit.

Only advance PART after reaching its target. For any `partial` archive, repeat the same PART
after restore. Download files and end the session before starting the next one.

## Accuracy and final model choice

Compare transformer candidates on the same validation split; keep the test split untouched
while tuning. Evaluate the selected configuration once on the frozen test split. Report
aircraft mAP@50:95, recall and missed defects; engine AUROC, AP and false alarms. Do not
promise 95% accuracy or compare aircraft mAP with engine AUROC. More transformers alone do
not guarantee improvement. The full seven-run comparison is for experiments; an interviewer
can be shown the best validated aircraft model, one engine model, and the baseline evidence.

## Closely related projects

- Aircraft-skin comparison: https://github.com/cparyoto/Aircraft-Skin-Defect-Detection-YOLOv9-Vs.-RT-DETR
- Official MMR and AeBAD: https://github.com/zhangzilongc/MMR
- Industrial anomaly framework: https://github.com/open-edge-platform/anomalib

These are references, not proof of our model's accuracy. Keep source attribution and check
licenses before incorporating their code or distributing data/weights. Current local weights
exceed GitHub's ordinary 100 MiB file limit. Keep code on main and publish evaluated model
assets separately with metrics, a model card, checksums and redistribution permission.
