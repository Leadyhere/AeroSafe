# Deformable DETR recovery (Account 3)

Use this after the existing setup and dataset-preparation cells succeed. Keep the
working data paths and original SESSION_START. Do not rerun old checkout/hotfix
cells afterward: those restore older helper versions. The detached-HEAD message
is not the training failure.

The observed failure was non-finite predicted boxes after an optimizer step,
including with mixed precision disabled. Its exact T4-specific cause has not
been reproduced locally. The revised path explicitly constructs the model on
CPU, loads and checks the full pretrained safetensors state (including shared
aliases), then replaces only classification heads. It uses FP32 and native
PyTorch deformable attention, and rejects non-finite loss/gradients before an
optimizer update. There is no silent NaN replacement or skipped-batch workaround.

Local verification: 80 tests passed; 3 synthetic CPU optimizer steps plus 3 AGDD
and 3 aircraft steps on training images at reduced resolution (128 pixels for
real data) had finite losses and gradients. Saved state reloaded identically.
This is not a T4/800-pixel validation, an accuracy result, or a guarantee of
convergence. The Kaggle smoke test at the configured resolution must pass.

## Update cell

This changes only the listed source files, not config, datasets, or checkpoints.

```python
import os
import sys
import subprocess
import importlib
from pathlib import Path
import yaml

os.chdir('/kaggle/working/aeroinspect')
assert 'SESSION_START' in globals(), 'Run the session setup first.'
subprocess.run(['git', 'fetch', 'origin', 'main'], check=True, timeout=180)
subprocess.run([
    'git', 'restore', '--source', 'origin/main', '--',
    'src/deformable_loading.py', 'src/aircraft_model.py',
    'scripts/train_aircraft.py', 'scripts/kaggle_session.py', 'src/data.py',
], check=True)
import scripts.kaggle_session as ks
importlib.reload(ks)
config_path = Path('config.yaml')
config = yaml.safe_load(config_path.read_text())
config['training']['mixed_precision'] = False
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
for path in [Path(config['paths']['processed']) / 'aircraft/train.json',
             Path(config['paths']['processed']) / 'aircraft/validation.json',
             Path(config['paths']['processed']) / 'aircraft_auxiliary/agdd_train.json']:
    assert path.is_file(), f'Run your working dataset preparation cell first: {path}'
print('Updated. Run the smoke test next.')
```

## Smoke-test cell

Do not hide exceptions or continue to full training after a failure. The new
Deformable smoke test runs up to three batches in each of the two phases.

```python
SMOKE_PASSED = False
ks.bounded_command([
    sys.executable, 'scripts/train_aircraft.py', '--mode', 'smoke',
    '--transformer', 'deformable_detr', '--config', 'config.yaml',
], SESSION_START, 9.5)
SMOKE_PASSED = True
print('SMOKE PASSED: ready for cycle 1.')
```

## Cycle 1 cell

```python
assert globals().get('SMOKE_PASSED') is True, 'Smoke test must pass first.'
ks.run_until('deformable_detr', 12, SESSION_START)
```

Target 12 includes 3 AGDD and 9 aircraft epochs. Download the verified archive
and matching .sha256 from the notebook Output file browser under
`/kaggle/working/aeroinspect/artifacts`; keep config.yaml too. Use the Output
browser if generated FileLink URLs return 404. Do not reset the session until
outputs are safely saved/downloaded.

## Cycle 2

In a fresh session, restore only a trusted archive from this revised run using
its actual path, then redo current data configuration/preparation, update and
smoke test. Old-loading-policy full checkpoints are rejected, not deleted.
Use `ks.run_until('deformable_detr', 23, SESSION_START, resume_required=True)`
only after the first target is complete. A time-limited partial run must resume
the same target first. Do not loop over targets or reset the clock mid-session.

The existing soft budget accounts for setup and reserves packaging time; the
worker process cutoff is 10.5 hours after SESSION_START. No fixed number of
epochs or completed download transfer is guaranteed within that budget.
