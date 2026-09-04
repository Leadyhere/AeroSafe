# AeroInspect local dataset audit

Audit date: 2026-09-04

This report describes the dataset files observed under `datasets/`. It separates structural validity from
scientific suitability: a dataset can contain readable files and still have weak labels, duplicate
content, class imbalance, or a domain mismatch.

## Executive result

- No corrupt real image files were found in ASDD, aircraftsurface1, AGDD, IMDD, or AeBAD.
- No referenced COCO image files or AeBAD-S anomaly masks are missing.
- Raw ASDD and aircraftsurface1 overlap heavily. Duplicate-aware preparation is mandatory.
- IMDD has no localization annotations and must not enter detector training until real boxes are added.
- The extracted BladeSynth directory is present; the automated engine audit verifies it before training.
- More images do not guarantee real-world accuracy. Label precision, independent scenes, domain coverage,
  and an external test set matter more than the raw file count.

## ASDD

- 1,103 RGB images, all readable and 640 x 640.
- 1,674 COCO boxes; no missing references or invalid zero-area boxes.
- One box extends slightly beyond declared image bounds and is clipped safely during loading.
- Raw annotations: Crack 396, Missing Bolt 381, composite `Dent Scratch or Paint Falling` 895, and
  generic `defect` 2.
- The composite and generic labels map to `surface_damage`; they are not falsely split into dent,
  scratch, or paint damage.
- COCO category tables contain duplicate `Crack` names with unused IDs. This is metadata noise; the
  normalized loader resolves it by name.
- No byte-identical files occur within the ASDD export or across its published train/valid/test folders.

## aircraftsurface1

- 11,091 RGB images, all readable and 640 x 640.
- 30,246 COCO boxes; no missing image/category references or invalid zero-area boxes.
- 108 boxes extend slightly outside declared bounds and are clipped safely.
- 24 images have no boxes and remain valid negative detector examples.
- Raw annotations: Missing-Head 11,000, Paint-off 5,508, Dent 4,667, Crack 3,925, Corrosion 3,496,
  Scratch 935, and Repair 715.
- `Repair` is not a defect class and its 715 annotations are intentionally excluded.
- An unused category named `2` appears in each COCO category table and is ignored.
- Two byte-identical duplicate files occur within this export.

## Combined aircraft detector corpus

- 12,194 raw images become 11,866 unique images after removing 328 exact copies.
- 326 of the 328 exact duplicate groups are shared between ASDD and aircraftsurface1.
- 195 exact-image groups disagree at the normalized label-set level, usually because ASDD uses the
  broad `surface_damage` label while aircraftsurface1 supplies a specific dent, scratch, or paint label.
- Preparation now retains the most specific/rich annotation record in each exact group, using
  aircraftsurface1 only as a deterministic tie-breaker. It does not merge conflicting boxes blindly.
- The verified original box distribution is missing-fastener 11,280, paint-damage 5,504, dent 4,667,
  crack 4,234, corrosion 3,494, scratch 935, and broad surface-damage 642. These names remain in audit
  metadata, while training collapses all 30,756 retained boxes to one reliable `defect` target.
- 1,896 perceptual near-duplicate relationships are grouped into a single prepared split so related
  augmentations cannot leak between training, validation, and testing.
- RT-DETR v2, RT-DETR, Deformable DETR, and Faster R-CNN use the exact same grouped binary split. This
  makes their frozen-test comparison meaningful and removes cross-dataset taxonomy conflicts.

## AGDD

- 438 usable paired RGB images, all readable and 640 x 640: 394 train images and 44 validation images
  across the two supplied modalities.
- Every image has a matching rectangular YOLO label file; no malformed or out-of-range normalized boxes
  were found.
- Original label counts before duplicating across modalities are: contusion 138, scratches 217, crack 29,
  and spot 215.
- Contusion, scratches, and crack map to the binary `defect` target. `spot` is excluded because its meaning
  is not defensibly compatible with the target.
- AGDD is used only for auxiliary detector pretraining; final validation/testing uses the main aircraft
  corpus.

## IMDD aircraft subset

- 4,281 RGB-encoded but visually monochrome 512 x 512 images; all files and all 4,281 CSV rows match.
- Class counts are dent 1,346, crack 996, missing-head 928, paint-off 789, and scratch 222. The largest
  class is about 6.1 times the smallest.
- No byte-identical duplicates or cross-class exact conflicts were found.
- Perceptual screening found 117 near pairs in 95 groups. One cross-class hash candidate was manually
  reviewed and was a false-positive similarity, not the same image.
- 1,673 images (39.1%) have black pixels over more than 10% of their outer border, mainly from rotation
  augmentation. This is acceptable for an image-level auxiliary check but is a domain-shift warning.
- The CSV provides one class and text description per image, with no bounding boxes or masks.
- Current role: audit-only and eligible for a future external image-level presence evaluation. It is
  excluded from all four aircraft-localization runs.
- To use it for detector training, manually annotate true boxes for a reviewed subset in Roboflow or CVAT,
  export new COCO annotations, and keep that boxed derivative separate from the original CSV dataset.
  Whole-image pseudo-boxes are prohibited.

## AeBAD

- AeBAD-S contains 521 raw normal training images and 1,639 test images. One exact duplicate occurs within
  normal training (`IMG_7138.png` in two domain folders), leaving 520 unique normals.
- The corrected grouped split produces 416 training normals and 104 threshold-calibration normals for seed
  42, with no repeated filename shared across those sets.
- AeBAD-S test contains 490 normal and 1,149 anomalous images.
- All 1,149 anomaly masks exist, match their image dimensions, are non-empty, and contain foreground.
- No exact image is shared between AeBAD-S training and test.
- AeBAD-V contains 707 real training frames and 2,703 test frames after metadata filtering. Sampling every
  tenth frame independently per video supplies 73 auxiliary normal frames.
- The archive contains 287 macOS resource/metadata files. `._*` and `__MACOSX` entries are explicitly
  ignored everywhere.

## BladeSynth readiness contract

The usable extracted dataset is `datasets/Dataset_bladesynth/`. Preparation requires the official five
image classes—Normal, Dent, Nick, Scratch, and Corrosion—with exactly 2,500 images per class. It decodes all
five classes, reports their dimensions, counts mask files, and rejects an incomplete or corrupt export.
The leftover `.crdownload` is ignored and must not be uploaded as training data.

- Verified: 2,500 1024 x 1024 images in each class (12,500 images total), plus 10,000 anomaly masks.
- Only the 2,500 Normal images are eligible for the separate auxiliary MMR experiment.

The primary MMR remains a real-only AeBAD-S model. A separate experimental MMR checkpoint uses only
BladeSynth Normal images plus sampled AeBAD-V normal frames for auxiliary pretraining, then fine-tunes on
AeBAD-S. Synthetic anomaly classes are never treated as normal, and experimental metrics are clearly
marked in the seven-run comparison.

## Accuracy expectations

The detector corpus is large enough for meaningful transfer learning, but no accuracy can be promised in
advance. Within-dataset mAP may be good while performance on new maintenance-camera images remains weak
because the source images are curated, augmented, grayscale in IMDD, and unevenly labeled. The AeBAD-S
normal set is relatively small, but pretrained feature extractors and anomaly detection are designed for
that regime; the official domain shift remains deliberately difficult.

Judge success using held-out mAP50-95, binary defect AP/recall, image AUROC, image average precision,
pixel AUROC/AUPRO, and—most importantly—a frozen external dataset collected with the intended camera and
inspection procedure. Do not choose a model from training accuracy alone.
