# Data module

## Scope

Module 1 provides reproducible public-data preparation and a specified phone
collection protocol. It does not train a classifier, run self-supervised
experiments, implement an Android app, or claim phone accuracy.

The source is **Human Activity Recognition Using Smartphones, v1.0** by
Reyes-Ortiz, Anguita, Ghio, Oneto, and Parra. The dataset includes 30 participants,
six activities, and 10,299 supplied windows. Acquisition used a waist-mounted
Samsung Galaxy S II at 50 Hz. Each window has 128 readings (2.56 seconds), with
50% overlap. The official partition has 7,352 training and 2,947 test windows.
The public package contains processed signal windows, not original continuous
raw recordings. Session IDs and original timestamps are unavailable.

## Reproducible source

- [Dataset record and citation](https://doi.org/10.24432/C54S4K).
- [Official downloadable wrapper](https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip).
- Wrapper SHA-256: `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
- Nested `UCI HAR Dataset.zip` SHA-256:
  `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0`.

The downloader verifies the pinned hash before accepting either the official
wrapper or the original nested archive. A changed source requires inspection
and an explicit version update. Large archives and prepared data stay in the
ignored `ml/data/` directory. The setup commands are in
[`ml/README.md`](../ml/README.md).

Attribution: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X.
(2013). *Human Activity Recognition Using Smartphones* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C54S4K.
The accompanying paper is Anguita et al. (2013), *A Public Domain Dataset for
Human Activity Recognition Using Smartphones*, ESANN.

The current UCI record lists CC BY 4.0. The legacy README inside the archive
also contains an older noncommercial notice. Both source statements are
recorded here without resolving that discrepancy. The repository's MIT license
covers its code, not a relicensing of the source dataset. Committed replay
fixtures are synthetic; source-derived samples remain local.

## Fixed participant split

[`subjects-v1.json`](../ml/configs/subjects-v1.json) is the authoritative list.
Five official-training subjects were selected before any model experiments.
This is one fixed development split, not a claim that it is optimal.

| Role | Subject IDs | Count |
| --- | --- | --- |
| Training | 1, 3, 6, 7, 8, 14, 15, 16, 17, 21, 22, 25, 26, 27, 28, 30 | 16 |
| Validation | 5, 11, 19, 23, 29 | 5 |
| Test | 2, 4, 9, 10, 12, 13, 18, 20, 24 | 9 |

The official test subjects are unchanged. Training and validation together
cover exactly the official training subjects. All three sets are disjoint.
No random window split, cross-subject fit, or held-out pretraining is permitted.
The official package lacks session boundaries, so no session identifiers are
inferred from adjacent rows.

## Preparation and acceptance

1. Verify the source archive before reading files; no general-purpose archive
   extraction is needed.
2. Read six inertial channels in the declared order. Validate expected row
   counts, 128 columns, finite values, label mapping, and official subject sets.
3. Apply the fixed subject lists and verify complete, disjoint coverage, with
   all six classes represented in each partition.
4. Estimate per-channel population mean and standard deviation only on the
   16 training subjects. Accumulate in float64 and save the fit provenance.
5. Standardize all partitions with those same statistics, cast to float32,
   and retain labels, subject IDs, and original zero-based row indices.
6. Write a manifest with source/configuration hashes, partition counts,
   per-class counts, array shapes, software versions, and artifact hashes.
7. Save a small local reference with one training window per class and its
   standardized output. Verify saved artifacts and read a batch back.

Only structural counts and signal-processing checks are produced here. There
are no training scores or model-selection results. The test signals are
validated and transformed, never used to fit preprocessing.

## Outputs

`ml/data/prepared/uci-har-v1/` contains `train.npz`, `validation.npz`,
`test.npz`, `normalization.json`, `replay.json`, and `manifest.json`.
Each NPZ contains `x` (float32 `[N,128,6]`), `y` (int64, 0–5),
`subject_id` (int64), and `source_row` (int64). Train/validation source rows
refer to the original official training files; test rows refer to official test
files. Source order is preserved within each prepared partition.

See [preprocessing](../contracts/preprocessing-v1.md),
[model interface](../contracts/model-interface-v1.md), and
[recording format and collection protocol](../contracts/sensor-record-v1.md).
Future model code should consume these artifacts without silently redefining
channel order, labels, subject splits, or statistics.

## Phone preparation

The offline reader validates one raw recording, resamples asynchronous streams,
rejects windows spanning timing gaps, and reuses the saved training statistics.
The phone-session registry is a collection plan. Actual device details, consent,
recordings, independent phone results, and on-device replay remain future work.
The public data's filtering cannot be assumed equivalent to this reference's
phone processing; preserve and report that difference.
