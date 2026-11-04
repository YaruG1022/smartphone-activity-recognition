# Model interface v1

Status: input and label conventions are used by data preparation and the
supervised classifiers. The [supervised guide](../docs/supervised-module.md)
describes local checkpoints and `.pt2` CPU checks. Android runtime selection
and on-device validation remain pending.

## Input

- Name: `sensor_window`.
- Shape: `[batch, 128, 6]`; the phone normally uses batch size 1.
- Type: float32, finite values only.
- Channel order and units before standardization:
  `total_acc_x`, `total_acc_y`, `total_acc_z` in g, then
  `body_gyro_x`, `body_gyro_y`, `body_gyro_z` in rad/s.
- Input values are standardized according to `preprocessing-v1.md`, using the
  exported training-only `normalization.json` exactly once.
- A model may transpose internally, but its external interface remains
  time-major within each batch item. Do not infer the layout from dimension
  sizes or put normalization both outside and inside the model.

## Classes

| Prepared ID / output index | UCI ID | Name |
| --- | --- | --- |
| 0 | 1 | `WALKING` |
| 1 | 2 | `WALKING_UPSTAIRS` |
| 2 | 3 | `WALKING_DOWNSTAIRS` |
| 3 | 4 | `SITTING` |
| 4 | 5 | `STANDING` |
| 5 | 6 | `LAYING` |

Keep UCI's spelling `LAYING` as the machine-readable name. The UI may display
"Lying down". This is a six-class, single-label task. There is no trained
unknown or transition class in v1.

## Classifier output and export manifest

The logical output is float32 `activity_logits` with shape `[batch, 6]`, in
the table's order. Apply softmax only when probabilities are needed. Current
Python classifiers return a tensor; `.pt2` graph argument names are generated
by PyTorch. The export manifest provides the logical names and semantics.
The initial CPU exports fix batch size to 1.

Each export must identify its model version and SHA-256, preprocessing version,
input shape, channel order, class order, output semantics, runtime requirements,
and the exact normalization file and its SHA-256. Training provenance must link
the subject split, source archive, configuration, and selected checkpoint.
Quantized interfaces require a new explicit specification.
