# Preprocessing v1

Status: v1 specification implemented for public-data preparation and the offline
phone reference in `ml/src/harss_data/`. No phone
collection or on-device equivalence has been demonstrated yet.

## Canonical signal

| Index | Channel | Unit | Meaning |
| --- | --- | --- | --- |
| 0 | `total_acc_x` | g | Device x acceleration, including gravity |
| 1 | `total_acc_y` | g | Device y acceleration, including gravity |
| 2 | `total_acc_z` | g | Device z acceleration, including gravity |
| 3 | `body_gyro_x` | rad/s | Device x angular velocity |
| 4 | `body_gyro_y` | rad/s | Device y angular velocity |
| 5 | `body_gyro_z` | rad/s | Device z angular velocity |

The gyroscope names follow the UCI files; they do not mean that a gravity
subtraction is applied to angular velocity. UCI `body_acc_*`, the complete
nine-channel representation, and the 561-feature vectors are not v1 inputs.

For Android `TYPE_ACCELEROMETER`, divide every axis by **9.80665** to convert
m/s² to g. Keep `TYPE_GYROSCOPE` in rad/s. Do not use
`TYPE_LINEAR_ACCELERATION`, remove gravity, clip values to [-1, 1], rotate with
the display, or independently normalize each window. Android axes are device
axes: x points right, y points toward the top, and z points out of the screen
in the device's natural orientation. Record actual placement and orientation.

## Public windows

Read `total_acc_{x,y,z}_{train,test}.txt` and
`body_gyro_{x,y,z}_{train,test}.txt` from `Inertial Signals/` in UCI HAR v1.
Each text row is one existing 128-point window at 50 Hz. Stack channels in the
order above, giving `[windows, 128, 6]`. Preserve source row indices and subject
IDs. Do not join neighboring rows into a continuous recording or re-window them:
the package does not provide session boundaries or original timestamps.

The source windows already overlap by 64 points. Use `subjects-v1.json` to
separate complete subjects; never randomly divide rows between partitions.
The supplied labels are 1–6; prepared labels are 0–5 in the same order.

The source `features_info.txt` describes a median filter and a third-order
20 Hz Butterworth low-pass filter. Its separate 0.3 Hz gravity separation is
not performed by HARSS because the selected acceleration includes gravity.
The 561-feature normalization mentioned in the source README does not apply to
these six inertial channels. No additional filter is applied to UCI windows.

## Offline phone reference

Raw events follow [sensor-record-v1.md](sensor-record-v1.md). The reference:

1. Validates the entire event file, including strictly increasing timestamps
   within each sensor, finite readings, and both required sensors. Timestamps
   are integer nanoseconds from the same monotonic device clock.
2. Restricts processing to the manually confirmed stable activity interval.
3. Builds a common 50 Hz grid over the intersection of both sensor ranges and
   that stable interval, starting at the latest starting timestamp. No
   extrapolation is allowed. Subtract the grid origin before floating-point
   interpolation to preserve timestamp precision.
4. Interpolates each sensor axis linearly. A point requiring interpolation
   across a sensor gap greater than 100 ms is invalid. An exact observed point
   remains valid. A gap breaks the sequence; windows cannot span it.
5. Converts acceleration to g, keeps angular velocity in rad/s, and emits only
   complete 128-point windows with a 64-point stride within each valid segment.
   Short segments are discarded. Windows never span files or sessions.
6. Applies the saved public-training normalization below, without refitting.

The 100 ms limit is an initial engineering rule, not an experimentally validated
quality threshold. The reference applies **no additional noise filter** to
phone events. Interpolation is not an anti-alias filter. Thus v1 does not
reproduce UCI's acquisition and filtering pipeline, especially at different
actual sampling rates. Preserve raw events and report this domain difference;
validate a revised causal filtering policy on development sessions before
claiming phone accuracy. A filtering change requires a new preprocessing
version and may require retraining. Offline interpolation also does not prove
that streaming latency or buffer handling is acceptable on a phone.

## Training-only standardization

After the subject split, estimate one mean and population standard deviation
per channel over **only training windows and their time points**. Overlapping
source time points consequently receive repeated weight; do not interpret this
as unique-time weighting. Accumulate in float64, using population variance
(`ddof=0`), and store mean, standard deviation, fit subject IDs, window count,
channel order, and contract version in `normalization.json`.

For each value, compute `(x - mean[channel]) / scale[channel]` using the saved
float64 statistics, then cast to float32. Use standard deviation as scale,
except that a value below `1e-8` uses scale 1. No clipping is applied. Reuse
these statistics unchanged for validation, test, and phone recordings.
Neither validation nor test signals may fit these statistics, pretraining, or
distillation targets. A future low-label experiment must separately declare
which training signals and activity labels it uses.

## Reference checks

`testdata/preprocessing-v1.json` contains synthetic values, units, statistics,
and expected standardized values. Data preparation also writes a small local
UCI replay fixture containing one training window per class. Neither fixture
contains expected activity predictions: no classifier has been trained.
Initial cross-platform numeric tolerances are `atol=1e-5`, `rtol=1e-4` for
preprocessing; device checks are still pending.

## Sources

- [UCI dataset record](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  and the downloaded v1 `README.txt` and `features_info.txt`.
- [Android motion sensors](https://developer.android.com/develop/sensors-and-location/sensors/sensors_motion).
- [Android sensor coordinate system](https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview#sensors-coords).
