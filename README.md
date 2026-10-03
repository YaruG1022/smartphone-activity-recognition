# HARSS

Human activity recognition with smartphone sensors.

## Repository structure

```text
mobile/       Android app: sensor recording, preprocessing, inference, and UI
ml/           Python data preparation, training, evaluation, and model export
contracts/    Shared sensor-data and model-input specifications
testdata/     Small, fixed samples for cross-platform replay tests
docs/         Architecture notes, research, and experiment records
```

The Android project will be created in `mobile/`. Model code belongs in
`ml/src/`, configuration files in `ml/configs/`, and Python tests in
`ml/tests/`. Keep large datasets and generated models out of Git unless a
specific artifact is needed for a reproducible release.
