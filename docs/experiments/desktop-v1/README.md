# Desktop activity-recognition experiments

## Does masked pretraining help with fewer labels?

The paired comparison uses the same one-dimensional convolutional neural network
(CNN) in both arms. It was the highest-scoring architecture in the earlier
[supervised validation comparison](../supervised-v1/README.md). The data split
stays fixed at 16 training, five validation, and nine official test subjects.
Every window contains 128 time steps and six standardized sensor channels.
[Data preparation](../../data-module.md) documents the channel order and
training-only normalization.

For each of three seeds (570, 571, 572), pretraining uses the signals from all
5,658 training windows without activity labels. It masks two nonoverlapping
16-step spans across all six channels, giving 25% masked time steps per window.
Masked values are replaced by zero in standardized units. The model minimizes
mean squared error on those steps for 15 fixed epochs.
The pretrained arm transfers the encoder and trains the six-class classifier
end to end. The scratch arm starts from random weights. The classifier head
initialization, labeled subset, batch order, optimizer, maximum training budget,
and validation checkpoint rule are matched within each seed and label budget.
The labeled subsets are nested within each seed. They include 566, 1,415, and
5,658 windows at the 10%, 25%, and 100% budgets. The sampler includes at least
one window from each observed training subject/activity pair and fills the rest
in proportion to stratum size. The budgets count windows, which may overlap
in time.

## How is the final classifier selected?

The four full-label supervised baselines and the full-label pretrained CNN are
eligible. The 100% scratch reruns remain in the paired experiment. Candidates
are ranked by mean validation macro-F1, then fewer parameters, then candidate ID.
For the leading candidate, the checkpoint with the highest validation macro-F1
is selected, with the lower seed breaking exact ties. The freeze records the
selected checkpoint and all comparison checkpoint hashes before official test
scoring. The test evaluation reports accuracy, macro-F1, per-class scores, and
confusion matrices for the fixed comparison. The paired contrast is pretrained
minus scratch for each label budget and seed.

## Reproduce

Run from the repository root with Python from the project environment. Follow
the [data setup](../../../ml/README.md) and
[supervised setup](../../supervised-module.md) first. The optional worker flag
runs one process per seed with two CPU threads per process.

```text
python -m harss_supervised
python -m harss_ssl --workers 3
python ml/scripts/freeze_desktop_experiment.py --output ml/outputs/desktop-v1-freeze.json
python ml/scripts/evaluate_frozen_test.py --freeze ml/outputs/desktop-v1-freeze.json --verify-only
python ml/scripts/evaluate_frozen_test.py --freeze ml/outputs/desktop-v1-freeze.json --output ml/outputs/desktop-v1
```

The recorded freeze identifies the exact artifacts from this execution. Create
a new freeze for a reproduced run. Raw signals, model weights, and per-window
predictions remain in local ignored directories.
