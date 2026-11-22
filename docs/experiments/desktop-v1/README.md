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
in time. Both arms use channel normalization fitted without labels on all
training windows.

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

## Results

Masked pretraining improved mean validation macro-F1 at all three label
budgets. On the official test set, all three mean differences favored scratch
by 0.23 to 0.71 percentage points. Individual pairs included both gains and
losses at every budget. The benefit on new subjects remains uncertain.

| Label budget | Validation scratch | Validation pretrained | Test scratch | Test pretrained | Test paired change (mean ± SD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.9271 | 0.9328 | 0.9021 | 0.8985 | −0.0036 ± 0.0143 |
| 25% | 0.9305 | 0.9373 | 0.9093 | 0.9021 | −0.0071 ± 0.0133 |
| 100% | 0.9358 | 0.9420 | 0.9108 | 0.9085 | −0.0023 ± 0.0279 |

All values are macro-F1 over the same six classes. Each row uses three seed
pairs. SD is the sample standard deviation over those pairs. The seeds share
the same subject split, so this spread measures variation across runs on one
held-out group of nine subjects. Each pretraining run took about 23 seconds
with three concurrent seed workers and two CPU threads per worker. Training
masked reconstruction error fell from 0.762–0.779 at epoch 1 to 0.409–0.414
at epoch 15. [Validation results](validation-results.json) ·
[Official test results](test-results.json)

The full-label pretrained CNN had the highest mean validation macro-F1
(0.9420). The selected checkpoint was seed 572 at epoch 19, with validation
macro-F1 of 0.9485. It scored **92.09% accuracy and 0.9211 macro-F1** on
2,947 official test windows. Its two lowest test recalls were standing
(80.83%) and sitting (85.13%). The other checkpoints were scored for the
fixed comparison. The model selection remains the one recorded before
scoring in commit `c248b66`. [Validation freeze](freeze.json) ·
[Official test results](test-results.json)

| Full-label supervised baseline | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Logistic regression (one fit) | 0.8320 | 0.8288 |
| CNN | 0.9104 ± 0.0101 | 0.9108 ± 0.0099 |
| Gated recurrent unit (GRU) | 0.8872 ± 0.0061 | 0.8863 ± 0.0068 |
| CNN–GRU | 0.9067 ± 0.0098 | 0.9068 ± 0.0101 |

The original CNN baseline and the 100% scratch reruns produced identical
weights and validation and test logits for each seed. The 28 test score
records therefore include three repeated predictions from those matched
reruns. Accuracy, macro-F1, per-class results, and confusion matrices were
independently recomputed from the saved predictions. The aggregate scores and
paired differences were checked against the individual runs.

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
