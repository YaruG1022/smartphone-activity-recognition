# Supervised comparison

## Question and scope

Which full-label supervised baseline performs best on the fixed validation
subjects, and what are its training and desktop prediction costs?

This module compares logistic regression, a 1D CNN, a GRU, and a CNN–GRU using
Module 1's unchanged six-channel standardized windows. Training has 5,658
windows from 16 subjects; validation has 1,694 windows from five other subjects.
All training labels are available. The experiment loader never opens
`test.npz`; final test scoring is reserved for the later frozen comparison
with the self-supervised work. This module performs no pretraining,
distillation, augmentation, or phone adaptation.

The protocol is fixed in
[`supervised-v1.json`](../ml/configs/supervised-v1.json). It is one configuration
per architecture, not an exhaustive hyperparameter search.

## Models

| Model | Representation and classifier |
| --- | --- |
| Logistic regression | Per-channel mean, population standard deviation, minimum, maximum, and RMS: 30 features from the same standardized windows. Training-only feature scaling followed by multinomial logistic regression: `lbfgs`, C=1, tolerance 1e-6, at most 2,000 iterations. |
| CNN | Conv1d 6→32, kernel 5, batch normalization, ReLU, max pooling by 2; repeat 32→64. Concatenate global mean and maximum, dropout 0.2, six-class head. |
| GRU | One unidirectional 6→64 GRU over 128 time points; mean pooling over its outputs, dropout 0.2, six-class head. |
| CNN–GRU | The same convolutional front end reduces 128 time points to 32; a 64→64 GRU, mean pooling, dropout 0.2, six-class head follow. |

Inputs are `[batch,128,6]` float32; neural outputs are six logits in the order
in [the model contract](../contracts/model-interface-v1.md). Statistical feature
order is channel-major, then `mean, std, min, max, rms`. Signal standardization
is not refitted or applied twice. The separate logistic feature scaler fits
only training features. None of UCI's separate 561-feature vectors is used.

## Training and selection

- Neural seeds: 570, 571, 572. Fit the deterministic logistic baseline once;
  repeating it would not estimate initialization variability.
- AdamW, learning rate 0.001, weight decay 0.0001; unweighted cross-entropy.
- Batch size 128, training shuffling, no dropped final batch, no augmentation.
- At most 30 epochs; select highest validation macro-F1, with earlier epochs
  winning exact ties. Stop after six epochs without improvement.
- Clip gradient norm at 5.0; nonfinite losses or gradients fail the run.
- CPU, two PyTorch threads and a two-thread BLAS limit. Seed Python, NumPy,
  and PyTorch; request deterministic PyTorch operations.

Before formal fits, each neural architecture must memorize 24 training-only
windows, four per class. This separate mechanics check disables dropout, uses
learning rate 0.01, and allows at most 200 steps. Its accuracy is not a
generalization result and does not select formal hyperparameters.

Save validation accuracy, macro-F1 over the fixed six classes, class
precision/recall/F1, support, and a 6×6 confusion matrix. Save local predictions
with subject IDs and original source row IDs. Report all three neural seeds
with mean and sample standard deviation (`ddof=1`). The baseline has one score
and no invented seed standard deviation. Across-seed dispersion is not a
confidence interval over unseen people.

Rank architectures by mean validation macro-F1, then fewer classifier
parameters, then model name. Different architectures have different parameter
counts and may stop at different epochs. These validation scores select
checkpoints and models, so they remain development estimates.

## Checkpoints, exports, and timing

The logistic NPZ stores coefficients, intercepts, feature-scaler state, and
class order. Reloaded NumPy logits must match the fitted pipeline. Neural
checkpoints store weights, architecture settings, selected epoch, seed, class
order, and data provenance. Reloaded CPU weights must reproduce selected
validation logits exactly in the recorded runtime.

For each neural architecture, export its highest-validation-scoring seed
(ties use the lower seed) to `torch.export` `.pt2`, with fixed `[1,128,6]`
input. Reload and compare on six training windows, one per class, with
`atol=1e-4, rtol=1e-3` and matching predictions. This is a CPU format check;
Android runtime compatibility remains untested.

Training time includes the epoch loop, validation, and metrics. It excludes
initial imports, data preparation, separate sanity checks, and export. Desktop
timing uses 10 warmup and 50 timed CPU calls at batch size 1. Logistic timing
includes feature extraction/scaling; neural timing includes array wrapping and
forward prediction. Report median and 95th-percentile milliseconds. These are
desktop measurements from standardized windows, excluding sensor processing;
they do not establish phone latency, memory, power, or real-recording accuracy.

## Run and resume

Use Python 3.10+ from the repository root. Install CPU PyTorch explicitly, then
the optional comparison and notebook dependencies:

```text
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e "./ml[supervised,notebook]"
python -m unittest discover -s ml/tests -v
python -m harss_supervised
```

Here `python` means the project environment's interpreter, for example
`.\ml\.venv\Scripts\python.exe` on Windows. Prepare Module 1's data first.
The default output is ignored `ml/outputs/supervised-v1/`. A signature binds
configuration, training-code hash, data hashes, and software versions. Completed
fits are verified and reused. An incompatible signature fails; supply
`--output PATH` for a new comparison. An interrupted unfinished fit restarts;
completed fits do not retrain.

The interactive entry point is `ml/notebooks/02_supervised_comparison.ipynb`.
Its training switch calls the same implementation with the same resume checks.
Recorded aggregate results can also be viewed without retraining. Raw data,
full prediction files, and model artifacts stay local; the executed notebook,
aggregate results, configuration, and figures are repository deliverables.

## Interpretation limits and references

One fixed validation group and three seeds do not establish population-wide
generalization. No final test or phone result is reported here. Reproducibility
across hardware and software versions is not guaranteed. Logistic parameter
counts include coefficients and intercepts; its scaler state is retained
separately. Neural counts include trainable parameters, not buffers.

- [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [PyTorch 2.9 GRU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.GRU.html).
- [PyTorch 2.9 reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html).
- [PyTorch 2.9 export](https://docs.pytorch.org/docs/2.9/export.html).
- [Notebook execution](https://nbclient.readthedocs.io/en/latest/client.html).
- [Dataset attribution and input processing](data-module.md).
