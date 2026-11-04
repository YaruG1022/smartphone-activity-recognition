# Machine learning and data preparation

Module 1 is implemented as the small NumPy package `harss-data`. It downloads
or accepts a pinned UCI HAR archive, prepares subject-isolated six-channel
windows, saves training-only normalization, and validates offline phone
recordings. The supervised comparison is available as a separate optional
package; Android development remains later work.

Run the commands below from the **repository root**. Python 3.10 or newer is
required. The package depends only on NumPy; tests use Python's `unittest`.

For Module 2, see [the supervised comparison guide](../docs/supervised-module.md)
and `notebooks/02_supervised_comparison.ipynb`. Install the `supervised` and
`notebook` extras before running the comparison or the additional tests.

## Install

Windows PowerShell:

```powershell
py -3 -m venv ml/.venv
.\ml\.venv\Scripts\python.exe -m pip install -e ./ml
```

macOS/Linux:

```bash
python3 -m venv ml/.venv
ml/.venv/bin/python -m pip install -e ./ml
```

In the following commands, `python` means the interpreter in that environment.
On Windows, use `.\ml\.venv\Scripts\python.exe` directly if the environment
is not activated; on macOS/Linux, use `ml/.venv/bin/python`.

## Prepare public data

```text
python -m harss_data download
python -m harss_data prepare
python -m harss_data verify
python -m unittest discover -s ml/tests -v
```

`download` verifies an existing archive rather than downloading it again. Both
the UCI wrapper and its original nested archive are supported. The default
path is `ml/data/raw/uci-har-source.zip`; use `--archive PATH` for another local
copy. A modified or incomplete archive is rejected before signal files are read.

The downloader uses normal HTTPS certificate verification. If a Python runtime
has certificate-store problems on Windows, the equivalent system-validated
download is:

```powershell
New-Item -ItemType Directory -Force ml/data/raw | Out-Null
Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip' -OutFile ml/data/raw/uci-har-source.zip
.\ml\.venv\Scripts\python.exe -m harss_data download
```

Do not disable certificate verification. The final command verifies the same
pinned SHA-256 regardless of the transport used.

`prepare` uses `ml/configs/subjects-v1.json` and writes to
`ml/data/prepared/uci-har-v1/`. It rejects nonempty output directories to avoid
mixing runs. To repeat preparation, supply a new directory:

```text
python -m harss_data prepare --output ml/data/prepared/recheck
python -m harss_data verify --output ml/data/prepared/recheck
```

Outputs are three NPZ files, `normalization.json`, a local `replay.json`, and a
manifest containing source/configuration hashes and artifact checksums. Data
and local environments are ignored by Git. Partition sizes are 5,658 training,
1,694 validation, and 2,947 test windows. Each window has shape `[128, 6]`;
prepared labels are 0–5. All channels retain gravity/unit semantics defined
in [the preprocessing contract](../contracts/preprocessing-v1.md).

Load a batch without requiring PyTorch:

```python
from pathlib import Path
from harss_data.dataset import batches, load_prepared

training = load_prepared(Path("ml/data/prepared/uci-har-v1/train.npz"))
batch = next(batches(training, batch_size=32))
print(batch.x.shape, batch.x.dtype)  # (32, 128, 6), float32
print(batch.y.shape)                # (32,)
```

## Prepare one phone recording offline

The [recording contract](../contracts/sensor-record-v1.md) specifies metadata,
event CSV, and the collection protocol. The 36 entries in
`configs/phone-sessions-v1.csv` are plans, not collected data.

```text
python -m harss_data prepare-recording --metadata ml/data/phone/session.json --events ml/data/phone/events.csv --normalization ml/data/prepared/uci-har-v1/normalization.json --output ml/data/prepared/phone-session
```

This creates standardized windows, session/participant IDs, window start
timestamps, zero-based activity labels, and timing diagnostics. It never fits
new normalization statistics. A session marked `final_test` additionally
requires `--allow-final-test`; use that option only after freezing the method.
The reference rejects partial windows, interpolation across excessive gaps,
and readings outside the annotated stable interval.

The synthetic tests exercise this command, but no real phone data has been
collected. The reference does not reproduce all UCI filtering and does not
measure Android behavior, recognition accuracy, or streaming latency.

## Layout and provenance

- `src/harss_data/`: source verification, UCI reader, normalization, recording
  reference, artifact preparation, and command-line entry point.
- `configs/`: fixed subject split and planned phone sessions.
- `tests/`: checks for data leakage, numerical conventions, and raw-event timing.
- [`docs/data-module.md`](../docs/data-module.md): source, attribution, split,
  acceptance criteria, and limitations.
- [`docs/data-validation.json`](../docs/data-validation.json): recorded full-data
  validation and reproducibility checks.
