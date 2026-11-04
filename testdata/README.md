# Fixed test data

`preprocessing-v1.json` is a synthetic four-point Android signal pattern.
Repeat it 32 times to obtain a `[128, 6]` window. It includes manually specified
normalization and expected standardized output, covering acceleration unit
conversion, gravity retention, channel order, signs, and scaling. It contains
no public-source or personal recordings and no model predictions.

The Python tests run this fixture. Future Android code should replay it with
the same channel order and compare outputs using the declared tolerances.
Passing this fixture does not demonstrate live-phone accuracy or equivalence
to UCI filtering.

Public-data preparation writes a separate **local** `replay.json` containing
one training-source window per class, expected preprocessing, and source row
IDs. It stays in ignored `ml/data/` with the training statistics. Personal
recordings and large raw files also remain outside Git.
