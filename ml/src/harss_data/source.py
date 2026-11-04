"""Download and read the pinned UCI archives without extracting paths."""

import hashlib
import io
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from .spec import DATASET_SHA256, SOURCE_URL, WRAPPER_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_hash(path: Path) -> str:
    actual = sha256(path)
    if actual not in (WRAPPER_SHA256, DATASET_SHA256):
        raise ValueError(f"Unrecognized source SHA-256: {actual}; source not read")
    return actual


def download(destination: Path) -> Path:
    """Accept an existing verified archive, or download atomically with TLS."""
    destination = Path(destination)
    if destination.exists():
        verified_hash(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".part", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        verified_hash(Path(temporary))
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def open_dataset(path: Path) -> zipfile.ZipFile:
    """Return the inner dataset zip after verifying all relevant bytes."""
    actual = verified_hash(path)
    if actual == DATASET_SHA256:
        return zipfile.ZipFile(path)
    with zipfile.ZipFile(path) as wrapper:
        data = wrapper.read("UCI HAR Dataset.zip")
    if hashlib.sha256(data).hexdigest() != DATASET_SHA256:
        raise ValueError("Nested dataset archive does not match its pinned hash")
    return zipfile.ZipFile(io.BytesIO(data))
