"""Standard-library identity checks for openpi checkpoints, assets and server metadata.

Used both in the openpi server environment (fetch/serve) and in the client
environment (metadata verification). No openpi, numpy or network imports.
"""
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse


def local_path(data_home, url):
    """Same mapping as openpi.shared.download.maybe_download: <data_home>/<bucket>/<path>."""
    parsed = urlparse(url)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected a gs://bucket/path URL, got {url!r}")
    return Path(data_home).resolve()/parsed.netloc/parsed.path.strip("/")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root):
    """Order-independent manifest digest over relative paths and file contents; symlinks are rejected."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    entries, total = [], 0
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        for name in sorted(files):
            path = Path(directory)/name
            if path.is_symlink():
                raise ValueError(f"Symlink in checkpoint tree: {path}")
            total += path.stat().st_size
            entries.append(f"{path.relative_to(root).as_posix()}\0{file_sha256(path)}\n")
    if not entries:
        raise ValueError(f"Empty checkpoint directory: {root}")
    manifest = "".join(sorted(entries)).encode()
    return {"files": len(entries), "bytes": total, "tree_sha256": hashlib.sha256(manifest).hexdigest()}


def verify_artifacts(pin, data_home):
    """Check the cached checkpoint tree, norm stats, train-config assets and tokenizer; return local paths."""
    checkpoint = pin["checkpoint"]
    directory = local_path(data_home, checkpoint["dir"])
    found = tree_digest(directory)
    expected = {k: checkpoint[k] for k in ("files", "bytes", "tree_sha256")}
    if found != expected:
        raise ValueError(f"Checkpoint tree mismatch: expected {expected}, found {found}")
    if file_sha256(directory/checkpoint["norm_stats"]) != checkpoint["norm_stats_sha256"]:
        raise ValueError("Norm stats hash mismatch")
    assets = local_path(data_home, pin["config_assets"]["url"])/"norm_stats.json"
    if not assets.is_file() or file_sha256(assets) != pin["config_assets"]["norm_stats_sha256"]:
        raise ValueError(f"Train-config norm stats missing or hash mismatch: {assets}")
    tokenizer = local_path(data_home, pin["tokenizer"]["url"])
    if not tokenizer.is_file() or file_sha256(tokenizer) != pin["tokenizer"]["sha256"]:
        raise ValueError(f"Tokenizer missing or hash mismatch: {tokenizer}")
    return {"checkpoint": directory, "tokenizer": tokenizer}


def server_identity(pin):
    """Metadata the pinned launcher serves and the client requires."""
    return {"openpi_commit": pin["openpi"]["upstream_commit"], "train_config": pin["openpi"]["train_config"],
            "checkpoint_dir": pin["checkpoint"]["dir"], "checkpoint_tree_sha256": pin["checkpoint"]["tree_sha256"]}
