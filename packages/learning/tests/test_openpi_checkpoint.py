import hashlib
import json
from pathlib import Path

import pytest

from learning.integrations.openpi.checkpoint import (local_path, server_identity, tree_digest,
                                                     verify_artifacts)

ROOT = Path(__file__).resolve().parents[3]


def make_cache(home):
    root = home / "bucket/ckpt"
    (root / "params/d").mkdir(parents=True)
    (root / "assets/droid").mkdir(parents=True)
    (root / "params/_METADATA").write_bytes(b"meta")
    (root / "params/d/shard").write_bytes(b"weights")
    (root / "assets/droid/norm_stats.json").write_bytes(b"{}")
    (home / "assets/cfg").mkdir(parents=True)
    (home / "assets/cfg/norm_stats.json").write_bytes(b"{}")
    (home / "big_vision").mkdir()
    (home / "big_vision/tok.model").write_bytes(b"tokenizer")
    pin = {"openpi": {"upstream_commit": "abc", "train_config": "cfg"},
           "checkpoint": {"dir": "gs://bucket/ckpt", **tree_digest(root), "norm_stats": "assets/droid/norm_stats.json",
                          "norm_stats_sha256": hashlib.sha256(b"{}").hexdigest()},
           "config_assets": {"url": "gs://assets/cfg", "norm_stats_sha256": hashlib.sha256(b"{}").hexdigest()},
           "tokenizer": {"url": "gs://big_vision/tok.model", "sha256": hashlib.sha256(b"tokenizer").hexdigest()}}
    return root, pin


def test_local_path_matches_openpi_cache_layout(tmp_path):
    assert local_path(tmp_path, "gs://openpi-assets-simeval/pi05_droid_jointpos/") == \
        tmp_path.resolve() / "openpi-assets-simeval/pi05_droid_jointpos"
    for bad in ("/local/path", "s3://bucket/x", "gs://bucket"):
        with pytest.raises(ValueError):
            local_path(tmp_path, bad)


def test_verify_artifacts_accepts_exact_tree(tmp_path):
    root, pin = make_cache(tmp_path)
    assert verify_artifacts(pin, tmp_path)["checkpoint"] == root.resolve()
    assert pin["checkpoint"]["files"] == 3 and pin["checkpoint"]["bytes"] == 13


@pytest.mark.parametrize("change", ["content", "rename", "extra", "tokenizer", "config_assets", "symlink"])
def test_verify_artifacts_rejects_any_change(tmp_path, change):
    root, pin = make_cache(tmp_path)
    if change == "content":
        (root / "params/d/shard").write_bytes(b"weightz")
    elif change == "rename":
        (root / "params/d/shard").rename(root / "params/d/shard2")
    elif change == "extra":
        (root / "params/extra").write_bytes(b"")
    elif change == "tokenizer":
        (tmp_path / "big_vision/tok.model").write_bytes(b"other")
    elif change == "config_assets":
        (tmp_path / "assets/cfg/norm_stats.json").unlink()
    else:
        (root / "params/link").symlink_to(root / "params/_METADATA")
    with pytest.raises(ValueError):
        verify_artifacts(pin, tmp_path)


def test_missing_checkpoint_and_server_identity(tmp_path):
    _, pin = make_cache(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_artifacts(pin, tmp_path / "empty")
    real = json.loads((ROOT / "configs/policies/pi05_droid_jointpos.json").read_text())
    identity = server_identity(real)
    assert identity["openpi_commit"] == real["openpi"]["upstream_commit"]
    assert identity["checkpoint_tree_sha256"] == real["checkpoint"]["tree_sha256"]
