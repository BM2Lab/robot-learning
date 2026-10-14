"""Exact policy/companion snapshots; no ML imports until explicitly requested."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil


@dataclass(frozen=True)
class PinnedSnapshots:
    policy: Path
    companion: Path
    policy_revision: str
    companion_revision: str


def prepare_snapshots(pin, cache_dir, *, download=False):
    """Download only policy weights and companion config/tokenizer (not VLM weights)."""
    for key in ("policy", "vlm_and_tokenizer"):
        if not re.fullmatch(r"[0-9a-f]{40}", pin[key]["revision"]):
            raise ValueError("Model revisions must be immutable 40-character commit IDs")
    cache = Path(cache_dir).expanduser().resolve()
    if download:
        parent = next(p for p in (cache, *cache.parents) if p.exists())
        if shutil.disk_usage(parent).free < 4 * 1024**3:
            raise RuntimeError("Model cache needs at least 4 GiB free before downloading")
    from huggingface_hub import snapshot_download
    policy = pin["policy"]
    companion = pin["vlm_and_tokenizer"]
    paths = []
    for descriptor, patterns in ((policy, ["*.json", "*.safetensors"]),
                                 (companion, ["*.json", "*.txt", "*.model"])):
        path = Path(snapshot_download(repo_id=descriptor["repo_id"], revision=descriptor["revision"],
                                     cache_dir=str(cache), allow_patterns=patterns,
                                     local_files_only=not download))
        if path.name != descriptor["revision"]:
            raise RuntimeError("Resolved snapshot does not match pinned revision")
        paths.append(path)
    config_path = paths[0] / "config.json"
    if sha256(config_path.read_bytes()).hexdigest() != policy["config_sha256"]:
        raise RuntimeError("Checkpoint config hash mismatch")
    config = json.loads(config_path.read_text())
    expected = pin["checkpoint_features"]
    if (config["input_features"]["observation.state"]["shape"] != expected["state_shape"]
            or config["output_features"]["action"]["shape"] != expected["action_shape"]
            or config["chunk_size"] != expected["chunk_size"]):
        raise RuntimeError("Checkpoint features do not match manifest")
    required = ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    for name in required:
        if not (paths[0] / name).is_file():
            raise RuntimeError(f"Incomplete policy snapshot: {name}")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        if not (paths[1] / name).is_file():
            raise RuntimeError(f"Incomplete companion snapshot: {name}")
    return PinnedSnapshots(*paths, policy["revision"], companion["revision"])
