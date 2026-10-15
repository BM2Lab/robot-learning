"""Explicit runtime paths, pinned assets, and runtime receipts (no home fallbacks)."""
from hashlib import sha256
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import shutil


def file_hash(path):
    with open(path, "rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def check_runtime(pin, runtime_dir, *, download=False):
    runtime = Path(runtime_dir).expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    values = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "TOKENIZERS_PARALLELISM": "false",
              "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "0" if download else "1",
              "TRANSFORMERS_OFFLINE": "0" if download else "1",
              "HF_HOME": str(runtime / "huggingface"), "LIBERO_CONFIG_PATH": str(runtime / "libero"),
              "NUMBA_CACHE_DIR": str(runtime / "numba"), "MPLCONFIGDIR": str(runtime / "matplotlib"),
              "XDG_CACHE_HOME": str(runtime), "CUDA_CACHE_PATH": str(runtime / "cuda")}
    # Must run before any MuJoCo/GL/LIBERO/model imports.
    import sys
    if "libero.libero" in sys.modules or "mujoco" in sys.modules:
        raise RuntimeError("Configure native runtime before importing LIBERO/MuJoCo")
    os.environ.update(values)
    actual = {name: version(name) for name in pin["runtime_versions"]}
    if actual != pin["runtime_versions"]:
        raise RuntimeError(f"Unqualified runtime: {actual}; expected {pin['runtime_versions']}")
    return runtime, actual


def prepare_assets(pin, runtime, *, download=False):
    from huggingface_hub import snapshot_download
    if download and shutil.disk_usage(runtime).free < 2 * 1024**3:
        raise RuntimeError("Asset cache requires 2 GiB free")
    descriptor = pin["assets"]
    root = Path(snapshot_download(repo_id=descriptor["repo_id"], repo_type="dataset",
                                 revision=descriptor["revision"], cache_dir=str(runtime / "huggingface/hub"),
                                 local_files_only=not download))
    if root.name != descriptor["revision"]:
        raise RuntimeError("Wrong asset revision")
    for name in ("scenes", "articulated_objects", "stable_scanned_objects", "stable_hope_objects",
                 "turbosquid_objects"):
        if not (root / name).is_dir():
            raise RuntimeError(f"Incomplete asset snapshot: {name}")
    # hf-libero 0.1.4 otherwise prompts at import and downloads unpinned assets
    # under ~/.cache/libero. Its cache hook is isolated here and version-guarded.
    package = Path(distribution("hf-libero").locate_file("libero/libero"))
    paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
             "init_states": str(package / "init_files"), "assets": str(root),
             "datasets": str(runtime / "unused-datasets-NOT-DOWNLOADED")}
    import yaml
    config_dir = runtime / "libero"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        if yaml.safe_load(config_file.read_text()) != paths:
            raise RuntimeError("Existing LIBERO config differs; use a new runtime directory")
    else:
        config_file.write_text(yaml.safe_dump(paths))
    import libero.libero as native
    native._assets_path_cache = str(root)
    return root


def asset_receipt(root):
    # Hash all 422 MB once, not on every simulation step.
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != ".gitattributes"}
    return {"files": len(files), "tree_sha256": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
