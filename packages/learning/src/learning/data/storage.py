"""Explicit artifact storage policy; bounded /tmp verification is opt-in."""
import os
from pathlib import Path
import shutil


def validate_artifact_path(path, required_bytes, smoke=False):
    path = Path(path).expanduser().resolve()
    if smoke:
        if not path.is_relative_to(Path("/tmp")):
            raise ValueError("Smoke artifacts must be explicitly placed under /tmp")
    else:
        value = os.environ.get("LAB_DATA_ROOT")
        if not value:
            raise RuntimeError("Provision LAB_DATA_ROOT before persistent dataset jobs")
        root = Path(value).expanduser().resolve()
        if not root.is_dir() or not path.is_relative_to(root) or path == root:
            raise ValueError("Artifact directory must be a child of existing LAB_DATA_ROOT")
        # A dedicated data mount is required unless the host has only a system disk and says so.
        if (os.environ.get("LAB_DATA_ROOT_ON_SYSTEM_DISK") != "1"
                and not any(os.path.ismount(p) for p in (root, *root.parents) if str(p) != "/")):
            raise RuntimeError("LAB_DATA_ROOT is not on a mounted data filesystem "
                               "(single-disk host: set LAB_DATA_ROOT_ON_SYSTEM_DISK=1)")
        # Reject source, Python environments, and workspace runtime state.
        for parent in (root, *root.parents):
            if (parent/"AGENTS.md").exists() or (parent/"pyvenv.cfg").exists() or (parent/".git").exists():
                raise ValueError("Datasets may not be stored in source/workspace/environment trees")
    parent = next(p for p in path.parents if p.exists())
    if not os.access(parent, os.W_OK) or shutil.disk_usage(parent).free < required_bytes+64*1024**2:
        raise RuntimeError("Artifact filesystem is not writable or lacks reserved capacity")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifacts: {path}")
    return path
