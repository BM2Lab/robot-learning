"""Reference export remains optional, bounded and non-overwriting."""
import subprocess
import sys

import pytest

from learning.integrations.libero.reference import export_reference


def test_optional_imports_remain_lazy():
    subprocess.run([sys.executable, "-B", "-c",
        "import sys; import learning.integrations.libero.reference; "
        "assert not any(x in sys.modules for x in ('torch','mujoco','libero','robosuite','lerobot'))"], check=True)


def test_export_never_overwrites_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        export_reference(tmp_path / "unused.json", tmp_path / "unused-runtime", tmp_path)


def test_export_requires_existing_parent(tmp_path):
    with pytest.raises(RuntimeError, match="existing output parent"):
        export_reference(tmp_path / "unused.json", tmp_path / "unused-runtime", tmp_path / "absent" / "output")
