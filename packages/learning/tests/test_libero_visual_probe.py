"""Frozen-input factorial coverage without importing the model or simulator."""
from pathlib import Path
import runpy


def test_all_state_and_camera_combinations():
    module = runpy.run_path(str(Path(__file__).resolve().parents[3]/
        "scripts/probe_libero_visual_inputs.py"))
    cases = module["conditions"]()
    assert len(cases) == len(set(cases)) == 8
    for state in ("native", "isaac"):
        assert set((a, w) for s, a, w in cases if s == state) == {
            ("native", "native"), ("native", "isaac"),
            ("isaac", "native"), ("isaac", "isaac")}
