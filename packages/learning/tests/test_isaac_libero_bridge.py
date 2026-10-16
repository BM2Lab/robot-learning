import base64
import subprocess
import sys

import pytest

from contracts.libero import LiberoIOSpec
from contracts.episodes import SampleContext
from contracts.samples import ObservationPacket, RGBSample
from learning.integrations.isaaclab.libero import bounded_chunk, native_features


def packet():
    spec = LiberoIOSpec()
    context = SampleContext("episode", 0, 0, "simulation")
    encoded = base64.b64encode(bytes(256*256*3)).decode()
    images = tuple(RGBSample(c.key, c.optical_frame, 256, 256, 1, context, encoded)
                   for c in spec.observation.cameras)
    return ObservationPacket(spec.contract_id, context, (0.,)*8, images, "soup into basket")


def test_native_chunk_clipping_is_explicit_and_audited():
    raw = [[2., 0., 0., 0., 0., 0., -2.], [0.]*7]
    chunk, audit = bounded_chunk(packet(), raw)
    assert len(chunk.values) == 1
    assert chunk.values[0][0] == 1 and chunk.values[0][-1] == -1
    assert audit["components_clipped"] == 2
    assert audit["peak_raw_input"] == 2
    assert chunk.valid_until_ns == 100_000_000


def test_native_decode_does_not_reflip_images():
    result = native_features(packet())
    assert result["state"].shape == (8,)
    assert set(result["images"]) == {"observation.images.image", "observation.images.image2"}
    assert result["images"]["observation.images.image"].shape == (256, 256, 3)


@pytest.mark.parametrize("raw", [[[0.]*6], [[float("nan")]*7], []])
def test_bad_chunks_rejected(raw):
    with pytest.raises(ValueError):
        bounded_chunk(packet(), raw)


def test_bridge_imports_no_ml_or_simulator():
    subprocess.run([sys.executable, "-B", "-c",
        "import sys; import learning.integrations.isaaclab.libero; "
        "assert not any(x in sys.modules for x in ('torch','isaaclab','lerobot','mujoco'))"], check=True)
