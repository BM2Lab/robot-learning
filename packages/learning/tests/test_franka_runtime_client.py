"""End-to-end against franka-runtime's mock service. Skipped unless both are provided:
FRANKA_RUNTIME_CHECKOUT (for its mock config and contract) and FRANKA_DROID_MOCK_BIN."""
import os
from pathlib import Path
import subprocess
import time

import pytest

from contracts.droid import DroidActionChunk, DroidIOSpec
from contracts.episodes import SampleContext
from learning.integrations.franka_runtime.client import DroidRuntimeClient, RuntimeRequestError

CHECKOUT, BINARY = os.environ.get("FRANKA_RUNTIME_CHECKOUT"), os.environ.get("FRANKA_DROID_MOCK_BIN")
pytestmark = pytest.mark.skipif(not (CHECKOUT and BINARY), reason="franka-runtime mock not provided")


@pytest.fixture
def mock_runtime():
    configs = Path(CHECKOUT)/"configs/fr3"
    process = subprocess.Popen([BINARY, "--config", str(configs/"droid.jointpos.mock.json"),
                                "--contract", str(configs/"droid_fr3_jointpos_v1.contract.json")],
                               env={**os.environ, "FRANKA_DROID_TOKEN": "loopback"}, stdout=subprocess.PIPE, text=True)
    line = process.stdout.readline()
    port = int(line.split("127.0.0.1:")[1].split()[0])
    yield DroidIOSpec.from_json((configs/"droid_fr3_jointpos_v1.contract.json").read_text()), port
    process.terminate()
    process.wait(5)


def client(spec, port, token="loopback"):
    return DroidRuntimeClient("127.0.0.1", port, spec, token=token, client_clock_id="test:CLOCK_MONOTONIC",
                              owner_id="pytest", lease_duration_ns=60*10**9, max_uncertainty_ns=2_000_000)


def test_session_submit_applied_and_release(mock_runtime):
    spec, port = mock_runtime
    runtime = client(spec, port)
    ready = runtime.open("req-1")
    assert ready.capability_id.startswith("droid-jointpos-v1:") and runtime.hello["versions"]["backend"] == "mock"
    q = runtime.state().q_rad
    now = time.monotonic_ns()
    rows = tuple((*[v+0.01*(i+1) for v in q], 0.) for i in range(8))
    chunk = DroidActionChunk(SampleContext("ep", 0, now, "monotonic"), spec.action, (*q, 0.), rows, now+533_333_333)
    envelope, receipt = runtime.submit(chunk)
    assert receipt.status == "accepted" and runtime.next_sequence == 2
    deadline, statuses = time.monotonic()+2, []
    while "applied" not in statuses and time.monotonic() < deadline:
        statuses += [r.status for r in runtime.receipts()[0] if r.sequence == 1]
    assert "applied" in statuses
    runtime.release("test")
    runtime.close()


def test_wrong_token_is_refused(mock_runtime):
    spec, port = mock_runtime
    runtime = client(spec, port, token="wrong")
    with pytest.raises(RuntimeRequestError, match="token"):
        runtime.open("req-2")
    runtime.close()
