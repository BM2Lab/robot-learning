"""Finite P2 neutral-action IPC smoke; never loads a policy or real driver."""
import argparse
import json
from pathlib import Path

from learning.data.storage import validate_artifact_path
from learning.integrations.isaaclab.libero import LiberoSimulatorClient, bounded_chunk, native_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = validate_artifact_path(args.output_dir, 64*1024**2, smoke=True)
    output.mkdir(parents=True)
    import numpy as np
    from PIL import Image
    with LiberoSimulatorClient(args.socket) as client:
        first = client.observe()
        assert client.observe() == first
        features = native_features(first)
        for key, image in features["images"].items():
            Image.fromarray(image).save(output/(key.rsplit(".", 1)[1]+".png"))
        # Pending actions are invalidated by reset, including an old advance.
        chunk, _ = bounded_chunk(first, np.zeros((2, 7), dtype=np.float32), execute_steps=2)
        client.submit(chunk)
        packet = client.reset()
        if packet.context.episode_id == first.context.episode_id:
            raise RuntimeError("Reset did not advance episode identity")
        from learning.integrations.isaaclab.local_client import RemoteError
        try:
            client.advance(first.context)
        except RemoteError:
            pass
        else:
            raise RuntimeError("Old advance accepted after reset")
        steps = min(3, client.receipt["receipt"]["max_steps"])
        audits = []
        terminal = None
        for _ in range(steps):
            chunk, audit = bounded_chunk(packet, np.zeros((1, 7), dtype=np.float32))
            client.submit(chunk)
            following, terminal, status = client.advance(packet.context)
            if (following.context.step_index != packet.context.step_index+1
                    or any(b.frame_index <= a.frame_index for a, b in zip(packet.images, following.images))):
                raise RuntimeError("Stale frame or source-step mismatch")
            native_features(following)
            packet = following
            audits.append(audit)
        if steps == client.receipt["receipt"]["max_steps"]:
            if terminal != {"reason": "horizon", "success": None} or not status["done"]:
                raise RuntimeError("Missing explicit terminal capture at the configured horizon")
        report = {"passed": True, "steps": steps, "model_actions_applied": 0,
                  "receipt": client.receipt, "initial_state": first.state, "terminal": terminal,
                  "audits": audits, "repeated_observe_stable": True, "reset_invalidates_pending_actions": True}
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
