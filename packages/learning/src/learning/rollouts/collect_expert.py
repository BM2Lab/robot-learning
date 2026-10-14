"""Collect privileged simulation labels over V2; target coordinates never become inputs."""
import importlib.metadata
import json
import math
import sys
import hashlib
import inspect
from pathlib import Path
from contracts.actions import ActionChunk
from learning.data.demonstrations import EpisodeBuffer, load_episode
from learning.data.storage import validate_artifact_path
from learning.integrations.isaaclab.local_client import LocalSimulatorClient


def collect_expert(socket_path, spec, output, episodes=2, seed=123, max_steps=200, smoke=False):
    if not 2 <= episodes <= 10000 or not 1 <= max_steps <= 200 or seed < 0:
        raise ValueError("Use 2..10000 episodes, 1..200 steps, nonnegative seed")
    if smoke and episodes > 4:
        raise ValueError("Temporary smoke collection is limited to four episodes")
    reserved = episodes*max_steps*sum(c.height*c.width*3 for c in spec.observation.cameras)*2
    output = validate_artifact_path(output, reserved, smoke)
    validation_count = max(1, math.ceil(episodes*.2))
    with LocalSimulatorClient(socket_path, spec) as client:
        if not client.receipt["receipt"].get("expert_demonstrations_enabled"):
            raise ValueError("Start the simulator with --enable-expert-demonstrations")
        output.mkdir(parents=True)
        manifest = {"schema_version": 1, "complete": False, "contract": spec.to_dict(),
            "simulator_receipt": client.receipt["receipt"], "seed": seed,
            "label_source": "privileged_scripted_ik_expert", "episodes": [],
            "split_rule": "last ceil(0.2*N) episode seeds held out; no frame-level mixing",
            "runtime": {"python": sys.version, "numpy": importlib.metadata.version("numpy")}}
        manifest["source_sha256"] = {"collector": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "episode_archive": hashlib.sha256(Path(inspect.getfile(EpisodeBuffer)).read_bytes()).hexdigest()}
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for index in range(episodes):
            packet = client.reset(seed=seed+index)
            buffer = EpisodeBuffer(spec, max_steps)
            terminal = None
            for _ in range(max_steps):
                label = client.request("expert_action", {"context": packet.context.to_dict()})
                if label["label_source"] != "privileged_scripted_ik_expert":
                    raise ValueError("Unexpected demonstration label source")
                chunk = ActionChunk.from_dict(label["chunk"])
                chunk.validate_for(spec, packet.context)
                if len(chunk.values) != 1:
                    raise ValueError("Expert collection uses single-step labels")
                client.submit(chunk)
                following, terminal, _ = client.advance(packet.context)
                # Pair the pre-action sample with the exact submitted label.
                # Auto-reset's following observation is never appended to this episode.
                buffer.append(packet, chunk.values[0])
                packet = following
                if terminal is not None:
                    break
            if terminal is None:
                raise RuntimeError("Collection step budget ended before a complete episode")
            split = "validation" if index >= episodes-validation_count else "train"
            entry = buffer.save(output/f"episode_{index:06d}.npz", terminal, seed+index, split)
            load_episode(output/entry["file"], spec)
            manifest["episodes"].append(entry)
            (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        manifest["complete"] = True
        manifest["all_successful"] = all(e["terminal"]["success"] and not e["terminal"]["timeout"] and not e["terminal"]["invalid_state"] for e in manifest["episodes"])
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
