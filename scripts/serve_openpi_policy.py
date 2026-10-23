"""Serve a pinned openpi policy with verified identity, using upstream openpi as a library.

Run in the openpi server environment. Before loading the model it checks that the
imported openpi comes from a clean checkout at the pinned commit and that the cached
checkpoint and tokenizer match the pin; it never downloads. The server publishes
``server_identity(pin)`` as metadata, which the robot-learning client requires.
One zero-input warmup inference (the first call includes JIT compilation) runs
before listening, so the first real chunk can meet its deadline.
Binds 127.0.0.1 unless ``--host`` is given explicitly.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess


def openpi_checkout(expected_commit):
    import openpi
    root = Path(openpi.__file__).resolve().parents[2]
    git = ["git", "-C", str(root)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head != expected_commit or dirty:
        raise RuntimeError(f"openpi at {root} is {head}{' (modified)' if dirty else ''}, pinned {expected_commit}")
    return root


def warmup(policy, pin, repo_root):
    """One discarded inference built from the native contract's keys and shapes."""
    import time
    import numpy as np
    from contracts.droid import DroidIOSpec
    spec = DroidIOSpec.from_json((repo_root/pin["native_contract"]).read_text())
    obs = {c.key: np.zeros((c.height, c.width, 3), np.uint8) for c in spec.observation.cameras}
    obs.update({spec.observation.joint_key: np.zeros(7, np.float32),
                spec.observation.gripper_key: np.zeros(1, np.float32), spec.observation.instruction_key: "warmup"})
    start = time.monotonic()
    policy.infer(obs)
    return time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-pin", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, required=True, help="OPENPI_DATA_HOME holding verified artifacts")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    os.environ["OPENPI_DATA_HOME"] = str(args.data_home.resolve())
    logging.basicConfig(level=logging.INFO, force=True)
    from learning.integrations.openpi.checkpoint import server_identity, verify_artifacts
    pin = json.loads(args.policy_pin.read_text())
    root = openpi_checkout(pin["openpi"]["upstream_commit"])
    paths = verify_artifacts(pin, args.data_home)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    policy = policy_config.create_trained_policy(config.get_config(pin["openpi"]["train_config"]), paths["checkpoint"])
    logging.info("Warmup inference took %.1f s", warmup(policy, pin, Path(__file__).resolve().parents[1]))
    identity = server_identity(pin)
    logging.info("Verified openpi %s and checkpoint %s; serving on %s:%d", root, paths["checkpoint"],
                 args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(policy, host=args.host, port=args.port,
                                                  metadata=identity).serve_forever()


if __name__ == "__main__":
    main()
