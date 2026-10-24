"""Run one recorded DROID episode: cameras + pinned openpi server + franka-runtime DROID service.

Talks only to a franka-runtime service; never to robot hardware directly. Against the
D4 mock (``franka-droid-mock``) this is the loopback test; against a real runtime it is
subject to that runtime's commissioning and authorization. The session token is read
from a file only the user can read; it never appears on the command line.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import time

from contracts.droid import DroidIOSpec
from learning.data.storage import validate_artifact_path
from learning.integrations.franka_runtime.client import DroidRuntimeClient
from learning.integrations.openpi.cameras import DroidCameraRouter
from learning.integrations.openpi.droid import OpenPIDroidClient, load_pin
from learning.rollouts.droid_runner import DroidEpisodeRecorder, DroidRunner


def source_revision(path):
    """Git commit and dirty flag of the checkout containing ``path`` (None outside Git)."""
    import subprocess
    directory = Path(path).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def read_token(path):
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"{path} must not be group/world accessible (chmod 600)")
    token = path.read_text().strip()
    if not token:
        raise ValueError("Empty token file")
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy-pin", type=Path, required=True)
    parser.add_argument("--exterior-camera", type=Path, required=True, help="sensors RealSense config")
    parser.add_argument("--wrist-camera", type=Path, required=True, help="sensors RealSense config")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--runtime-host", required=True)
    parser.add_argument("--runtime-port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="allow a bounded output directory under /tmp")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--replan-after-rows", type=int, default=5,
                        help="5 of 8 rows leaves about 90 ms slack at the measured 110 ms capture-to-receipt p95")
    parser.add_argument("--shadow", action="store_true",
                        help="require a runtime whose hello reports a shadow backend (evaluates, never executes)")
    parser.add_argument("--max-uncertainty-us", type=int, default=2000)
    parser.add_argument("--max-consecutive-rejections", type=int, default=3)
    parser.add_argument("--preview-dir", type=Path,
                        help="write the latest policy-input images here for scripts/view_policy_inputs.py")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 500:
        parser.error("--max-steps must be 1..500")
    output = validate_artifact_path(args.output_dir, 2*1024**3 if not args.smoke else 512*1024**2, smoke=args.smoke)
    token = read_token(args.token_file)
    from sensors.cameras.latest import LatestFrameReader
    from sensors.cameras.realsense import RealSenseCamera, RealSenseConfig
    spec = DroidIOSpec.from_json(args.contract.read_text())
    pin = load_pin(args.policy_pin)
    recorder = DroidEpisodeRecorder(output)
    with ExitStack() as stack:
        cameras, readers = {}, {}
        for key, path in ((spec.observation.cameras[0].key, args.exterior_camera),
                          (spec.observation.cameras[1].key, args.wrist_camera)):
            camera = stack.enter_context(RealSenseCamera(RealSenseConfig.from_json(path.read_text())))
            cameras[key] = camera.receipt
            readers[key] = stack.enter_context(LatestFrameReader(camera))
        router = DroidCameraRouter(spec, readers, max_age_ns=100_000_000, max_skew_ns=40_000_000)
        policy = OpenPIDroidClient.connect(spec, pin, args.policy_host, args.policy_port)
        runtime = DroidRuntimeClient(args.runtime_host, args.runtime_port, spec, token=token,
                                     client_clock_id=f"{os.uname().nodename}:runner:CLOCK_MONOTONIC",
                                     owner_id=f"droid-runner:{os.getpid()}", lease_duration_ns=300*10**9,
                                     max_uncertainty_ns=args.max_uncertainty_us*1000)
        stack.callback(runtime.close)
        ready = runtime.open(f"droid-run-{time.time_ns()}")
        backend = str(runtime.hello.get("versions", {}).get("backend", ""))
        if args.shadow != backend.startswith("shadow"):
            runtime.release("backend/shadow mismatch")
            raise SystemExit(f"Runtime backend {backend!r} does not match --shadow={args.shadow}; refusing to run")
        import learning
        import sensors
        sources = {"robot-learning": source_revision(learning.__file__), "sensors": source_revision(sensors.__file__)}
        recorder.event("session", hello=runtime.hello, ready=ready.to_dict(), clock=runtime.estimate.to_dict(),
                       policy=policy.receipt(), cameras=cameras, sources=sources)
        deadline = time.monotonic()+5
        while True:  # first fresh frames from both cameras
            try:
                router.capture()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        preview = None
        if args.preview_dir is not None:
            from learning.rollouts.policy_preview import PolicyPreview
            preview = PolicyPreview(args.preview_dir)
            stack.callback(preview.close)
        runner = DroidRunner(spec, policy, runtime, router, recorder, instruction=args.instruction, preview=preview,
                             max_steps=args.max_steps, replan_after_rows=args.replan_after_rows,
                             max_state_age_ns=100_000_000, max_state_camera_skew_ns=100_000_000,
                             max_consecutive_rejections=args.max_consecutive_rejections, shadow=args.shadow)
        summary = runner.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
