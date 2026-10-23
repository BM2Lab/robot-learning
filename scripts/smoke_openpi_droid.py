"""Bounded live-camera pi05 DROID inference smoke. NO robot: joint state is a static fixture.

Routes two sensors RealSense readers into validated DROID packets, queries a running
upstream openpi server and checks every returned chunk against the native contract.
Writes a receipt, per-step audits/chunks and the first step's 224x224 policy inputs
into a new output directory. Nothing is sent to any robot runtime.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import uuid

from contracts.droid import DroidIOSpec
from contracts.episodes import SampleContext
from learning.integrations.openpi.cameras import DroidCameraRouter, droid_packet
from learning.integrations.openpi.droid import ChunkRejected, OpenPIDroidClient, load_pin

# Static fixture near the DROID training state mean; NOT measured robot state.
FIXTURE_JOINTS = (0.016, 0.267, -0.017, -2.026, -0.033, 2.345, 0.083)


def write_ppm(path, image_sample):
    with open(path, "wb") as out:
        out.write(f"P6 {image_sample.width} {image_sample.height} 255\n".encode())
        out.write(image_sample.decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy-pin", type=Path, required=True)
    parser.add_argument("--exterior-camera", type=Path, required=True, help="sensors RealSense config")
    parser.add_argument("--wrist-camera", type=Path, required=True, help="sensors RealSense config")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gripper-width-m", type=float, default=0.08)
    parser.add_argument("--max-age-ms", type=float, default=100.)
    parser.add_argument("--max-skew-ms", type=float, default=40.)
    parser.add_argument("--allow-unverified-server", action="store_true",
                        help="admit upstream serve_policy.py, which publishes no identity")
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        parser.error("--steps must be 1..200")
    from sensors.cameras.latest import LatestFrameReader
    from sensors.cameras.realsense import RealSenseCamera, RealSenseConfig
    spec = DroidIOSpec.from_json(args.contract.read_text())
    pin = load_pin(args.policy_pin)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    episode = f"smoke-{uuid.uuid4()}"
    steps, receipt = [], {"robot": "none; static joint fixture", "fixture_joints": FIXTURE_JOINTS,
                          "gripper_width_m": args.gripper_width_m, "instruction": args.instruction}
    with ExitStack() as stack:
        cameras, readers = {}, {}
        for key, path in ((spec.observation.cameras[0].key, args.exterior_camera),
                          (spec.observation.cameras[1].key, args.wrist_camera)):
            camera = stack.enter_context(RealSenseCamera(RealSenseConfig.from_json(path.read_text())))
            cameras[key] = camera.receipt
            readers[key] = stack.enter_context(LatestFrameReader(camera))
        router = DroidCameraRouter(spec, readers, max_age_ns=int(args.max_age_ms*1e6),
                                   max_skew_ns=int(args.max_skew_ms*1e6))
        client = OpenPIDroidClient.connect(spec, pin, args.host, args.port,
                                           allow_unverified_server=args.allow_unverified_server)
        receipt.update(cameras=cameras, policy=client.receipt())
        deadline = time.monotonic()+5
        while True:  # Wait for both cameras' first fresh frames.
            try:
                router.capture()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        for step in range(args.steps):
            frames, camera_audit = router.capture()
            context = SampleContext(episode, step, camera_audit["newest_capture_ns"], "monotonic")
            packet = droid_packet(spec, context, FIXTURE_JOINTS, args.gripper_width_m, frames, args.instruction)
            if step == 0:
                for image in packet.images:
                    write_ppm(args.output_dir/f"{image.optical_frame}_224.ppm", image)
            rows = None
            try:
                chunk, audit = client.infer(packet)
                now = time.monotonic_ns()
                rows = [list(r) for r in chunk.values]
                try:
                    chunk.validate_for(spec, SampleContext(episode, step, now, "monotonic"))
                    accepted = True
                except ValueError as exc:
                    accepted = f"acceptance: {exc}"
            except ChunkRejected as exc:  # Contract rejection is a recorded outcome, not a crash.
                now, audit, accepted = time.monotonic_ns(), exc.audit, f"contract: {exc}"
            steps.append({"step": step, "camera": camera_audit, "policy": audit,
                          "capture_to_chunk_ms": (now-context.timestamp_ns)/1e6, "accepted": accepted,
                          "executable_rows": rows})
    latency = sorted(s["capture_to_chunk_ms"] for s in steps)
    receipt["summary"] = {"steps": len(steps), "accepted": sum(s["accepted"] is True for s in steps),
        "capture_to_chunk_ms": {"min": latency[0], "median": latency[len(latency)//2], "max": latency[-1]},
        "gripper_clipped_total": sum(s["policy"]["gripper_clipped"] for s in steps),
        "rejected": [{"step": s["step"], "reason": s["accepted"]} for s in steps if s["accepted"] is not True],
        "max_joint_deviation_rad": max(s["policy"]["max_joint_deviation_rad"] for s in steps),
        "max_joint_deviation_full_chunk_rad": max(s["policy"]["max_joint_deviation_full_chunk_rad"] for s in steps),
        "policy_infer_ms": sorted(s["policy"]["policy_timing"]["infer_ms"] for s in steps
                                  if s["policy"].get("policy_timing"))[len(steps)//2]}
    (args.output_dir/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (args.output_dir/"steps.json").write_text(json.dumps(steps, indent=1)+"\n")
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
