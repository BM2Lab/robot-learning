"""Bounded P3 pretrained rollout through the paused P2 socket (not a benchmark)."""
import argparse
import json
from pathlib import Path
import time
import traceback

from learning.data.storage import validate_artifact_path
from learning.integrations.libero.config import load_pin
from learning.integrations.libero.preflight import check_runtime, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    output = validate_artifact_path(args.output, 64*1024**2, smoke=args.smoke)
    pin = load_pin(args.pin)
    runtime, versions = check_runtime(pin, args.runtime)
    output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "task_success": None,
              "scope": "P3 fixed-fixture diagnostic; task predicate unqualified",
              "seed": args.seed, "noise_seed_schedule": "seed + policy_step",
              "execute_steps_per_inference": 1, "video_fps_simulation": 20,
              "runtime_versions": versions, "pin": pin,
              "runner_sha256": file_hash(__file__), "steps": 0}
    writer = None
    rows = []
    start = time.perf_counter()
    try:
        import imageio.v2 as imageio
        import numpy as np
        from learning.integrations.isaaclab.libero import (
            LiberoSimulatorClient, native_features, bounded_chunk)
        from learning.integrations.lerobot.smolvla import SmolVLAInference
        # Connect before loading weights so the waiting endpoint's receipt is saved.
        with LiberoSimulatorClient(args.socket, timeout_s=300) as client:
            report["endpoint"] = client.receipt
            (output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
            model = SmolVLAInference(pin, runtime/"huggingface/hub", device=args.device)
            model.reset()
            packet = client.reset(seed=args.seed)
            writer = imageio.get_writer(output/"rollout.mp4", fps=20)

            def record_frame(observation):
                features = native_features(observation)
                writer.append_data(np.concatenate([features["images"][key] for key in
                    ("observation.images.image", "observation.images.image2")], axis=1))
                return features

            features = record_frame(packet)
            for step in range(280):
                raw, latency = model.predict_native(**features, seed=args.seed+step)
                chunk, audit = bounded_chunk(packet, raw, execute_steps=1)
                source = packet
                client.submit(chunk)
                packet, terminal, status = client.advance(source.context)
                rows.append({"source_context": source.context.to_dict(),
                             "source_state": list(source.state),
                             "noise_seed": args.seed+step, "raw_action": raw[0, 0].tolist(),
                             "applied_action": list(chunk.values[0]), "clipping": audit,
                             "inference_seconds": latency, "next_state": list(packet.state),
                             "terminal": terminal})
                features = record_frame(packet)
                report["steps"] = len(rows)
                if len(rows) % 20 == 0:
                    print(f"P3: {len(rows)}/280 steps; inference {latency:.3f}s", flush=True)
                if terminal is not None:
                    report.update(complete=True, terminal=terminal, final_status=status)
                    break
            if not report["complete"]:
                raise RuntimeError("Endpoint did not terminate within the bounded horizon")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        report["wall_seconds"] = time.perf_counter()-start
        report["video_frames"] = len(rows)+1 if writer is not None else 0
        if rows:
            report["mean_inference_seconds"] = sum(r["inference_seconds"] for r in rows)/len(rows)
            report["clipped_components"] = sum(r["clipping"]["components_clipped"] for r in rows)
        (output/"steps.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
        (output/"manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Diagnostic artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
