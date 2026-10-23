"""Frozen-input P3 image/state swaps; inference only, no physics or success claim."""
import argparse
import itertools
import json
from pathlib import Path
import time
import traceback

from learning.data.storage import validate_artifact_path
from learning.integrations.libero.config import load_pin
from learning.integrations.libero.preflight import check_runtime, file_hash


def conditions():
    return tuple(itertools.product(("native", "isaac"), repeat=3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Qualified P2 paired lossless PNG directory")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--isaac-run", type=Path, required=True)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    pin = load_pin(args.pin)
    reference = json.loads(args.reference.read_text())
    baseline = json.loads((args.isaac_run/"manifest.json").read_text())
    qualification = json.loads((args.images/"report.json").read_text())
    if (not qualification["passed"] or not baseline["complete"] or baseline["pin"] != pin
            or file_hash(args.reference) != baseline["endpoint"]["receipt"]["visual_reference_sha256"]
            or qualification["receipt"]["visual_reference_sha256"] != file_hash(args.reference)):
        raise ValueError("Expected matching qualified P2 images/reference and completed pinned P3 run")
    output = validate_artifact_path(args.output, 16*1024**2, smoke=args.smoke)
    runtime, versions = check_runtime(pin, args.runtime)
    output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "scope": "Frozen reset input causal swaps; no physics or task-success measurement",
              "pin": pin, "runtime_versions": versions, "noise_seeds": [42, 43, 44],
              "condition_order": ["state", "agentview", "wrist"],
              "runner_sha256": file_hash(__file__), "reference_sha256": file_hash(args.reference),
              "qualification_sha256": file_hash(args.images/"report.json"), "rows": []}
    start = time.perf_counter()
    try:
        import numpy as np
        from PIL import Image
        from learning.integrations.lerobot.smolvla import SmolVLAInference
        model = SmolVLAInference(pin, runtime/"huggingface/hub", device="cuda")
        images = {}
        report["image_sha256"] = {}
        for domain, view in itertools.product(("native", "isaac"), ("agentview", "eye_in_hand")):
            path = args.images/f"{domain}_{view}.png"
            image = np.asarray(Image.open(path))
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError("Expected lossless processed 256x256 RGB PNGs")
            images[domain, view] = image.copy()
            report["image_sha256"][path.name] = file_hash(path)
        isaac_rows = json.loads((args.isaac_run/"steps.json").read_text())
        states = {"native": np.asarray(reference["visual"]["state"], dtype=np.float32),
                  "isaac": np.asarray(isaac_rows[0]["source_state"], dtype=np.float32)}
        report["states"] = {k: v.tolist() for k, v in states.items()}
        instruction = reference["environment"]["instruction"]
        report["instruction"] = instruction
        chunks = {}
        for state_domain, agent_domain, wrist_domain in conditions():
            key = f"state_{state_domain}__agent_{agent_domain}__wrist_{wrist_domain}"
            for seed in report["noise_seeds"]:
                model.reset()  # No cross-condition observation/history leakage.
                actions, latency = model.predict_native(states[state_domain], {
                    "observation.images.image": images[agent_domain, "agentview"],
                    "observation.images.image2": images[wrist_domain, "eye_in_hand"]}, instruction, seed=seed)
                chunks[key, seed] = actions[0].copy()
                report["rows"].append({"condition": key, "seed": seed,
                    "raw_chunk": actions[0].tolist(), "inference_seconds": latency})
            print(f"Frozen-input condition complete: {key}", flush=True)
        native_key = "state_native__agent_native__wrist_native"
        for row in report["rows"]:
            a = chunks[row["condition"], row["seed"]]
            b = chunks[native_key, row["seed"]]
            row["first_action_max_abs_vs_native"] = float(np.abs(a[0]-b[0]).max())
            row["chunk_rms_vs_native"] = float(np.sqrt(np.mean((a-b)**2)))
            row["first_translation_l2_vs_native"] = float(np.linalg.norm(a[0, :3]-b[0, :3]))
            row["first_action_bounded"] = a[0].clip(-1, 1).tolist()
        model.reset()
        repeat, _ = model.predict_native(states["native"], {
            "observation.images.image": images["native", "agentview"],
            "observation.images.image2": images["native", "eye_in_hand"]}, instruction, seed=42)
        report["repeat_same_input_max_abs"] = float(np.abs(repeat[0]-chunks[native_key, 42]).max())
        if report["repeat_same_input_max_abs"] > 1e-5:
            raise RuntimeError("Frozen input repeat is not reproducible")
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.perf_counter()-start
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Frozen-input artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
