"""Paired inference on lossless native/Isaac images at identical measured poses."""
import argparse
import json
from pathlib import Path

from learning.data.storage import validate_artifact_path
from learning.integrations.libero.config import load_pin
from learning.integrations.libero.preflight import check_runtime, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sequence", "renders", "pin", "runtime", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    sequence = json.loads((a.sequence/"poses.json").read_text())
    renders = json.loads((a.renders/"report.json").read_text())
    if not sequence["complete"] or not renders["complete"] or renders["sequence_sha256"] != file_hash(a.sequence/"poses.json"):
        raise ValueError("Expected a qualified matched-pose rendering")
    pin = load_pin(a.pin)
    if pin != sequence["pin"]:
        raise ValueError("Policy pin differs")
    out = validate_artifact_path(a.output, 16*1024**2, smoke=a.smoke)
    runtime, versions = check_runtime(pin,a.runtime)
    import numpy as np
    from PIL import Image
    from learning.integrations.lerobot.smolvla import SmolVLAInference
    out.mkdir(parents=True)
    report = {"complete": False, "scope": "Matched poses; common native fresh state/instruction/noise; no Isaac physics",
              "sequence_sha256": file_hash(a.sequence/"poses.json"), "renders_sha256": file_hash(a.renders/"report.json"),
              "runner_sha256": file_hash(__file__), "runtime": versions, "pin": pin, "frames": []}
    try:
        model = SmolVLAInference(pin,runtime/"huggingface/hub",device="cuda")
        for frame in sequence["frames"]:
            predictions = {}
            row = {"step": frame["step"], "noise_seed": 42+frame["step"], "image_sha256": {}}
            for domain, directory in (("native",a.sequence),("isaac",a.renders)):
                images = {}
                for key, short in (("observation.images.image","agentview"),("observation.images.image2","eye_in_hand")):
                    path = directory/f"{domain}_{frame['step']}_{short}.png"
                    if domain == "native" and file_hash(path) != sequence["files_sha256"][path.name]:
                        raise ValueError("Native image changed")
                    images[key] = np.asarray(Image.open(path)).copy()
                    row["image_sha256"][path.name] = file_hash(path)
                model.reset()
                chunk, _ = model.predict_native(np.asarray(frame["state"],dtype=np.float32), images,
                    frame["instruction"],seed=row["noise_seed"])
                predictions[domain] = chunk[0]
                row[domain+"_first_action"] = chunk[0,0].tolist()
            delta = predictions["isaac"]-predictions["native"]
            row["chunk_rms_difference"] = float(np.sqrt(np.mean(delta**2)))
            row["first_action_max_difference"] = float(np.abs(delta[0]).max())
            n, i = predictions["native"][0,:3], predictions["isaac"][0,:3]
            denominator = float(np.linalg.norm(n)*np.linalg.norm(i))
            row["first_translation_cosine"] = float(n@i/denominator) if denominator > 1e-10 else None
            report["frames"].append(row)
            print(f"Matched pose {frame['step']}: chunk RMS={row['chunk_rms_difference']:.4f}, translation cosine={row['first_translation_cosine']}",flush=True)
        report["complete"] = True
    finally:
        (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
