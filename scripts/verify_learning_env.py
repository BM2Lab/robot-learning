"""Verify the pinned inference environment without loading weights or robots."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cuda", action="store_true")
    parser.add_argument("--check-model-config", action="store_true",
                        help="Download only pinned config.json into the configured HF cache")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pin = json.loads((root / "configs/policies/smolvla_base.json").read_text())
    expected = {"lerobot": pin["lerobot"]["version"],
                "torch": pin["runtime_profile"]["torch"],
                "torchvision": pin["runtime_profile"]["torchvision"]}
    versions = {name: importlib.metadata.version(name) for name in
                (*expected, "transformers", "huggingface-hub", "numpy")}
    for name, version in expected.items():
        if versions[name] != version:
            raise RuntimeError(f"Wrong {name}: {versions[name]} != {version}")
    if sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the isolated Python 3.12 learning environment")
    if importlib.util.find_spec("isaaclab") is not None:
        raise RuntimeError("Isaac Lab leaked into the standalone learning environment")

    import contracts
    import learning
    import torch
    import torchvision
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    result = {"python": sys.version.split()[0], "environment": sys.prefix,
              "versions": versions, "smolvla_import": SmolVLAPolicy.__name__,
              "cuda_build": torch.version.cuda, "weights_loaded": False,
              "hardware_access": False}
    if args.check_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check GPU access/driver without changing Isaac")
        x = torch.ones((16, 16), device="cuda")
        if not torch.isfinite(x @ x).all().item():
            raise RuntimeError("CUDA smoke calculation failed")
        result["gpu"] = torch.cuda.get_device_name(0)
    if args.check_model_config:
        from huggingface_hub import hf_hub_download
        from hashlib import sha256
        policy = pin["policy"]
        path = hf_hub_download(policy["repo_id"], "config.json", revision=policy["revision"])
        digest = sha256(Path(path).read_bytes()).hexdigest()
        if digest != policy["config_sha256"]:
            raise RuntimeError("Pinned checkpoint configuration hash mismatch")
        config = json.loads(Path(path).read_text())
        assert config["type"] == pin["policy_type"]
        assert config["input_features"]["observation.state"]["shape"] == pin["checkpoint_features"]["state_shape"]
        assert config["output_features"]["action"]["shape"] == pin["checkpoint_features"]["action_shape"]
        # Exercise the selected release's config parser, not only raw JSON.
        parsed = SmolVLAConfig.from_pretrained(Path(path).parent)
        assert parsed.chunk_size == pin["checkpoint_features"]["chunk_size"]
        result["policy_revision"] = policy["revision"]
        result["config_sha256"] = digest
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
