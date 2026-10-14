"""Export complete successful V3 archives to LeRobot and verify every saved frame."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/fr3_visual_reach_v3")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from learning.integrations.lerobot.demonstrations import export_demonstrations
    report = export_demonstrations(args.source, args.output_dir, args.repo_id, args.smoke)
    print(json.dumps({key: report[key] for key in ("passed", "checks", "frames", "episodes", "splits", "runtime")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
