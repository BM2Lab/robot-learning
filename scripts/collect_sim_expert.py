"""Record V3 expert demonstrations through an opt-in V2 simulator endpoint."""
import argparse
import json
from pathlib import Path
from contracts.policies import PolicyIOSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="Explicit bounded /tmp verification, not persistent collection")
    args = parser.parse_args()
    from learning.rollouts.collect_expert import collect_expert
    report = collect_expert(args.socket, PolicyIOSpec.from_json(args.contract.read_text()), args.output_dir,
        args.episodes, args.seed, args.max_steps, args.smoke)
    print(json.dumps({"complete": report["complete"], "all_successful": report["all_successful"],
        "episodes": len(report["episodes"]), "frames": sum(e["frames"] for e in report["episodes"]),
        "output": str(args.output_dir)}, indent=2), flush=True)
    return 0 if report["all_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
