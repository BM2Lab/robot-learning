"""L0 native MuJoCo reset/render/neutral-step check; no model download."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-dir", required=True, help="Explicit POSIX cache/config directory")
    parser.add_argument("--download", action="store_true", help="Allow pinned asset downloads")
    parser.add_argument("--smoke", action="store_true", help="Bounded /tmp artifacts only")
    args = parser.parse_args()
    from learning.evaluation.libero import run_evaluation
    print(json.dumps(run_evaluation(args.config, args.output_dir, args.runtime_dir,
                                  download=args.download, smoke=args.smoke, verify_only=True), indent=2))


if __name__ == "__main__":
    main()
