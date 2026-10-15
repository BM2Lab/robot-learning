"""L1/L2 pinned pretrained SmolVLA evaluation in native LIBERO, not Isaac."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--download", action="store_true", help="Allow pinned model/asset downloads")
    parser.add_argument("--smoke", action="store_true", help="At most two episodes, explicit /tmp output")
    args = parser.parse_args()
    from learning.evaluation.libero import run_evaluation
    print(json.dumps(run_evaluation(args.config, args.output_dir, args.runtime_dir,
                                  download=args.download, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
