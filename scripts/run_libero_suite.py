"""L3 small native LIBERO reference suite; no live GUI or Isaac imports."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--smoke", action="store_true", help="Three tasks × two episodes maximum; /tmp only")
    parser.add_argument("--download", action="store_true", help="Explicitly allow pinned child downloads")
    args = parser.parse_args()
    from learning.evaluation.libero_suite import LiberoSuiteRunner
    result = LiberoSuiteRunner(args.config, args.output_dir, args.runtime_dir,
                              smoke=args.smoke, download=args.download).run()
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
