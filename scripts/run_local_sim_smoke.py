"""Verify V2 local IPC from the learning environment; no ML model is loaded."""
import argparse
import json
from pathlib import Path
from contracts.policies import PolicyIOSpec
from learning.rollouts.local_smoke import verify_local_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 20 or not 1 <= args.timeout <= 60:
        parser.error("Use 1..20 cycles and 1..60 seconds RPC timeout")
    spec = PolicyIOSpec.from_json(args.contract.read_text())
    result = verify_local_session(args.socket, spec, args.cycles, args.timeout)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
