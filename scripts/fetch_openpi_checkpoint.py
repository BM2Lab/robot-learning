"""Fetch the pinned openpi checkpoint, train-config assets and tokenizer into an explicit OPENPI_DATA_HOME, then verify them.

Run in the openpi server environment. Downloads use openpi's own maybe_download, so
the cache layout is exactly what the server reads. ``--report-only`` prints the found
digests without requiring a match (for recording a new pin).
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-pin", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, required=True, help="OPENPI_DATA_HOME cache directory")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.data_home.is_dir():
        parser.error(f"--data-home must be an existing directory: {args.data_home}")
    os.environ["OPENPI_DATA_HOME"] = str(args.data_home.resolve())
    from openpi.shared import download
    from learning.integrations.openpi.checkpoint import file_sha256, local_path, tree_digest, verify_artifacts
    pin = json.loads(args.policy_pin.read_text())
    download.maybe_download(pin["checkpoint"]["dir"])
    download.maybe_download(pin["tokenizer"]["url"], gs={"token": "anon"})
    download.maybe_download(pin["config_assets"]["url"])
    if args.report_only:
        print(json.dumps({"checkpoint": tree_digest(local_path(args.data_home, pin["checkpoint"]["dir"])),
                          "tokenizer_sha256": file_sha256(local_path(args.data_home, pin["tokenizer"]["url"]))},
                         indent=2))
        return
    paths = verify_artifacts(pin, args.data_home)
    print(json.dumps({"verified": True, **{k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
