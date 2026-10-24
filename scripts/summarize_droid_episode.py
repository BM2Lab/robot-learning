"""Timing, acceptance and identity summary of one recorded DROID episode (run_droid_runtime.py output).

Reports clock uncertainty, startup wait, state/camera timing, policy and capture-to-receipt
latency, accepted→applied delay, row-0 lead and row-to-row speed, rejection causes,
faults and revisions. With ``--candidate-limits`` (a franka-runtime physical config with
``acceptance.max_target_lead_rad`` and ``acceptance.max_consecutive_row_speed_rad_s``) it
also counts how many accepted chunks those physical bounds would have rejected, so mock
acceptance rates are not mistaken for physical ones. Standard library only.
"""
import argparse
import json
from pathlib import Path


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values)-1, int(q*(len(values)-1)+0.5))]
    return {"n": len(values), "min": values[0], "median": pick(.5), "p95": pick(.95), "max": values[-1]}


def category(reason):
    for key in ("leads measured", "rad/s", "below table", "workspace box", "outside margin", "latched",
                "deviation", "expired", "Superseded"):
        if key in reason:
            return key
    return reason[:60]


def summarize(directory, candidate=None, row_period_s=1/15):
    events = [json.loads(line) for line in (Path(directory)/"episode.jsonl").read_text().splitlines()]
    session = next((e for e in events if e["kind"] == "session"), {})
    steps = [e for e in events if e["kind"] == "step"]
    receipts = [e["receipt"] for e in events if e["kind"] == "receipt"]
    end = next((e for e in events if e["kind"] == "episode_end"), {})
    accepted_at, delays = {}, []
    for r in receipts:  # drain order: an applied receipt follows its own accepted receipt
        if r["status"] == "accepted":
            accepted_at[r["sequence"]] = r["timestamp_ns"]
        elif r["status"] == "applied" and r["sequence"] in accepted_at:
            delays.append((r["timestamp_ns"]-accepted_at.pop(r["sequence"]))/1e6)
    leads, speeds, capture_to_receipt, would_reject = [], [], [], {"lead": 0, "row_speed": 0}
    for step in steps:
        if "chunk" not in step:
            continue
        rows, q = step["chunk"]["values"], step["state"]["q_rad"]
        lead = [abs(a-b) for a, b in zip(rows[0][:7], q)]
        speed = [max((abs(r2[j]-r1[j])/row_period_s for r1, r2 in zip(rows, rows[1:])), default=0.) for j in range(7)]
        leads.append(max(lead))
        speeds.append(max(speed))
        capture_to_receipt.append((step["client_ns"]-step["camera"]["newest_capture_ns"])/1e6)
        if candidate and step["receipt"]["status"] == "accepted":
            would_reject["lead"] += any(l > b for l, b in zip(lead, candidate["max_target_lead_rad"]))
            would_reject["row_speed"] += any(s > b for s, b in zip(speed, candidate["max_consecutive_row_speed_rad_s"]))
    rejections = {}
    for step in steps:
        reason = step.get("contract_rejection") or (step["receipt"]["reason"] if step["receipt"]["status"] == "rejected" else None)
        if reason:
            rejections[category(reason)] = rejections.get(category(reason), 0)+1
    ready = next((e for e in events if e["kind"] in ("runtime_ready", "runtime_not_ready")), None)
    clock = session.get("clock", {})
    return {
        "outcome": end.get("outcome"), "released": end.get("released"), "fault": end.get("fault"),
        "steps": len(steps), "accepted": sum(s.get("receipt", {}).get("status") == "accepted" for s in steps),
        "chunks_applied": end.get("chunks_applied"), "rejections": rejections,
        "clock_uncertainty_us": clock.get("uncertainty_ns", 0)/1e3 if clock else None,
        "runtime_ready_wait_ms": None if ready is None else ready["wait_ms"],
        "state_camera_skew_ms": stats([s["camera"]["state_camera_skew_ms"] for s in steps]),
        "policy_roundtrip_ms": stats([s["policy"].get("roundtrip_ms") for s in steps]),
        "policy_infer_ms": stats([(s["policy"].get("policy_timing") or {}).get("infer_ms") for s in steps]),
        "capture_to_receipt_ms": stats(capture_to_receipt),
        "accepted_to_applied_ms": stats(delays),
        "row0_lead_rad": stats(leads), "row_speed_rad_s": stats(speeds),
        "candidate_would_reject_accepted": would_reject if candidate else None,
        "revisions": {"runtime": session.get("hello", {}).get("versions"),
                      "contract_sha256": session.get("hello", {}).get("contract_sha256"),
                      "policy": {k: session.get("policy", {}).get(k) for k in ("openpi", "checkpoint", "server_verified")},
                      "sources": session.get("sources"),
                      "cameras": {k: {f: v.get("device", {}).get(f) for f in ("name", "serial_number", "firmware_version")}
                                  for k, v in session.get("cameras", {}).items()}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--candidate-limits", type=Path, help="franka-runtime physical config JSON")
    args = parser.parse_args()
    candidate = None
    if args.candidate_limits:
        candidate = json.loads(args.candidate_limits.read_text())["acceptance"]
        if candidate.get("max_target_lead_rad") is None or candidate.get("max_consecutive_row_speed_rad_s") is None:
            parser.error("Candidate config lacks acceptance lead/row-speed values")
    print(json.dumps(summarize(args.episode_dir, candidate), indent=2))


if __name__ == "__main__":
    main()
