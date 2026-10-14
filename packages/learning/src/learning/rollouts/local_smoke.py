"""V2 finite IPC acceptance/reset/deadline checks from the isolated learning process."""
import hashlib
import sys
from contracts.actions import ActionChunk
from learning.integrations.isaaclab.local_client import LocalSimulatorClient, RemoteError
from learning.policies.mock_cartesian import MockCartesianPolicy


def verify_local_session(socket_path, spec, cycles=3, timeout_s=30):
    checks = {}
    policy = MockCartesianPolicy(spec)
    steps = 0
    terminals = []
    with LocalSimulatorClient(socket_path, spec, timeout_s) as client:
        receipt = client.receipt["receipt"]
        packet = client.reset(seed=42)
        first_hashes = {image.key: hashlib.sha256(image.decode()).hexdigest() for image in packet.images}
        checks["two_rgb_cameras"] = len(packet.images) == 2 and all(len(i.decode()) == 256*256*3 for i in packet.images)
        checks["seven_joint_state"] = len(packet.state) == 7
        checks["simulation_clock"] = packet.context.clock == "simulation"
        for _ in range(cycles):
            chunk = policy.predict(packet)
            client.submit(chunk)
            for _ in chunk.values:
                previous = packet
                packet, terminal, status = client.advance(packet.context)
                steps += 1
                if terminal:
                    terminals.append(terminal)
                    checks["automatic_reset_queue_clear"] = status["queued_actions"] == 0
                    break
                checks["policy_step_time"] = checks.get("policy_step_time", True) and packet.context.timestamp_ns-previous.context.timestamp_ns == 50_000_000
                checks["camera_frame_advance"] = checks.get("camera_frame_advance", True) and all(b.frame_index > a.frame_index for a, b in zip(previous.images, packet.images))
            checks["chunk_consumed"] = checks.get("chunk_consumed", True) and client.request("status")["queued_actions"] == 0
        # Leave a chunk partially consumed, then explicitly reset.
        old_chunk = policy.predict(packet)
        client.submit(old_chunk)
        packet, _, _ = client.advance(packet.context)
        packet = client.reset(seed=123)
        checks["reset_identity_and_queue"] = (packet.context.episode_id != old_chunk.observation.episode_id
            and packet.context.step_index == 0 and client.request("status")["queued_actions"] == 0)
        try:
            client.submit(old_chunk)
        except RemoteError:
            checks["old_episode_chunk_rejected"] = True
        else:
            checks["old_episode_chunk_rejected"] = False
        try:
            client.advance(packet.context)
        except RemoteError:
            checks["empty_queue_does_not_step"] = client.observe().context == packet.context
        else:
            checks["empty_queue_does_not_step"] = False
        # Valid at acceptance but expires before the second scheduled row.
        expiry_chunk = ActionChunk(packet.context, spec.action, ((0., 0., 0.),)*2,
            packet.context.timestamp_ns+20_000_000)
        client.submit(expiry_chunk)
        stale_context = packet.context
        packet, _, _ = client.advance(packet.context)
        try:
            client.advance(stale_context)
        except RemoteError:
            checks["stale_step_rejected"] = client.observe().context == packet.context
        else:
            checks["stale_step_rejected"] = False
        try:
            client.advance(packet.context)
        except RemoteError:
            checks["execution_deadline_and_queue_clear"] = (client.observe().context == packet.context
                and client.request("status")["queued_actions"] == 0)
        else:
            checks["execution_deadline_and_queue_clear"] = False
        # A duplicate request id cannot execute a command twice.
        from contracts.wire import read_message, write_message, PROTOCOL_VERSION
        write_message(client.stream, {"protocol": PROTOCOL_VERSION, "id": str(client.counter), "op": "reset", "payload": {}})
        response = read_message(client.stream)
        checks["duplicate_request_rejected"] = response["ok"] is False and client.observe().context == packet.context
        checks["no_isaac_or_ml_imports"] = not any(name in sys.modules for name in ("isaaclab", "isaacsim", "torch", "transformers", "lerobot"))
    return {"passed": all(checks.values()), "checks": checks, "mock_policy_steps": steps,
        "terminal_events": terminals, "initial_image_sha256": first_hashes,
        "contract_id": spec.contract_id, "policy": "mock_cartesian", "pretrained_vla_actions_used": False,
        "simulator_receipt": receipt, "learning_python": sys.executable, "transport": "unix_json_base64_v1"}
