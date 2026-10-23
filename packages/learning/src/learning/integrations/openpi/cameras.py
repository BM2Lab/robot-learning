"""Route live camera frames into a native DROID ObservationPacket.

Readers are duck-typed: ``latest(max_age_ns)`` returning a frame with ``image``
(uint8 RGB HWC), ``optical_frame``, ``sequence``, ``capture_ns`` and
``sensor_monotonic_ns`` (e.g. BM2Lab sensors LatestFrameReader). learning does not
depend on the sensors distribution. Resizing uses upstream openpi resize_with_pad.
"""
from contracts.samples import ObservationPacket, RGBSample
from .droid import encode_rgb


class DroidCameraRouter:
    def __init__(self, spec, readers, *, max_age_ns, max_skew_ns):
        expected = {c.key for c in spec.observation.cameras}
        if set(readers) != expected:
            raise ValueError(f"Supply exactly the contract cameras {sorted(expected)}")
        for value, field in ((max_age_ns, "max_age_ns"), (max_skew_ns, "max_skew_ns")):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        self.spec, self.readers = spec, dict(readers)
        self.max_age_ns, self.max_skew_ns = max_age_ns, max_skew_ns

    def capture(self):
        """Fresh, mutually synchronous frames keyed by contract camera key, plus a timing audit."""
        frames = {}
        for camera in self.spec.observation.cameras:
            frame = self.readers[camera.key].latest(self.max_age_ns)
            if frame.optical_frame != camera.optical_frame:
                raise ValueError(f"{camera.key} is routed to {frame.optical_frame}, expected {camera.optical_frame}")
            frames[camera.key] = frame
        stamps = [f.capture_ns for f in frames.values()]
        skew = max(stamps)-min(stamps)
        if skew > self.max_skew_ns:
            raise ValueError(f"Camera capture skew {skew/1e6:.1f} ms exceeds bound")
        sensor = [f.sensor_monotonic_ns for f in frames.values()]
        audit = {"capture_skew_ms": skew/1e6, "newest_capture_ns": max(stamps),
                 "sensor_skew_ms": None if None in sensor else (max(sensor)-min(sensor))/1e6,
                 "sequences": {k: f.sequence for k, f in frames.items()}}
        return frames, audit


def droid_packet(spec, context, joint_position, gripper_width_m, frames, instruction):
    """Validated packet: measured joints, Franka Hand width mapped to closure, resized images."""
    from openpi_client.image_tools import resize_with_pad
    state = (*joint_position, spec.gripper.closure_from_width(gripper_width_m))
    images = []
    for camera in spec.observation.cameras:
        frame = frames[camera.key]
        image = resize_with_pad(frame.image, camera.height, camera.width)
        images.append(RGBSample(camera.key, camera.optical_frame, camera.height, camera.width,
                                frame.sequence, context, encode_rgb(image)))
    packet = ObservationPacket(spec.contract_id, context, state, tuple(images), instruction)
    packet.validate_for(spec)
    return packet
