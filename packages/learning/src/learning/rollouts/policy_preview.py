"""Latest policy-input images on disk for a live viewer (scripts/view_policy_inputs.py).

The runner offers each observation packet; a background thread writes only the newest one,
so a slow disk never delays the control loop. Files are replaced atomically: one PNG per
camera key (exactly the arrays sent to the policy) and ``latest.json`` with step metadata.
"""
import json
import os
from pathlib import Path
import threading
import time


def camera_file(key):
    return key.rsplit("/", 1)[-1]+".png"


class PolicyPreview:
    def __init__(self, directory, clock=time.monotonic_ns):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._cv = threading.Condition()
        self._latest = None
        self._closing = False
        self.written = 0
        self._thread = threading.Thread(target=self._run, name="policy-preview", daemon=True)
        self._thread.start()

    def offer(self, step, packet):
        """Keep only the newest packet; never blocks on disk."""
        with self._cv:
            self._latest = (step, packet, self.clock())
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._latest is None and not self._closing:
                    self._cv.wait()
                if self._latest is None:
                    return
                item, self._latest = self._latest, None
            try:
                self._write(*item)
            except Exception:  # a preview must never affect the episode
                pass

    def _write(self, step, packet, offered_ns):
        import numpy as np
        from PIL import Image
        cameras = []
        for image in packet.images:
            array = np.frombuffer(image.decode(), dtype=np.uint8).reshape(image.height, image.width, 3)
            name = camera_file(image.key)
            tmp = self.directory/(name+".tmp")
            Image.fromarray(array).save(tmp, format="PNG")
            os.replace(tmp, self.directory/name)
            cameras.append({"key": image.key, "file": name, "height": image.height, "width": image.width})
        meta = {"step": step, "episode_id": packet.context.episode_id, "instruction": packet.instruction,
                "observation_ns": packet.context.timestamp_ns, "offered_ns": offered_ns, "cameras": cameras,
                "state": list(packet.state)}
        tmp = self.directory/"latest.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, self.directory/"latest.json")
        self.written += 1

    def close(self, timeout_s=1.):
        with self._cv:
            self._closing = True
            self._cv.notify()
        self._thread.join(timeout_s)
