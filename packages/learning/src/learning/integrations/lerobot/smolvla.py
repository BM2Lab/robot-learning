"""Checkpoint-native inference ONLY. No robot action mapping or simulator imports."""
from importlib.metadata import version
import time

from .snapshots import prepare_snapshots


class SmolVLAInference:
    def __init__(self, pin, cache_dir, *, device="cuda", download=False):
        if version("lerobot") != pin["lerobot"]["version"]:
            raise RuntimeError("Unsupported LeRobot version for this model manifest")
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (PolicyProcessorPipeline, policy_action_to_transition,
                                      transition_to_policy_action)
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        self.pin, self.device = pin, device
        self.snapshots = prepare_snapshots(pin, cache_dir, download=download)
        config = SmolVLAConfig.from_pretrained(self.snapshots.policy, local_files_only=True)
        config.device = device
        # Full policy weights contain the VLM. Instantiate architecture from the
        # pinned LOCAL companion, then strictly restore the policy checkpoint.
        config.load_vlm_weights = False
        config.vlm_model_name = str(self.snapshots.companion)
        self.policy = SmolVLAPolicy.from_pretrained(self.snapshots.policy, config=config,
                                                   local_files_only=True, strict=True)
        self.pre = PolicyProcessorPipeline.from_pretrained(
            self.snapshots.policy, config_filename="policy_preprocessor.json", local_files_only=True,
            overrides={"tokenizer_processor": {"tokenizer_name": str(self.snapshots.companion)},
                       "device_processor": {"device": device}})
        self.post = PolicyProcessorPipeline.from_pretrained(
            self.snapshots.policy, config_filename="policy_postprocessor.json", local_files_only=True,
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        self.reset()

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def predict_native(self, state, images, instruction, *, seed=0):
        """Return unnormalized checkpoint-native values, NOT Cartesian increments."""
        import numpy as np
        import torch
        features = self.pin["checkpoint_features"]
        state = np.asarray(state, dtype=np.float32)
        if list(state.shape) != features["state_shape"] or not np.isfinite(state).all():
            raise ValueError("Wrong checkpoint state dimensions or non-finite state")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A nonempty language instruction is required")
        if set(images) != set(features["cameras"]):
            raise ValueError("Supply all and only the checkpoint's named camera views")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        batch = {"observation.state": torch.from_numpy(state.copy()), "task": instruction}
        channels, height, width = features["camera_shape_chw"]
        for name, image in images.items():
            if not isinstance(image, np.ndarray) or image.shape != (height, width, channels) or image.dtype != np.uint8:
                raise ValueError("Camera data must be uint8 RGB HWC with declared dimensions")
            batch[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.)
        chunk_size = self.policy.config.chunk_size
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn((1, chunk_size, self.policy.config.max_action_dim),
                            generator=generator, device=self.device)
        if self.device != "cpu":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(self.pre(batch), noise=noise)
            actions = self.post(normalized)
        if self.device != "cpu":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        expected = (1, chunk_size, features["action_shape"][0])
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
            raise RuntimeError("Invalid pretrained action chunk")
        return actions.detach().cpu().numpy(), elapsed
