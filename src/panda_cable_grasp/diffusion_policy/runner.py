"""Online receding-horizon execution for the visual Diffusion Policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..dynamicvla.adapter import (
    DynamicVLATaskSpaceAdapter,
    make_dynamicvla_observation,
)
from ..env.environment import CableGraspEnv
from .config import (
    ACTION_HIGH,
    ACTION_LOW,
    STATE_HIGH,
    STATE_LOW,
    denormalize_array,
    normalize_array,
)
from .model import DiffusionPolicy


def _missing_dependency_message() -> str:
    return (
        "Diffusion Policy requires the optional dependencies; install with "
        "`python -m pip install -e \".[diffusion]\"`"
    )


class DiffusionPolicyRunner:
    """Adapt the DynamicVLA observation schema to a learned DP checkpoint.

    The runner deliberately accepts the dictionary produced by
    ``make_dynamicvla_observation``.  It stores a short history, samples a
    future action chunk, executes only ``action_horizon`` commands, and then
    replans.  The decoded task-space command goes through the existing
    DynamicVLATaskSpaceAdapter, so all workspace, quaternion, IK, and velocity
    safeguards remain shared across methods.
    """

    def __init__(
        self,
        env: CableGraspEnv,
        checkpoint: Path,
        *,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        try:
            payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:  # torch < 2.6 compatibility
            payload = torch.load(self.checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"invalid Diffusion Policy checkpoint: {checkpoint}")
        if payload.get("format") != "panda_cable_diffusion_policy_v2":
            raise ValueError(
                "unsupported Diffusion Policy checkpoint format; retrain with "
                "the DynaMimicGen-style v2 implementation"
            )
        config_values = dict(payload.get("config", {}))
        from .config import DiffusionPolicyConfig

        self.config = DiffusionPolicyConfig(**config_values)
        self.model = DiffusionPolicy(self.config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.using_ema = payload.get("ema_model") is not None
        if self.using_ema:
            self.model.load_state_dict(payload["ema_model"])
        self.model.eval()
        self.state_low = np.asarray(payload.get("state_low", STATE_LOW), dtype=np.float32)
        self.state_high = np.asarray(payload.get("state_high", STATE_HIGH), dtype=np.float32)
        self.action_low = np.asarray(payload.get("action_low", ACTION_LOW), dtype=np.float32)
        self.action_high = np.asarray(payload.get("action_high", ACTION_HIGH), dtype=np.float32)
        self.deterministic = bool(deterministic)
        self.adapter = DynamicVLATaskSpaceAdapter(env)
        self._opst_history: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._wrist_history: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._state_history: deque[np.ndarray] = deque(maxlen=self.config.observation_horizon)
        self._action_queue: deque[np.ndarray] = deque()
        self._inference_count = 0
        self._last_model_action = np.full(8, np.nan, dtype=np.float32)
        self._last_observation_index = 0
        self.result = "running"
        self.finished = False
        self._generator: torch.Generator | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset()
        self._opst_history.clear()
        self._wrist_history.clear()
        self._state_history.clear()
        self._action_queue.clear()
        self._inference_count = 0
        self._last_model_action[:] = np.nan
        self._last_observation_index = 0
        self.result = "running"
        self.finished = False
        if seed is None or not self.deterministic:
            self._generator = None
        else:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

    @staticmethod
    def _state_from_observation(observation: dict[str, Any]) -> np.ndarray:
        state = observation["observation.state"]
        position = np.asarray(state["end_effector"]["pos"], dtype=np.float32)
        quaternion = np.asarray(state["end_effector"]["quat"], dtype=np.float32)
        if position.shape == (1, 3):
            position = position[0]
        if quaternion.shape == (1, 4):
            quaternion = quaternion[0]
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("DynamicVLA end-effector state must be shapes (1,3)/(1,4)")
        return np.concatenate((position, quaternion)).astype(np.float32)

    def _append_observation(self, observation: dict[str, Any]) -> None:
        state = self._state_from_observation(observation)
        opst = np.asarray(observation["observation.images.opst_cam"])
        wrist = np.asarray(observation["observation.images.wrist_cam"])
        if opst.ndim == 4:
            opst = opst[0]
        if wrist.ndim == 4:
            wrist = wrist[0]
        if opst.ndim != 3 or wrist.ndim != 3 or opst.shape[-1] != 3 or wrist.shape[-1] != 3:
            raise ValueError("DynamicVLA camera observations must be RGB images")
        opst = cv2.resize(
            opst, (self.config.image_width, self.config.image_height),
            interpolation=cv2.INTER_AREA,
        )
        wrist = cv2.resize(
            wrist, (self.config.image_width, self.config.image_height),
            interpolation=cv2.INTER_AREA,
        )
        self._opst_history.append(opst.astype(np.float32) / 255.0)
        self._wrist_history.append(wrist.astype(np.float32) / 255.0)
        self._state_history.append(normalize_array(state, self.state_low, self.state_high))
        while len(self._opst_history) < self.config.observation_horizon:
            self._opst_history.appendleft(self._opst_history[0].copy())
            self._wrist_history.appendleft(self._wrist_history[0].copy())
            self._state_history.appendleft(self._state_history[0].copy())

    def _sample_action_chunk(self) -> None:
        opst = np.asarray(self._opst_history, dtype=np.float32)
        wrist = np.asarray(self._wrist_history, dtype=np.float32)
        state = np.asarray(self._state_history, dtype=np.float32)
        tensors = {
            "opst_cam": torch.from_numpy(opst.transpose(0, 3, 1, 2))[None].to(self.device),
            "wrist_cam": torch.from_numpy(wrist.transpose(0, 3, 1, 2))[None].to(self.device),
            "state": torch.from_numpy(state)[None].to(self.device),
        }
        normalized_chunk = self.model.sample(
            tensors["opst_cam"], tensors["wrist_cam"], tensors["state"],
            generator=self._generator,
        )[0].detach().cpu().numpy()
        model_chunk = denormalize_array(
            normalized_chunk, self.action_low, self.action_high
        )
        execute_count = min(self.config.action_horizon, len(model_chunk))
        self._action_queue.extend(np.asarray(model_chunk[:execute_count], dtype=np.float32))
        self._inference_count += 1

    def action(self, observation: dict[str, Any] | None = None) -> np.ndarray:
        if observation is None:
            observation = make_dynamicvla_observation(
                self.env, instruction=None, index=self._last_observation_index
            )
        self._append_observation(observation)
        self._last_observation_index += 1
        if not self._action_queue:
            self._sample_action_chunk()
        task_space_action = self._action_queue.popleft()
        self._last_model_action[:] = task_space_action
        self.adapter.set_model_action(task_space_action)
        return self.adapter.action()

    def policy_info(self) -> dict[str, Any]:
        diagnostics = self.adapter.diagnostics()
        return {
            "diffusion_inference_count": self._inference_count,
            "diffusion_action_queue": len(self._action_queue),
            "diffusion_model_action": self._last_model_action.copy(),
            "diffusion_position_clipped": bool(diagnostics["position_clipped"]),
            "diffusion_quaternion_repaired": bool(diagnostics["quaternion_repaired"]),
        }
