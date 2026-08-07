"""供强化学习使用的 Gymnasium 接口。

该文件只负责把观测和动作转换成固定长度向量，并计算训练奖励。线缆扰动、
接触、弹性夹持代理和成功判定仍完全由 CableGraspEnv 决定。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import mujoco
import numpy as np

from cable_grasp_env import CableGraspEnv, EnvConfig, rotation_to_quat


@dataclass
class RLConfig:
    """RL接口参数，不改变底层线缆物理。"""

    joint_delta_scale: float = 0.08  # 每个50 Hz动作允许改变的关节目标，单位rad
    target_velocity_scale: float = 1.0
    arm_velocity_scale: float = 3.0
    finger_velocity_scale: float = 0.5
    hand_linear_velocity_scale: float = 1.0
    hand_angular_velocity_scale: float = 4.0


class RLCableGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Panda动态线缆抓取的连续控制环境。

    观测不包含完整线缆形状，只给策略目标段状态与机器人本体状态。动作不经过
    脚本状态机：策略可在任意时刻移动任意关节并开合夹爪。
    """

    metadata = {"render_modes": []}
    # Menagerie Panda主指垫中心：手指根部0.0584 m + 指垫局部0.0445 m。
    PAD_CENTER_LOCAL = np.array([0.0, 0.0, 0.1029])

    OBSERVATION_NAMES = (
        "target_pos_x", "target_pos_y", "target_pos_z",
        "target_vel_x", "target_vel_y", "target_vel_z",
        "target_relative_x", "target_relative_y", "target_relative_z",
        *(f"arm_qpos_{i}" for i in range(1, 8)),
        *(f"arm_qvel_{i}" for i in range(1, 8)),
        "left_finger_qpos", "right_finger_qpos",
        "left_finger_qvel", "right_finger_qvel",
        "hand_pos_x", "hand_pos_y", "hand_pos_z",
        "hand_quat_w", "hand_quat_x", "hand_quat_y", "hand_quat_z",
        "hand_linear_vel_x", "hand_linear_vel_y", "hand_linear_vel_z",
        "hand_angular_vel_x", "hand_angular_vel_y", "hand_angular_vel_z",
    )

    def __init__(
        self,
        *,
        seed: int = 20260804,
        disturbance_strength: float = 1.5,
        episode_seconds: float = 28.0,
        rl_config: RLConfig | None = None,
    ):
        super().__init__()
        self.rl_config = rl_config or RLConfig()
        self.base_env = CableGraspEnv(EnvConfig(
            seed=seed,
            disturbance_strength=disturbance_strength,
            episode_seconds=episode_seconds,
        ))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(len(self.OBSERVATION_NAMES),), dtype=np.float32
        )
        self._previous_distance = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0

        joint_range = self.base_env.model.jnt_range[self.base_env.arm_joint_ids]
        self._arm_center = joint_range.mean(axis=1)
        self._arm_half_range = np.maximum(0.5 * np.ptp(joint_range, axis=1), 1e-6)

    @property
    def model(self) -> mujoco.MjModel:
        return self.base_env.model

    @property
    def data(self) -> mujoco.MjData:
        return self.base_env.data

    def _hand_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_env.hand_id,
            velocity,
            0,
        )
        return velocity[3:].copy(), velocity[:3].copy()

    def _grasp_center_position(self) -> np.ndarray:
        """返回两块真实主指垫之间的中心，而不是前方的虚拟IK控制点。"""
        rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        return (
            self.data.xpos[self.base_env.hand_id]
            + rotation @ self.PAD_CENTER_LOCAL
        )

    def _observation(self) -> np.ndarray:
        """返回经过固定尺度归一化的目标状态和机器人本体状态。"""
        target_position = self.base_env.target_position()
        target_velocity = self.base_env.target_velocity()
        hand_position = self._grasp_center_position()
        hand_rotation = self.data.xmat[self.base_env.hand_id].reshape(3, 3)
        hand_quat = rotation_to_quat(hand_rotation)
        hand_linear_velocity, hand_angular_velocity = self._hand_velocity()

        arm_qpos = self.data.qpos[self.base_env.arm_qpos_adr]
        arm_qvel = self.data.qvel[self.base_env.arm_dof_adr]
        finger_qpos = self.data.qpos[self.base_env.finger_qpos_adr]
        finger_dof_adr = self.model.jnt_dofadr[self.base_env.finger_joint_ids]
        finger_qvel = self.data.qvel[finger_dof_adr]

        # 归一化只改变网络输入尺度，不改变MuJoCo中的物理状态。
        target_position_normalized = (
            target_position - np.array([0.55, 0.0, 0.20])
        ) / np.array([0.40, 0.50, 0.40])
        target_relative = (target_position - hand_position) / 0.50
        hand_position_normalized = (
            hand_position - np.array([0.55, 0.0, 0.25])
        ) / np.array([0.45, 0.55, 0.45])

        observation = np.concatenate([
            target_position_normalized,
            target_velocity / self.rl_config.target_velocity_scale,
            target_relative,
            (arm_qpos - self._arm_center) / self._arm_half_range,
            arm_qvel / self.rl_config.arm_velocity_scale,
            2.0 * finger_qpos / 0.04 - 1.0,
            finger_qvel / self.rl_config.finger_velocity_scale,
            hand_position_normalized,
            hand_quat,
            hand_linear_velocity / self.rl_config.hand_linear_velocity_scale,
            hand_angular_velocity / self.rl_config.hand_angular_velocity_scale,
        ]).astype(np.float32)
        return np.clip(observation, -10.0, 10.0)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """把[-1,1]策略动作映射为MuJoCo合法的执行器命令。"""
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if normalized.shape != (8,):
            raise ValueError(f"Expected RL action shape (8,), got {normalized.shape}")

        current_qpos = self.data.qpos[self.base_env.arm_qpos_adr]
        target_qpos = current_qpos + normalized[:7] * self.rl_config.joint_delta_scale
        joint_range = self.model.jnt_range[self.base_env.arm_joint_ids]

        mujoco_action = np.empty(8)
        mujoco_action[:7] = np.clip(target_qpos, joint_range[:, 0], joint_range[:, 1])
        # -1为完全闭合，+1为完全张开；中间值是连续夹爪目标，而不是策略外规则。
        gripper_range = self.model.actuator_ctrlrange[7]
        mujoco_action[7] = gripper_range[0] + 0.5 * (
            normalized[7] + 1.0
        ) * (gripper_range[1] - gripper_range[0])
        return mujoco_action

    def _reward(
        self, action: np.ndarray, success: bool, info: dict
    ) -> tuple[float, dict[str, float]]:
        """连续奖励只描述任务进展，不修改环境或执行器动作。"""
        distance = float(np.linalg.norm(
            self.base_env.target_position() - self._grasp_center_position()
        ))
        progress = self._previous_distance - distance
        self._previous_distance = distance

        pairs = self.base_env._finger_contact_pairs()
        contacting_fingers = len({finger for _, finger in pairs})
        grasped = float(self.base_env.grasp_confirmed)
        if self.base_env.grasp_state is None:
            grasped_height = 0.0
        else:
            body_id = self.base_env.grasp_state.body_id
            grasped_height = float(np.clip(
                (self.data.xpos[body_id, 2] - 0.045) / 0.20, 0.0, 1.0
            ))

        components = {
            "reward_progress": 4.0 * progress,
            "reward_reach": 0.02 * math.exp(-8.0 * distance),
            "reward_contact": 0.10 * contacting_fingers / 2.0,
            "reward_grasp": 0.25 * grasped,
            "reward_lift": 0.50 * grasped * grasped_height,
            "reward_cable_lift": 0.20 * grasped * float(info["lifted_fraction"]),
            "reward_success": 10.0 * float(success),
            "reward_action_cost": -0.001 * float(np.mean(np.square(action))),
        }
        return float(sum(components.values())), components

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            # 底层环境只使用自己的Generator；显式同步后可复现并行环境中的每一轮。
            self.base_env.rng = np.random.default_rng(seed)
        _, info = self.base_env.reset(randomize=True)
        self._previous_distance = float(np.linalg.norm(
            self.base_env.target_position() - self._grasp_center_position()
        ))
        self._episode_return = 0.0
        self._episode_steps = 0
        return self._observation(), self._augment_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        normalized_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        mujoco_action = self._convert_action(normalized_action)
        _, _, success, truncated, info = self.base_env.step(mujoco_action)
        reward, reward_components = self._reward(normalized_action, success, info)

        self._episode_return += reward
        self._episode_steps += 1
        info = self._augment_info(info)
        info.update(reward_components)
        info["episode_return"] = self._episode_return
        info["episode_steps"] = self._episode_steps
        info["mujoco_action"] = mujoco_action.copy()
        return self._observation(), reward, bool(success), bool(truncated), info

    def _augment_info(self, info: dict) -> dict:
        result = dict(info)
        result["target_distance"] = float(np.linalg.norm(
            self.base_env.target_position() - self._grasp_center_position()
        ))
        result["raw_finger_contacts"] = len(self.base_env.finger_contacts())
        result["unique_contacting_fingers"] = len({
            finger for _, finger in self.base_env._finger_contact_pairs()
        })
        result["ever_grasped"] = self.base_env.last_grasped_body_id is not None
        return result

    def close(self) -> None:
        # 原生MuJoCo数据没有额外线程或窗口；保留Gymnasium标准接口。
        return None
