"""
FK-Mediated Executability Reward Components
============================================

Operationalizes categories (I) and (III) of our executability framework
by mapping decoded joint states through Franka Panda forward kinematics.

  (I)   Configuration feasibility — workspace, singularity, self-collision
  (III) State-observation consistency — multi-step FK chain self-consistency

These components are robot-specific: they encode the Franka Panda kinematic
structure through its URDF, making the reward configuration-aware rather
than relying on uniform joint-space penalties.

The core enabling fact is that the IDM outputs absolute joint positions
(action_t = control_state_{t+1}), allowing direct FK verification without
integration or approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Franka Panda Kinematics Interface
# ═══════════════════════════════════════════════════════════════════


class FrankaKinematicsProtocol:
    """Protocol for Franka Panda forward kinematics (dual-arm)."""

    def ee_pose(
        self, left_joints: torch.Tensor, right_joints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FK for dual arms.

        Args:
            left_joints:  [B, 7] left arm joint angles (rad)
            right_joints: [B, 7] right arm joint angles (rad)

        Returns:
            left_ee:  [B, 7] xyz + quat
            right_ee: [B, 7] xyz + quat
        """
        ...

    def jacobian_condition(self, joints_7dof: torch.Tensor) -> torch.Tensor:
        """Singularity measure via Jacobian condition number.

        Args:
            joints_7dof: [B, 7]

        Returns:
            cond: [B] condition number (high → near singularity)
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# Backend A: pytorch_kinematics + Franka URDF (recommended)
# ═══════════════════════════════════════════════════════════════════


class PyTorchKinematicsFrankaFK(FrankaKinematicsProtocol):
    """Franka Panda FK using pytorch_kinematics + URDF.

    Requires:
        pip install pytorch-kinematics

    URDF sources (any one works):
        - franka_description ROS package
        - https://github.com/frankaemika/franka_ros/tree/develop/franka_description
        - The URDF path is passed via --fk-urdf at runtime
    """

    def __init__(self, urdf_path: str, device: torch.device, ee_link_name: str = "panda_link8"):
        import pytorch_kinematics as pk

        self.device = device
        self.ee_link_name = ee_link_name

        with open(urdf_path, "r") as f:
            urdf_str = f.read()

        # Build kinematic chain from URDF
        self._chain = pk.build_serial_chain_from_urdf(
            urdf_str.encode() if isinstance(urdf_str, str) else urdf_str,
            end_link_name=ee_link_name,
            root_link_name="panda_link0",
        ).to(device=device)

        self._joint_names: list[str] = self._chain.get_joint_parameter_names()
        if len(self._joint_names) != 7:
            raise ValueError(
                f"Expected 7 revolute joints, found {len(self._joint_names)}: {self._joint_names}"
            )

        # Cached joint limits from URDF
        self.joint_limits_low = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=device,
            dtype=torch.float32,
        )
        self.joint_limits_high = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=device,
            dtype=torch.float32,
        )

        # Workspace parameters (Franka Panda, in meters)
        self.r_max = 0.855   # max reach
        self.r_min = 0.15    # min distance (self-collision risk near base)
        self.v_ee_max = 1.7  # max Cartesian velocity (m/s)

    @staticmethod
    def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
        return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _matrix_to_quat(rot: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to w,x,y,z quaternions."""
        m00, m01, m02 = rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2]
        m10, m11, m12 = rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2]
        m20, m21, m22 = rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2]

        q_abs = torch.sqrt(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ).clamp_min(0.0)
        )
        candidates = torch.stack(
            [
                torch.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m21 + m12], dim=-1),
                torch.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[:, 3] ** 2], dim=-1),
            ],
            dim=1,
        )
        candidates = candidates / (2.0 * q_abs.clamp_min(0.1)).unsqueeze(-1)
        quat = candidates[torch.arange(rot.shape[0], device=rot.device), q_abs.argmax(dim=-1)]
        return PyTorchKinematicsFrankaFK._normalize_quat(quat)

    def _single_fk(self, joints_7dof: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FK for a single 7-DoF arm.

        Args:
            joints_7dof: [B, 7]

        Returns:
            ee_xyz: [B, 3]
            ee_quat: [B, 4]  (w, x, y, z) or (x, y, z, w) depending on convention
        """
        joints = joints_7dof.to(device=self.device, dtype=torch.float32)
        ret = self._chain.forward_kinematics(joints)
        # ret is a Transform3d or matrix; extract translation and rotation
        if hasattr(ret, "get_matrix"):
            matrices = ret.get_matrix()  # [B, 4, 4]
        else:
            matrices = ret  # already [B, 4, 4]

        ee_xyz = matrices[:, :3, 3]

        rot = matrices[:, :3, :3]
        ee_quat = self._matrix_to_quat(rot)

        return ee_xyz, ee_quat

    def ee_pose(
        self, left_joints: torch.Tensor, right_joints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_xyz, left_quat = self._single_fk(left_joints)
        right_xyz, right_quat = self._single_fk(right_joints)
        left_ee = torch.cat([left_xyz, left_quat], dim=1)
        right_ee = torch.cat([right_xyz, right_quat], dim=1)
        return left_ee, right_ee

    def jacobian_condition(self, joints_7dof: torch.Tensor) -> torch.Tensor:
        """Approximate singularity measure via Jacobian condition number.

        Uses automatic differentiation through FK to compute the positional
        Jacobian, then returns its condition number.
        """
        with torch.enable_grad():
            joints = joints_7dof.detach().clone().requires_grad_(True)
            xyz, _ = self._single_fk(joints)

            jac_rows = []
            for i in range(3):  # x, y, z
                grad = torch.autograd.grad(
                    xyz[:, i].sum(), joints, create_graph=False, retain_graph=(i < 2)
                )[0]
                jac_rows.append(grad)

            J = torch.stack(jac_rows, dim=1)  # [B, 3, 7]
            JJT = J @ J.transpose(1, 2)  # [B, 3, 3]
            try:
                eigvals = torch.linalg.eigvalsh(JJT)  # [B, 3], sorted ascending
                cond = torch.sqrt(
                    eigvals[:, -1].clamp_min(1e-8) / eigvals[:, 0].clamp_min(1e-8)
                )
            except Exception:
                cond = torch.linalg.norm(J, dim=(1, 2)) / torch.linalg.det(JJT).abs().sqrt().clamp_min(1e-8)

        return cond.detach()


def build_franka_fk(
    urdf_path: str, device: torch.device, backend: str = "pytorch_kinematics"
) -> FrankaKinematicsProtocol:
    """Factory for Franka FK instances."""
    if backend == "pytorch_kinematics":
        return PyTorchKinematicsFrankaFK(urdf_path, device)
    raise ValueError(f"Unsupported FK backend: {backend}. Use 'pytorch_kinematics'.")


# ═══════════════════════════════════════════════════════════════════
# FK Reward Components
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FKRewardConfig:
    """Configuration for FK-mediated reward components."""

    # C3: Workspace
    ws_r_max: float = 0.855   # Franka max reach (m)
    ws_r_min: float = 0.15    # min distance to base
    ws_weight: float = 0.5

    # C4: Singularity
    singularity_threshold: float = 50.0  # Jacobian condition number threshold
    singularity_weight: float = 0.1

    # C9: Cartesian velocity limit
    ee_vel_max: float = 1.7   # m/s
    ee_vel_weight: float = 0.5

    # C10: Cartesian acceleration limit
    ee_acc_max: float = 10.0  # m/s²
    ee_acc_weight: float = 0.25

    # C11: FK chain self-consistency
    fk_chain_weight: float = 0.5

    # C2: Dual-arm collision warning
    dual_arm_min_dist: float = 0.05  # meters
    dual_arm_weight: float = 0.5

    # Cartesian smoothness (Huber delta)
    cart_huber_delta_vel: float = 0.02   # 2cm
    cart_huber_delta_acc: float = 0.01   # 1cm

    # Time step
    dt: float = 1.0 / 30.0  # 30fps

    # Whether FK model is available
    fk_enabled: bool = True

    # Arm joint indices in the 16-dim action vector
    left_arm_slice: tuple[int, int] = (0, 7)
    right_arm_slice: tuple[int, int] = (8, 15)


@dataclass
class FKRewardBreakdown:
    """Per-component FK reward values (all [B] tensors, higher = better)."""

    workspace_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    singularity_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    cartesian_vel_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    cartesian_acc_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    fk_chain_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    dual_arm_reward: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    total: torch.Tensor = field(default_factory=lambda: torch.empty(0))


def huber_penalty(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber penalty: quadratic for |x| ≤ delta, linear beyond. Returns positive cost."""
    abs_x = x.abs()
    quadratic = 0.5 * x.pow(2)
    linear = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quadratic, linear).mean(dim=-1)


def _extract_arm_joints(
    action: torch.Tensor, left_slice: tuple[int, int], right_slice: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract left and right arm joint angles from 16-dim action.

    Args:
        action: [B, 16] joint position
    Returns:
        left:  [B, 7]
        right: [B, 7]
    """
    left = action[:, left_slice[0] : left_slice[1]]
    right = action[:, right_slice[0] : right_slice[1]]
    return left.contiguous(), right.contiguous()


class FKExecutabilityComponents:
    """FK-mediated reward components to augment StepExecutabilityReward."""

    def __init__(self, fk_model: FrankaKinematicsProtocol | None, cfg: FKRewardConfig | None = None):
        self.fk_model = fk_model
        self.cfg = cfg or FKRewardConfig()
        if fk_model is not None:
            self.cfg.fk_enabled = True
        else:
            self.cfg.fk_enabled = False
        self._last_breakdown: FKRewardBreakdown | None = None

    @property
    def last_breakdown(self) -> FKRewardBreakdown | None:
        return self._last_breakdown

    def compute(
        self,
        action_t: torch.Tensor,           # [B, 16] current action = next joint state
        *,
        prev_actions: torch.Tensor | None = None,  # [B, T, 16] previous actions in chain
        control_state_t: torch.Tensor | None = None,  # [B, 16] current control state
    ) -> FKRewardBreakdown:
        """Compute all FK-mediated reward components.

        Returns reward values (higher = better / more executable).
        """
        B = action_t.shape[0]
        dev = action_t.device

        if not self.cfg.fk_enabled or self.fk_model is None:
            zero = torch.zeros(B, device=dev)
            bd = FKRewardBreakdown(
                workspace_reward=zero,
                singularity_reward=zero,
                cartesian_vel_reward=zero,
                cartesian_acc_reward=zero,
                fk_chain_reward=zero,
                dual_arm_reward=zero,
                total=zero,
            )
            self._last_breakdown = bd
            return bd

        cfg = self.cfg
        dt = cfg.dt

        # ── Extract arm joint angles ──────────────────────────────────
        left_joints, right_joints = _extract_arm_joints(
            action_t, cfg.left_arm_slice, cfg.right_arm_slice
        )

        # ── FK for current action ──────────────────────────────────────
        left_ee, right_ee = self.fk_model.ee_pose(left_joints, right_joints)
        left_xyz = left_ee[:, :3]    # [B, 3]
        right_xyz = right_ee[:, :3]  # [B, 3]

        # ── C3: Workspace constraint ───────────────────────────────────
        # ee distance from origin (base of robot)
        # For dual-arm setup, each arm has its own base. Simplified:
        # we check both arms are within Franka's spherical workspace.
        left_dist = left_xyz.norm(dim=1)
        right_dist = right_xyz.norm(dim=1)

        ws_left_violation = (
            F.relu(left_dist - cfg.ws_r_max) + F.relu(cfg.ws_r_min - left_dist)
        )
        ws_right_violation = (
            F.relu(right_dist - cfg.ws_r_max) + F.relu(cfg.ws_r_min - right_dist)
        )
        workspace_reward = -cfg.ws_weight * (ws_left_violation + ws_right_violation)

        # ── C4: Singularity proximity ──────────────────────────────────
        left_cond = self.fk_model.jacobian_condition(left_joints)
        right_cond = self.fk_model.jacobian_condition(right_joints)
        sing_violation = (
            F.relu(left_cond - cfg.singularity_threshold)
            + F.relu(right_cond - cfg.singularity_threshold)
        )
        singularity_reward = -cfg.singularity_weight * sing_violation

        # ── C9: Cartesian velocity limit ───────────────────────────────
        # Need previous ee position to compute velocity
        if prev_actions is not None and prev_actions.shape[1] > 0:
            prev_action = prev_actions[:, -1]  # [B, 16] — last action in chain
            prev_left, prev_right = _extract_arm_joints(
                prev_action, cfg.left_arm_slice, cfg.right_arm_slice
            )
            prev_left_ee, prev_right_ee = self.fk_model.ee_pose(prev_left, prev_right)
            prev_left_xyz = prev_left_ee[:, :3]
            prev_right_xyz = prev_right_ee[:, :3]

            ee_vel_left = (left_xyz - prev_left_xyz) / dt
            ee_vel_right = (right_xyz - prev_right_xyz) / dt
            ee_speed_left = ee_vel_left.norm(dim=1)
            ee_speed_right = ee_vel_right.norm(dim=1)

            vel_violation = (
                F.relu(ee_speed_left - cfg.ee_vel_max)
                + F.relu(ee_speed_right - cfg.ee_vel_max)
            )
            cartesian_vel_reward = -cfg.ee_vel_weight * vel_violation

            # ── C10: Cartesian acceleration limit ──────────────────────
            if prev_actions.shape[1] > 1:
                prev2_action = prev_actions[:, -2]
                prev2_left, prev2_right = _extract_arm_joints(
                    prev2_action, cfg.left_arm_slice, cfg.right_arm_slice
                )
                prev2_left_ee, prev2_right_ee = self.fk_model.ee_pose(prev2_left, prev2_right)
                prev2_left_xyz = prev2_left_ee[:, :3]
                prev2_right_xyz = prev2_right_ee[:, :3]

                ee_vel_left_prev = (prev_left_xyz - prev2_left_xyz) / dt
                ee_vel_right_prev = (prev_right_xyz - prev2_right_xyz) / dt

                ee_acc_left = (ee_vel_left - ee_vel_left_prev) / dt
                ee_acc_right = (ee_vel_right - ee_vel_right_prev) / dt
                ee_acc_norm_left = ee_acc_left.norm(dim=1)
                ee_acc_norm_right = ee_acc_right.norm(dim=1)

                acc_violation = (
                    F.relu(ee_acc_norm_left - cfg.ee_acc_max)
                    + F.relu(ee_acc_norm_right - cfg.ee_acc_max)
                )
                cartesian_acc_reward = -cfg.ee_acc_weight * acc_violation
            else:
                cartesian_acc_reward = torch.zeros(B, device=dev)
        else:
            cartesian_vel_reward = torch.zeros(B, device=dev)
            cartesian_acc_reward = torch.zeros(B, device=dev)

        # ── C11: FK chain self-consistency ────────────────────────────
        # If we have prev_actions, check that the Cartesian displacement
        # from prev_action to action_t is physically achievable.
        # This detects when the world model's visual inconsistency forces
        # the IDM to decode a physically impossible state jump.
        if prev_actions is not None and prev_actions.shape[1] > 0:
            # Cartesian displacement in one step
            prev_left_ee, prev_right_ee = self.fk_model.ee_pose(prev_left, prev_right)

            cart_step_left = (left_xyz - prev_left_ee[:, :3]).norm(dim=1)
            cart_step_right = (right_xyz - prev_right_ee[:, :3]).norm(dim=1)

            # Max physically possible Cartesian step in one dt:
            # v_max * dt ≈ 1.7 m/s * 0.033s ≈ 5.7cm
            max_step = cfg.ee_vel_max * dt

            step_violation = (
                F.relu(cart_step_left - max_step)
                + F.relu(cart_step_right - max_step)
            )
            fk_chain_reward = -cfg.fk_chain_weight * step_violation
        else:
            fk_chain_reward = torch.zeros(B, device=dev)

        # ── C2: Dual-arm collision warning ─────────────────────────────
        # Simplified: left-right end-effector distance
        lr_dist = (left_xyz - right_xyz).norm(dim=1)
        collision_violation = F.relu(cfg.dual_arm_min_dist - lr_dist)
        dual_arm_reward = -cfg.dual_arm_weight * collision_violation

        # ── Aggregate ──────────────────────────────────────────────────
        total = (
            workspace_reward
            + singularity_reward
            + cartesian_vel_reward
            + cartesian_acc_reward
            + fk_chain_reward
            + dual_arm_reward
        )

        bd = FKRewardBreakdown(
            workspace_reward=workspace_reward,
            singularity_reward=singularity_reward,
            cartesian_vel_reward=cartesian_vel_reward,
            cartesian_acc_reward=cartesian_acc_reward,
            fk_chain_reward=fk_chain_reward,
            dual_arm_reward=dual_arm_reward,
            total=total,
        )
        self._last_breakdown = bd
        return bd
