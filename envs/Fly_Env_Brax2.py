import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.base import Motion, Transform
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from typing import Any, List, Sequence, Dict, Tuple

from dm_control import mjcf as mjcf_dm

from jax.numpy import inf, ndarray
import mujoco
from mujoco import mjx

import numpy as np

import os
from pathlib import Path
from preprocessing.mjx_preprocess import ReferenceClip


def _bounded_quat_dist(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Computes a quaternion distance limiting the difference to a max of pi/2.

    This function supports an arbitrary number of batch dimensions, B.

    Args:
        source: a quaternion, shape (B, 4).
        target: another quaternion, shape (B, 4).

    Returns:
        Quaternion distance, shape (B, 1).
    """
    source /= jp.linalg.norm(source, axis=-1, keepdims=True)
    target /= jp.linalg.norm(target, axis=-1, keepdims=True)
    # "Distance" in interval [-1, 1].
    dist = 2 * jp.einsum("...i,...i", source, target) ** 2 - 1
    # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
    dist = jp.minimum(1.0, dist)
    # Divide by 2 and add an axis to ensure consistency with expected return
    # shape and magnitude.
    return 0.5 * jp.arccos(dist)[..., np.newaxis]


class FlyTracking(PipelineEnv):
    """Single clip rodent tracking"""

    def __init__(
        self,
        reference_clip,
        body_names: List[str],
        joint_names: List[str],
        end_eff_names: List[str],
        n_clips: int = 1,
        clip_length: int = 1001,
        mocap_hz: int = 500,
        mjcf_path: str = None,
        torque_actuators: bool = False,
        physics_timestep: float = 2e-4,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        ctrl_diff_cost_weight=0.01,
        pos_reward_weight=1.0,
        quat_reward_weight=1.0,
        joint_reward_weight=1.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        pos_scaling=400.0,
        quat_scaling=4.0,
        joint_scaling=0.25,
        angvel_scaling=0.5,
        bodypos_scaling=8.0,
        endeff_scaling=500.0,
        healthy_z_range=(-0.03, 0.1),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        inference_mode: bool = False,
        free_jnt: bool = False,
        **kwargs,
    ):

        # Convert to torque actuators
        if (torque_actuators) and (mjcf_path is None):
            from pathlib import Path

            mjcf_path = (Path(mjcf_path).parent / "fruitfly_force_fast.xml").as_posix()

        # root = mjcf_dm.from_path(mjcf_path)
        spec = mujoco.MjSpec()
        spec = spec.from_file(mjcf_path)
        # first_joint.delete()
        mj_model = spec.compile()
        print("Loaded Model:", mjcf_path)
        # for actuator in root.find_all("actuator"):
        #     actuator.gainprm = [actuator.forcerange[1]]
        #     del actuator.biastype
        #     del actuator.biasprm

        # mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = physics_timestep
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        max_physics_steps_per_control_step = int(
            (1.0 / (mocap_hz * mj_model.opt.timestep))
        )

        super().__init__(sys, **kwargs)
        if max_physics_steps_per_control_step % physics_steps_per_control_step != 0:
            raise ValueError(
                f"physics_steps_per_control_step ({physics_steps_per_control_step}) must be a factor of ({max_physics_steps_per_control_step})"
            )

        self._steps_for_cur_frame = (
            max_physics_steps_per_control_step / physics_steps_per_control_step
        )
        print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

        self._thorax_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), "thorax"
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in body_names
            ]
        )

        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in end_eff_names
            ]
        )
        self._n_clips = n_clips
        self._clip_length = clip_length
        self._reference_clip = reference_clip
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._ctrl_diff_cost_weight = ctrl_diff_cost_weight
        self._pos_scaling = pos_scaling
        self._joint_scaling = joint_scaling
        self._angvel_scaling = angvel_scaling
        self._bodypos_scaling = bodypos_scaling
        self._endeff_scaling = endeff_scaling
        self._quat_scaling = quat_scaling
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, rng = jax.random.split(rng, 3)

        start_frame = jax.random.randint(start_rng, (), 0, 44)

        info = {
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info)

    def reset_from_clip(self, rng, info) -> State:
        """Reset based on a reference clip."""
        _, rng1, rng2 = jax.random.split(rng, 3)

        # Get reference clip and select the start frame
        reference_frame = jax.tree_map(
            lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
        )

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # New pos from reference clip
        new_qpos = jp.concatenate(
            (
                reference_frame['position'],
                reference_frame['quaternion'],
                reference_frame['joints'],
            ),
            axis=0,
        )
        # new_qpos = self.sys.qpos0
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        # Randomly sample velocities 
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        # floor_z = self.sys.mj_model.geom('floor').pos[2]
        # # Physics step check if feet penetrate floor
        # def body_fn(vals):
        #     qpos,qvel,floor_z = vals
        #     qpos = qpos.at[2].set(qpos[2]+0.0001)
        #     return (qpos,qvel,floor_z)
        # def cond_fn(vals):
        #     qpos,qvel,floor_z = vals
        #     qpos = qpos.at[2].set(qpos[2]+0.0001)
        #     data = self.pipeline_init(qpos, qvel)
        #     feet_z = data.xpos[self._endeff_idxs,-1]
        #     print(feet_z>floor_z)
        #     return jp.all(feet_z>floor_z)
        # jax.lax.while_loop(cond_fun=cond_fn,body_fun=body_fn, init_val=(qpos,qvel,floor_z))
        
        data = self.pipeline_init(qpos, qvel)
        # Grab observations
        reference_obs, proprioceptive_obs = self._get_obs(data, info)

        # Used to intialize our intention network
        info["task_obs_size"] = reference_obs.shape[-1]

        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        reward, done, zero = jp.zeros(3)
        # Initialize metrics
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_ctrlcost": zero,
            "ctrl_diff_cost": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "termination": zero,
            "fall": zero,
        }

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Logic for moving to next frame to track to maintain timesteps alignment
        # TODO: Update this to just refer to model.timestep
        info = state.info.copy()
        info["steps_taken_cur_frame"] += 1
        info["cur_frame"] += jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 1, 0
        )
        info["steps_taken_cur_frame"] *= jp.where(
            info["steps_taken_cur_frame"] == self._steps_for_cur_frame, 0, 1
        )

        # Gets reference clip and indexes to current frame
        reference_clip = jax.tree_map(
            lambda x: x[info["cur_frame"]], self._get_reference_clip(info)
        )

        pos_distance = data.qpos[:3] - reference_clip['position']
        pos_reward = self._pos_reward_weight * jp.exp(
            -self._pos_scaling * jp.sum(pos_distance**2)
        )

        quat_distance = jp.sum(
            _bounded_quat_dist(data.qpos[3:7], reference_clip['quaternion']) ** 2
        )
        quat_reward = self._quat_reward_weight * jp.exp(
            -self._quat_scaling * quat_distance
        )

        joint_distance = jp.sum((data.qpos[7:] - reference_clip['joints']) ** 2)
        joint_reward = self._joint_reward_weight * jp.exp(
            -self._joint_scaling * joint_distance
        )
        info["joint_distance"] = joint_distance

        angvel_distance = jp.sum(
            (data.qvel[3:6] - reference_clip['angular_velocity']) ** 2
        )
        angvel_reward = self._angvel_reward_weight * jp.exp(
            -self._angvel_scaling * angvel_distance
        )
        info["angvel_distance"] = angvel_distance

        bodypos_distance = jp.sum(
            (
                data.xpos[self._body_idxs]
                - reference_clip['body_positions'][self._body_idxs]
            ).flatten()
            ** 2
        )
        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -self._bodypos_scaling * bodypos_distance
        )
        info["bodypos_distance"] = bodypos_distance

        endeff_distance = jp.sum(
            (
                data.xpos[self._endeff_idxs]
                - reference_clip['body_positions'][self._endeff_idxs]
            ).flatten()
            ** 2
        )
        endeff_reward = self._endeff_reward_weight * jp.exp(
            -self._endeff_scaling * endeff_distance
        )
        info["endeff_distance"] = endeff_distance

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] > max_z, 0.0, is_healthy)
        fall = 1.0 - is_healthy

        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        ctrl_diff_cost = self._ctrl_diff_cost_weight * jp.sum(
            jp.square(info["prev_ctrl"] - action)
        )
        info["prev_ctrl"] = action

        reference_obs, proprioceptive_obs = self._get_obs(data, info)
        obs = jp.concatenate([reference_obs, proprioceptive_obs])
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            - ctrl_cost
            - ctrl_diff_cost
        )

        # Raise done flag if terminating
        done = jp.max(jp.array([fall, too_far, bad_pose, bad_quat]))

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        obs = jp.nan_to_num(obs)

        from jax.flatten_util import ravel_pytree

        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, done]))

        state.metrics.update(
            pos_reward=pos_reward,
            quat_reward=quat_reward,
            joint_reward=joint_reward,
            angvel_reward=angvel_reward,
            bodypos_reward=bodypos_reward,
            endeff_reward=endeff_reward,
            reward_ctrlcost=-ctrl_cost,
            ctrl_diff_cost=ctrl_diff_cost,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            termination=done,
            fall=fall,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Returns reference clip; to be overridden in child classes"""
        return self._reference_clip

    def _get_reference_trajectory(self, info) -> ReferenceClip:
        """Slices ReferenceClip into the observation trajectory"""

        # Get the relevant slice of the reference clip
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    info["cur_frame"] + 1,
                    self._ref_len,
                )
            return jp.array([])

        return jax.tree_util.tree_map(f, self._get_reference_clip(info))

    def _get_obs(self, data: mjx.Data, info) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        ref_traj = self._get_reference_trajectory(info)

        track_pos_local = jax.vmap(
            lambda a, b: brax_math.rotate(a, b), in_axes=(0, None)
        )(
            ref_traj['position'] - data.qpos[:3],
            data.qpos[3:7],
        ).flatten()

        quat_dist = jax.vmap(
            lambda a, b: brax_math.relative_quat(a, b), in_axes=(0, None)
        )(
            ref_traj['quaternion'],
            data.qpos[3:7],
        ).flatten()

        joint_dist = (ref_traj['joints'] - data.qpos[7:])[:, self._joint_idxs].flatten()

        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (ref_traj['body_positions'] - data.xpos)[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        reference_obs = jp.concatenate(
            [
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

        prorioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
            ]
        )
        return reference_obs, prorioceptive_obs

    def render(
        self,
        trajectory: List[State],
        camera: str | None = None,
        width: int = 480,
        height: int = 320,
        scene_option: Any = None,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track1"
        return super().render(
            trajectory,
            camera=camera,
            width=width,
            height=height,
            scene_option=scene_option,
        )


class FlyMultiClipTracking(FlyTracking):
    def __init__(
        self,
        reference_clip,
        body_names: List[str],
        joint_names: List[str],
        end_eff_names: List[str],
        n_clips: int = 1,
        clip_length: int = 1001,
        mocap_hz: int = 500,
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_fast.xml",
        torque_actuators: bool = False,
        physics_timestep: float = 2e-4,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        ctrl_diff_cost_weight=0.01,
        pos_reward_weight=1.0,
        quat_reward_weight=1.0,
        joint_reward_weight=1.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        pos_scaling=400.0,
        quat_scaling=4.0,
        joint_scaling=0.25,
        angvel_scaling=0.5,
        bodypos_scaling=8.0,
        endeff_scaling=500.0,
        healthy_z_range=(-0.03, 0.1),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        inference_mode: bool = False,
        free_jnt: bool = False,
        **kwargs,
    ):
        super().__init__(
            None,
            body_names,
            joint_names,
            end_eff_names,
            n_clips,
            clip_length,
            mocap_hz,
            mjcf_path,
            torque_actuators,
            physics_timestep,
            ref_len,
            too_far_dist,
            bad_pose_dist,
            bad_quat_dist,
            ctrl_cost_weight,
            ctrl_diff_cost_weight,
            pos_reward_weight,
            quat_reward_weight,
            joint_reward_weight,
            angvel_reward_weight,
            bodypos_reward_weight,
            endeff_reward_weight,
            pos_scaling,
            quat_scaling,
            joint_scaling,
            angvel_scaling,
            bodypos_scaling,
            endeff_scaling,
            healthy_z_range,
            physics_steps_per_control_step,
            reset_noise_scale,
            solver,
            iterations,
            ls_iterations,
            inference_mode,
            free_jnt,
            **kwargs,
        )

        self._reference_clips = reference_clip
        self._n_clips = reference_clip['position'].shape[0]

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, clip_rng, rng = jax.random.split(rng, 4)

        start_frame = jax.random.randint(start_rng, (), 0, 44)
        clip_idx = jax.random.randint(clip_rng, (), 0, self._n_clips)
        info = {
            "clip_idx": clip_idx,
            "cur_frame": start_frame,
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info)

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Gets clip based on info["clip_idx"]"""
        return jax.tree_map(lambda x: x[info["clip_idx"]], self._reference_clips)


class FlyRunSim(PipelineEnv):
    def __init__(
        self,
        reference_clip,
        body_names: List[str],
        joint_names: List[str],
        end_eff_names: List[str],
        lower_leg_names: List[str],
        mjcf_path: str = "./assets/fruitfly/fruitfly_force_fast.xml",
        clip_length: int = 1000,
        obs_noise: float = 0.05,
        too_far_dist=0.01,
        bad_pose_dist=10.0,
        pos_reward_weight=1.0,
        tracking_lin_vel_weight=1.5,
        tracking_ang_vel_weight=0.8,
        lin_vel_z_weight=-5e-5,
        ang_vel_xy_weight=-1e-3,
        orientation_weight=1.0,
        torques_weight=-0.0002,
        action_rate_weight=-0.01,
        stand_still_weight=-0.5,
        foot_slip_weight=-0.1,
        termination_weight=-1.0,
        pos_scaling=200.0, 
        linvel_scaling=0.1,
        angvel_scaling=0.1,
        ang_vel_xy_scaling=-1.0,
        lin_vel_z_scaling=-1.0,
        orientation_scaling=5.0,
        torques_scaling=-1.0,
        action_rate_scaling=-1.0,
        stand_still_scaling=-1.0,
        foot_slip_scaling=-1.0,
        healthy_z_range=(-0.05, 0.1),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        action_scale: float = 0.3,
        physics_timestep: float = 2e-4,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        ref_len=15,
        free_jnt=True,
        inference_mode=False,
        torque_actuators=True,
        center_of_mass="thorax",
        **kwargs,
    ):

        # Convert to torque actuators
        if torque_actuators:
            mjcf_path = (Path(mjcf_path).parent / "fruitfly_force_fast.xml").as_posix()
        else:
            mjcf_path = (Path(mjcf_path).parent / "fruitfly_fast.xml").as_posix()

        spec = mujoco.MjSpec()
        spec = spec.from_file(mjcf_path)
        mj_model = spec.compile()

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations
        mj_model.opt.timestep = physics_timestep
        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._steps_for_cur_frame = 1
        print(f"self._steps_for_cur_frame: {self._steps_for_cur_frame}")

        self._thorax_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), "thorax"
        )

        self._joint_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("joint"), joint)
                for joint in joint_names
            ]
        )

        self._body_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in body_names
            ]
        )

        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in end_eff_names
            ]
        )

        self._lower_leg_idx = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in lower_leg_names
            ]
        )
        self._nv = sys.nv
        self._nq = sys.nq
        self._nu = sys.nu
        self._too_far_dist = too_far_dist
        self._bad_pose_dist = bad_pose_dist
        self._foot_radius = 0.00219
        self._init_pos = self.sys.qpos0[:3]
        self._default_pose = self.sys.qpos0[7:]
        self._action_scale = action_scale
        self._n_clips = 1
        self._clip_length = clip_length
        self._ref_len = ref_len
        self._ref_dim = (7 + self._nu) * self._ref_len
        self._prop_dim = self._nv + self._nq
        self._reference_clips = reference_clip
        self._reset_noise_scale = reset_noise_scale
        self._physics_timestep = physics_timestep
        self._free_jnt = free_jnt
        self._inference_mode = inference_mode
        self._pos_reward_weight = pos_reward_weight
        self._tracking_lin_vel_weight = tracking_lin_vel_weight
        self._tracking_ang_vel_weight = tracking_ang_vel_weight
        self._lin_vel_z_weight = lin_vel_z_weight
        self._ang_vel_xy_weight = ang_vel_xy_weight
        self._orientation_weight = orientation_weight
        self._torques_weight = torques_weight
        self._action_rate_weight = action_rate_weight
        self._stand_still_weight = stand_still_weight
        self._foot_slip_weight = foot_slip_weight
        self._termination_weight = termination_weight
        self._pos_scaling = pos_scaling
        self._linvel_scaling = linvel_scaling
        self._angvel_scaling = angvel_scaling
        self._lin_vel_z_scaling = lin_vel_z_scaling
        self._ang_vel_xy_scaling = ang_vel_xy_scaling
        self._orientation_scaling = orientation_scaling
        self._torques_scaling = torques_scaling
        self._action_rate_scaling=action_rate_scaling
        self._stand_still_scaling = stand_still_scaling
        self._foot_slip_scaling = foot_slip_scaling
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        _, start_rng, rng = jax.random.split(rng, 3)

        info = {
            "clip_idx": 0,
            "cur_frame": 0,
            "command": jp.zeros(3),
            "last_contact": jp.zeros(6, dtype=bool),
            "steps_taken_cur_frame": 0,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
            "angvel_distance": 0.0,
            "bodypos_distance": 0.0,
            "endeff_distance": 0.0,
            "prev_ctrl": jp.zeros((self.sys.nu,)),
        }

        return self.reset_from_clip(rng, info)

    def reset_from_clip(self, rng, info) -> State:
        """Reset based on a reference clip."""
        rng0, rng1, rng2 = jax.random.split(rng, 3)

        ##### Handle Additional Info #####
        if "prev_ctrl" not in info:
            info["prev_ctrl"] = jp.zeros(self._nu)
        if "last_vel" not in info:
            info["last_vel"] = jp.zeros(self._nv - 7)
        if "las_contact" not in info:
            info["last_contact"] = jp.zeros(6, dtype=bool)
        if "rng" not in info:
            info["rng"] = rng0
        if "command" not in info:
            info["command"] = self.sample_command(info["rng"])
        if "step" not in info:
            info["step"] = 0
        if 'last_pos' not in info:
            info['last_pos'] = 0

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)
        obs_history = jp.zeros(self._ref_dim)
        reference_obs, proprioceptive_obs = self._get_obs(data, info, obs_history)

        # Used to intialize our intention network
        info["task_obs_size"] = reference_obs.shape[-1]

        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        reward, done, zero = jp.zeros(3)
        metrics = {
            "total_dist": zero,
            "tracking_lin_vel": zero,
            "tracking_ang_vel": zero,
            "ang_vel_xy": zero,
            "lin_vel_z": zero,
            "orientation": zero,
            "torques": zero,
            "action_rate": zero,
            "stand_still": zero,
            "foot_slip": zero,
            "termination": 1.0,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        data0 = state.pipeline_state
        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        data = self.pipeline_step(data0, motor_targets)
        # data = self.pipeline_step(data0, action)

        info = state.info.copy()
        joint_angles = data.q[7:]
        joint_vel = data.qd[7:]
        x, xd = data.x, data.xd

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._thorax_idx][2] > max_z, 0.0, is_healthy)
        fall = 1.0 - is_healthy
        # up = jp.array([0.0, 0.0, 1.0])
        # done = jp.int32(jp.dot(brax_math.rotate(up, x.rot[self._thorax_idx - 1]), up) < 0)
        

        reference_obs, proprioceptive_obs = self._get_obs(data, info, state.obs)
        obs = jp.concatenate([reference_obs, proprioceptive_obs])

        ##### Tracking position from velocity #####
        target_pos = self._init_pos + info['step']*jp.concatenate((info['command'][:2],jp.zeros(1)))*self.dt
        pos_distance = (data.qpos[:3] - target_pos)
        pos_reward = self._pos_reward_weight * jp.exp(
            -self._pos_scaling * jp.sum(pos_distance**2)
        )
        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        ##### Tracking joint positions #####
        joint_distance = jp.sum((data.qpos[7:] - self._default_pose) ** 2)
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        info['joint_distance'] = joint_distance
        # ##### Tracking quaternion flat orientation ######
        # quat_distance = jp.sum(
        #     _bounded_quat_dist(data.qpos[3:7], jp.array()) ** 2
        # )
        # quat_reward = self._quat_reward_weight * jp.exp(
        #     -self._quat_scaling * quat_distance
        # )
        
        
        from jax.flatten_util import ravel_pytree
        flattened_vals, _ = ravel_pytree(data)
        num_nans = jp.sum(jp.isnan(flattened_vals))
        nan = jp.where(num_nans > 0, 1.0, 0.0)
        done = jp.max(jp.array([nan, fall, too_far, bad_pose]))

        # foot contact data based on z-position
        foot_pos = data.site_xpos[self._endeff_idxs] 
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]

        # Tracking of linear velocity commands (xy axes)
        local_vel = brax_math.rotate(xd.vel[0], brax_math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(info["command"][:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-self._linvel_scaling * lin_vel_error)
        info["bodypos_distance"] = lin_vel_error
        tracking_lin_vel = self._tracking_lin_vel_weight * lin_vel_reward
        # tracking_lin_vel = self._tracking_lin_vel_weight * self._reward_tracking_lin_vel(info['command'], x, xd)
        tracking_ang_vel = (
            self._tracking_ang_vel_weight
            * self._reward_tracking_ang_vel(info["command"], x, xd)
        )
        
        ang_vel_xy = jp.clip(self._ang_vel_xy_weight * self._reward_ang_vel_xy(xd),self._ang_vel_xy_scaling,0)
        lin_vel_z = jp.clip(self._lin_vel_z_weight * self._reward_lin_vel_z(xd),self._lin_vel_z_scaling,0)
        orientation = self._orientation_weight * jp.exp(-self._orientation_scaling*self._reward_orientation(x))
        torques = jp.clip(self._torques_weight * self._reward_torques(data.qfrc_actuator),self._torques_scaling,0)
        action_rate = jp.clip(self._action_rate_weight * self._reward_action_rate(
            action, info["prev_ctrl"]
        ), self._action_rate_scaling,0)
        stand_still = jp.clip(self._stand_still_weight * self._reward_stand_still(
            info["command"],
            joint_angles,
        ),self._stand_still_scaling,0)
        foot_slip = jp.clip(self._foot_slip_weight*self._reward_foot_slip(data, contact_filt_cm),self._foot_slip_scaling,0)
        termination = self._termination_weight * self._reward_termination(
            done, info["step"]
        )

        reward = (
            pos_reward
            + tracking_lin_vel
            + tracking_ang_vel
            + ang_vel_xy
            + lin_vel_z
            + orientation
            + torques
            + action_rate
            + stand_still
            + foot_slip
            + termination
        ) 

        # Handle nans during sim by resetting env
        reward = jp.nan_to_num(reward)
        # reward =  jp.clip(reward,0.0,10000.0)
        obs = jp.nan_to_num(obs)
        
        # state management
        info["prev_ctrl"] = action
        info["last_vel"] = joint_vel
        info["step"] += 1
        info["rng"] = rng
        info['last_contact'] = contact

        # sample new command if more than half the clip timesteps achieved
        state.info["command"] = jp.where(
            info["step"] > self._clip_length//2,
            self.sample_command(cmd_rng),
            info["command"],
        )
        # reset the step counter when done
        info["step"] = jp.where(
            (done > 0.5) | (info["step"] > self._clip_length), 0, info["step"]
        )

        state.metrics.update(
            total_dist=brax_math.normalize(x.pos[self._thorax_idx])[1],
            tracking_lin_vel=tracking_lin_vel,
            tracking_ang_vel=tracking_ang_vel,
            ang_vel_xy=ang_vel_xy,
            lin_vel_z=lin_vel_z,
            orientation=orientation,
            torques=torques,
            action_rate=action_rate,
            stand_still=stand_still,
            foot_slip=foot_slip,
            termination=termination,
        )
        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_reference_clip(self, info) -> ReferenceClip:
        """Gets clip based on info["clip_idx"]"""
        return jax.tree_map(lambda x: x[info["clip_idx"]], self._reference_clips)

    def _get_obs(
        self, data: mjx.Data, info, obs_history: jax.Array = None
    ) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""

        if obs_history is None:
            obs_history = jp.zeros(self._ref_dim)
        else:
            obs_history = obs_history[: self._ref_dim]

        inv_torso_rot = brax_math.quat_inv(data.x.rot[0])
        local_rpyrate = brax_math.rotate(data.xd.ang[0], inv_torso_rot)

        # stack observations through time
        obs = jp.concatenate(
            [
                jp.array([local_rpyrate[2]]) * 0.25,  # yaw rate (1)
                brax_math.rotate(
                    jp.array([0, 0, -1]), inv_torso_rot
                ),  # projected gravity (3)
                info["command"],  # command (3)
                data.q[7:] - self._default_pose, # motor angles
                info["prev_ctrl"],  # last action (self.nu)
            ]
        )

        reference_obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        proprioceptive_obs = jp.concatenate(
            [
                data.qpos,
                data.qvel,
            ]
        )

        return reference_obs, proprioceptive_obs

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-2.0, 2.0]  # min max [cm/s]
        lin_vel_y = [-0.1, 0.1]  # min max [cm/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1,key2,key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        # return jp.array([lin_vel_x[0],0,0])
        return new_cmd

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = brax_math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, prev_ctrl: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - prev_ctrl))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = brax_math.rotate(xd.vel[0], brax_math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-self._linvel_scaling * lin_vel_error)
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = brax_math.rotate(xd.ang[0], brax_math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-self._angvel_scaling * ang_vel_error)

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            brax_math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return (done < 0.5) & (step < self._clip_length)

    def _reward_foot_slip(
        self, pipeline_state: State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._endeff_idxs]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_idx]
        # pytype: enable=attribute-error
        offset = Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_idx - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def render(
        self,
        trajectory: List[State],
        camera: str | None = None,
        width: int = 480,
        height: int = 320,
        scene_option: Any = None,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(
            trajectory,
            camera=camera,
            width=width,
            height=height,
            scene_option=scene_option,
        )
