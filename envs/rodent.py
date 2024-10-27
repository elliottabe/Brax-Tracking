import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax
from brax import math as brax_math
from dm_control.locomotion.walkers import rescale
from dm_control import mjcf as mjcf_dm
from typing import List
import mujoco
from mujoco import mjx

import numpy as np

import os

class RodentSingleClip(PipelineEnv):

    def __init__(
        self,
        reference_clip,
        center_of_mass: str,
        # end_eff_names: List[str],
        appendage_names: List[str],
        body_names: List[str],
        joint_names: List[str],
        mocap_hz: int = 50,
        mjcf_path: str = "./assets/rodent.xml",
        scale_factor: float = 0.9,
        torque_actuators: bool = False,
        ref_len: int = 5,
        too_far_dist=0.1,
        bad_pose_dist=jp.inf,
        bad_quat_dist=jp.inf,
        ctrl_cost_weight=0.01,
        pos_reward_weight=10.0,
        quat_reward_weight=1.0,
        joint_reward_weight=10.0,
        angvel_reward_weight=1.0,
        bodypos_reward_weight=1.0,
        endeff_reward_weight=1.0,
        healthy_reward=0.25,
        healthy_z_range=(0.03, 0.5),
        physics_steps_per_control_step=10,
        reset_noise_scale=1e-3,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        **kwargs,
    ):
        root = mjcf_dm.from_path(mjcf_path)

        # Convert to torque actuators
        if torque_actuators:
            for actuator in root.find_all("actuator"):
                actuator.gainprm = [actuator.forcerange[1]]
                del actuator.biastype
                del actuator.biasprm

        rescale.rescale_subtree(
            root,
            scale_factor,
            scale_factor,
        )
        mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

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

        self._torso_idx = mujoco.mj_name2id(
            mj_model, mujoco.mju_str2Type("body"), center_of_mass
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

        # using this for appendage for now bc im to lazy to rename 
        self._endeff_idxs = jp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body)
                for body in appendage_names
            ]
        )

        self._mocap_hz = mocap_hz
        self._bad_pose_dist = bad_pose_dist
        self._too_far_dist = too_far_dist
        self._bad_quat_dist = bad_quat_dist
        self._ref_traj = reference_clip
        self._ref_len = ref_len
        self._pos_reward_weight = pos_reward_weight
        self._quat_reward_weight = quat_reward_weight
        self._joint_reward_weight = joint_reward_weight
        self._angvel_reward_weight = angvel_reward_weight
        self._bodypos_reward_weight = bodypos_reward_weight
        self._endeff_reward_weight = endeff_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

        start_frame = jax.random.randint(rng, (), 0, 44)

        info = {
            "start_frame": start_frame,
            "summed_pos_distance": 0.0,
            "quat_distance": 0.0,
            "joint_distance": 0.0,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # Add pos (without z height)
        qpos_with_pos = (
            jp.array(self.sys.qpos0).at[:2].set(self._track_pos[start_frame][:2])
        )

        # Add quat
        new_qpos = qpos_with_pos.at[3:7].set(self._track_quat[start_frame])

        # Add noise
        qpos = new_qpos + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, start_frame)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "quat_reward": zero,
            "joint_reward": zero,
            "angvel_reward": zero,
            "bodypos_reward": zero,
            "endeff_reward": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "too_far": zero,
            "bad_pose": zero,
            "bad_quat": zero,
            "fall": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()
        
        # Logic for getting current frame aligned with simulation time
        cur_frame = info["start_frame"] + (data.time // (1 / self._mocap_hz))

        pos_distance = data.qpos[:3] - self._track_pos[cur_frame]
        pos_reward = self._pos_reward_weight * jp.exp(-400 * jp.sum(pos_distance) ** 2)

        quat_distance = jp.sum(
            self._bounded_quat_dist(data.qpos[3:7], self._track_quat[cur_frame])
            ** 2
        )
        quat_reward = self._quat_reward_weight * jp.exp(-4.0 * quat_distance)

        joint_distance = (
            jp.sum(data.qpos[7:] - self._track_joint[state.cur_frame]) ** 2
        )
        joint_reward = self._joint_reward_weight * jp.exp(-0.5 * joint_distance)
        info["joint_distance"] = joint_distance

        angvel_reward = self._angvel_reward_weight * jp.exp(
            -0.5 * jp.sum(data.qvel[3:6] - self._track_angvel[cur_frame]) ** 2
        )

        bodypos_reward = self._bodypos_reward_weight * jp.exp(
            -6.0
            * jp.sum(
                (
                    data.xpos[self._body_idxs]
                    - self._track_bodypos[cur_frame][self._body_idxs]
                ).flatten()
            )
            ** 2
        )

        endeff_reward = self._endeff_reward_weight * jp.exp(
            -0.75
            * jp.sum(
                (
                    data.xpos[self._endeff_idxs]
                    - self._track_bodypos[cur_frame][self._endeff_idxs]
                ).flatten()
            )
            ** 2
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.xpos[self._torso_idx][2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.xpos[self._torso_idx][2] > max_z, 0.0, is_healthy)

        summed_pos_distance = jp.sum((pos_distance * jp.array([1.0, 1.0, 0.2])) ** 2)
        too_far = jp.where(summed_pos_distance > self._too_far_dist, 1.0, 0.0)
        info["summed_pos_distance"] = summed_pos_distance
        info["quat_distance"] = quat_distance
        bad_pose = jp.where(joint_distance > self._bad_pose_dist, 1.0, 0.0)
        bad_quat = jp.where(quat_distance > self._bad_quat_dist, 1.0, 0.0)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, cur_frame)
        reward = (
            joint_reward
            + pos_reward
            + quat_reward
            + angvel_reward
            + bodypos_reward
            + endeff_reward
            + self._healthy_reward
            - ctrl_cost
        )
        done = 1.0 - is_healthy
        done = jp.max(jp.array([done, too_far, bad_pose, bad_quat]))

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
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            too_far=too_far,
            bad_pose=bad_pose,
            bad_quat=bad_quat,
            fall=1 - is_healthy,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_obs(self, data: mjx.Data, cur_frame: int) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""
        
        # Get the relevant slice of the ref_traj
        def f(x):
            if len(x.shape) != 1:
                return jax.lax.dynamic_slice_in_dim(
                    x,
                    cur_frame + 1,
                    self._ref_len,
                )
            return jp.array([])

        ref_traj = jax.tree_util.tree_map(f, self._ref_traj)
        
        track_pos_local = jax.vmap(
            lambda a, b: brax_math.rotate(a, b), in_axes=(0, None)
        )(
            ref_traj.position
            - data.qpos[:3],
            data.qpos[3:7],
        ).flatten()

        quat_dist = jax.vmap(
            lambda a, b: brax_math.relative_quat(a, b), in_axes=(None, 0)
        )(
            data.qpos[3:7],
            ref_traj.quaternion,
        ).flatten()

        joint_dist = (
            ref_traj.joints
            - data.qpos[7:]
        )[:, self._joint_idxs].flatten()

        # TODO test if this works
        body_pos_dist_local = jax.vmap(
            lambda a, b: jax.vmap(brax_math.rotate, in_axes=(0, None))(a, b),
            in_axes=(0, None),
        )(
            (
                ref_traj.body_positions
                - data.xpos
            )[:, self._body_idxs],
            data.qpos[3:7],
        ).flatten()

        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                # data.cinert[1:].ravel(),
                # data.cvel[1:].ravel(),
                # data.qfrc_actuator,
                track_pos_local,
                quat_dist,
                joint_dist,
                body_pos_dist_local,
            ]
        )

    def _bounded_quat_dist(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    
#class RodentMultiClip(RodentSingleClip):
    