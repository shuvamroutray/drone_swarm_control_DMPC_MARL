# DMPCAviary_Obstacle_plots_Noise.py
import numpy as np
import os
import time
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.DMPCControl import DMPCControl
import random


class DMPCAviary(BaseAviary):
    """
    DMPCAviary with *soft* noise injection:
      - sensor noise (position & velocity) passed to DMPC (controller sees noisy state)
      - optional actuation noise (small additive RPM noise)
    This is much more stable than applying large external forces in pybullet and stresses the closed-loop control.
    """

    def __init__(self, num_drones: int = 2, target_pos: np.ndarray = np.array([2.0, 0.0, 1.5]),
                 obstacles=None, noise_sensor_enable: bool = True, noise_sensor_pos_std: float = 0.005,
                 noise_sensor_vel_std: float = 0.1, noise_actuation_enable: bool = False,
                 noise_actuation_rpm_std: float = 5.0, turbulence_params: dict = None, **kwargs):

        # number of drones
        self.NUM_DRONES = int(num_drones)

        # obstacles list
        if obstacles is None:
            self.OBSTACLE_POSITIONS = []
        else:
            self.OBSTACLE_POSITIONS = [np.asarray(o, dtype=float).reshape(3,) for o in obstacles]

        # target(s)
        target_arr = np.array(target_pos, dtype=float)
        if target_arr.shape == (3,):
            self.TARGET_POS = np.tile(target_arr.reshape(1, 3), (self.NUM_DRONES, 1))
        elif target_arr.shape == (self.NUM_DRONES, 3):
            self.TARGET_POS = target_arr.copy()
        else:
            raise ValueError("target_pos must be shape (3,) or (NUM_DRONES,3)")

        # DMPC params
        self.NP = 10
        self.D_SAFE = 0.5
        self.MAX_ACC = 5.0
        self.MAX_VEL = 2.0
        self.MAX_SAFE_VEL_PID = 0.4
        self.MISSION_DELAY_STEPS = 5

        # noise config (sensor + actuation)
        self.noise_sensor_enable = bool(noise_sensor_enable)
        self.noise_sensor_pos_std = float(noise_sensor_pos_std)
        self.noise_sensor_vel_std = float(noise_sensor_vel_std)
        self.noise_actuation_enable = bool(noise_actuation_enable)
        self.noise_actuation_rpm_std = float(noise_actuation_rpm_std)

        # logs
        self._noise_log = []       # appended (NUM_DRONES, 6) per control step: [pos_noise(3), vel_noise(3)]
        self._act_noise_log = []   # appended (NUM_DRONES, 4) RPM noise per control step (even when zeros)

        # call BaseAviary initializer (this sets CTRL_FREQ, PYB_FREQ, etc.)
        super().__init__(num_drones=self.NUM_DRONES, ctrl_freq=30, **kwargs)

        # derived time step
        self.DT = 1.0 / self.CTRL_FREQ

        # MKL safe env var
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        # low level PID controllers
        self.ctrl = [DSLPIDControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]

        # DMPC controllers
        self.dmpc_controllers = [
            DMPCControl(
                drone_id=i,
                Np=self.NP,
                dt=self.DT,
                target_pos=self.TARGET_POS[i].reshape(3, 1),
                max_acc=self.MAX_ACC,
                max_vel=self.MAX_VEL,
                d_safe=self.D_SAFE,
                num_drones=self.NUM_DRONES,
            )
            for i in range(self.NUM_DRONES)
        ]

        self.prev_predicted_trajectories = np.zeros((self.NUM_DRONES, 6 * (self.NP + 1)))

        # create obstacle visuals (if any)
        self.OBSTACLE_IDS = []
        if len(self.OBSTACLE_POSITIONS) > 0:
            for idx, pos in enumerate(self.OBSTACLE_POSITIONS):
                pos = np.asarray(pos).reshape(3,)
                if pos[2] < 0.05:
                    pos[2] = 0.05
                radius = max(0.06, 0.5 * self.D_SAFE)
                rgba = [1.0, 0.2, 0.2, 1.0]
                visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                    radius=float(radius),
                                                    rgbaColor=rgba,
                                                    physicsClientId=self.CLIENT)
                colShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                    radius=float(radius),
                                                    physicsClientId=self.CLIENT)
                obs_id = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=colShapeId,
                                           baseVisualShapeIndex=visualShapeId,
                                           basePosition=pos.tolist(),
                                           useMaximalCoordinates=True,
                                           physicsClientId=self.CLIENT)
                self.OBSTACLE_IDS.append(obs_id)
                try:
                    label_pos = (pos + np.array([0.0, 0.0, radius + 0.05])).tolist()
                    p.addUserDebugText(f"OBS{idx}", label_pos,
                                       textColorRGB=[1, 1, 1],
                                       textSize=1.2,
                                       lifeTime=0,
                                       physicsClientId=self.CLIENT)
                except Exception:
                    pass

        # target markers
        self.target_marker_ids = []
        for i in range(self.NUM_DRONES):
            sphere_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.05,
                rgbaColor=[1.0, 0.0, 0.0, 1.0],
                physicsClientId=self.CLIENT,
            )
            marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sphere_visual,
                basePosition=[0, 0, -10],
                useMaximalCoordinates=True,
                physicsClientId=self.CLIENT,
            )
            try:
                p.setCollisionFilterGroupMask(marker_id, -1, 0, 0, physicsClientId=self.CLIENT)
            except Exception:
                pass
            self.target_marker_ids.append(marker_id)

        # env-level solver instrumentation lists
        self._dmpc_solve_walltimes = [[] for _ in range(self.NUM_DRONES)]
        self._dmpc_solve_cpu_times = [[] for _ in range(self.NUM_DRONES)]
        self._dmpc_solve_status = [[] for _ in range(self.NUM_DRONES)]
        self._dmpc_solve_iters = [[] for _ in range(self.NUM_DRONES)]

    ################################################################################
    def _actionSpace(self):
        return spaces.Box(low=-1 * np.ones(1), high=+1 * np.ones(1), dtype=np.float32)

    def _observationSpace(self):
        obs_lower_bound = np.array(
            [
                [
                    -np.inf, -np.inf, 0.0,
                    -1.0, -1.0, -1.0,
                    -1.0, -np.pi, -np.pi, -np.pi,
                    -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                    0.0, 0.0, 0.0, 0.0,
                ] for _ in range(self.NUM_DRONES)
            ]
        )
        obs_upper_bound = np.array(
            [
                [
                    np.inf, np.inf, np.inf,
                    1.0, 1.0, 1.0,
                    1.0, np.pi, np.pi, np.pi,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                    self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM,
                ] for _ in range(self.NUM_DRONES)
            ]
        )
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################
    def _preprocessAction(self, action):
        all_rpms = np.zeros((self.NUM_DRONES, 4))

        # soft-start stabilization first 1.5 s
        soft_start_steps = int(self.PYB_FREQ * 1.5)
        if self.step_counter < soft_start_steps:
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                cur_dmpc_state = np.hstack([state[0:3], state[10:13]])
                self.dmpc_controllers[i].predicted_trajectory = np.tile(cur_dmpc_state, (self.NP + 1, 1))
                self.prev_predicted_trajectories[i, :] = self.dmpc_controllers[i].predicted_trajectory.flatten()

                hover_z = max(state[2], self.INIT_XYZS[i, 2]) + 0.1
                target_pos = np.array([state[0], state[1], hover_z])

                rpm_i, _, _ = self.ctrl[i].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=target_pos,
                    target_rpy=np.array([0.0, 0.0, state[9]]),
                    target_vel=np.zeros(3),
                )

                all_rpms[i, :] = np.clip(rpm_i, 0.0, self.MAX_RPM)

                try:
                    if hasattr(self, "target_marker_ids") and len(self.target_marker_ids) > i:
                        p.resetBasePositionAndOrientation(
                            self.target_marker_ids[i],
                            target_pos.tolist(),
                            [0, 0, 0, 1],
                            physicsClientId=self.CLIENT,
                        )
                except Exception:
                    pass

            # log zeros for noise logs so lengths align
            self._noise_log.append(np.zeros((self.NUM_DRONES, 6)))
            self._act_noise_log.append(np.zeros((self.NUM_DRONES, 4)))
            return all_rpms

        # DMPC loop
        # We'll collect per-step noise arrays to append as rows for all drones
        per_step_noise = np.zeros((self.NUM_DRONES, 6))
        per_step_act_noise = np.zeros((self.NUM_DRONES, 4))

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            cur_dmpc_state = np.hstack([state[0:3], state[10:13]])  # ground truth

            # measurement noise for DMPC input
            if self.noise_sensor_enable:
                pos_noise = np.random.randn(3) * self.noise_sensor_pos_std
                vel_noise = np.random.randn(3) * self.noise_sensor_vel_std
            else:
                pos_noise = np.zeros(3)
                vel_noise = np.zeros(3)

            noisy_cur_dmpc_state = np.hstack([state[0:3] + pos_noise, state[10:13] + vel_noise])

            per_step_noise[i, :] = np.hstack([pos_noise, vel_noise])

            # target activation delay / assigned target
            if self.step_counter < (soft_start_steps + self.MISSION_DELAY_STEPS * self.PYB_STEPS_PER_CTRL):
                target_pos_dmpc = state[0:3].copy()
            else:
                target_pos_dmpc = self.TARGET_POS[i].copy()

            self.dmpc_controllers[i].target_pos = target_pos_dmpc.reshape(3, 1)

            try:
                if hasattr(self, "target_marker_ids") and len(self.target_marker_ids) > i:
                    p.resetBasePositionAndOrientation(
                        self.target_marker_ids[i],
                        target_pos_dmpc.tolist(),
                        [0, 0, 0, 1],
                        physicsClientId=self.CLIENT,
                    )
            except Exception:
                pass

            # neighbor preds blocks (no changes)
            neighbor_blocks = [
                self.prev_predicted_trajectories[j, :]
                for j in range(self.NUM_DRONES) if j != i
            ]
            for obs_pos in self.OBSTACLE_POSITIONS:
                block = np.tile(np.hstack([obs_pos.reshape(3,), np.zeros(3,)]), (self.NP + 1, 1)).flatten()
                neighbor_blocks.append(block)
            if len(neighbor_blocks) > 0:
                neighbor_trajs_flat = np.hstack(neighbor_blocks)
            else:
                neighbor_trajs_flat = np.array([])

            # solve DMPC using noisy measurement (this exercises controller robustness)
            t0 = time.time()
            cpu0 = time.process_time()
            u_opt = self.dmpc_controllers[i].compute_control(noisy_cur_dmpc_state, neighbor_trajs_flat)
            t_wall = time.time() - t0
            t_cpu = time.process_time() - cpu0

            # store env-level instrumentation
            try:
                self._dmpc_solve_walltimes[i].append(float(t_wall))
            except Exception:
                self._dmpc_solve_walltimes[i].append(np.nan)
            try:
                self._dmpc_solve_cpu_times[i].append(float(t_cpu))
            except Exception:
                self._dmpc_solve_cpu_times[i].append(np.nan)

            ctrl = self.dmpc_controllers[i]
            self._dmpc_solve_status[i].append(getattr(ctrl, "solver_status", [None])[-1] if getattr(ctrl, "solver_status", None) else None)
            self._dmpc_solve_iters[i].append(getattr(ctrl, "solver_iters", [np.nan])[-1] if getattr(ctrl, "solver_iters", None) else np.nan)

            # DMPC predicted next state (controller stored predicted_trajectory)
            target_dmpc_state = self.dmpc_controllers[i].predicted_trajectory[1, :]
            P_target = target_dmpc_state[0:3]
            V_target_DMPC = target_dmpc_state[3:6]

            pos_error = P_target - state[0:3]
            V_request = V_target_DMPC.copy()
            if np.linalg.norm(V_request) < 1e-6:
                V_request = pos_error / self.DT

            norm = np.linalg.norm(V_request)
            if norm > self.MAX_SAFE_VEL_PID:
                V_request = (V_request / norm) * self.MAX_SAFE_VEL_PID

            rpm_i, _, _ = self.ctrl[i].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=P_target,
                target_rpy=np.array([0.0, 0.0, state[9]]),
                target_vel=V_request,
            )

            # actuation noise
            if self.noise_actuation_enable:
                rpm_noise = np.random.randn(4) * self.noise_actuation_rpm_std
            else:
                rpm_noise = np.zeros(4)

            per_step_act_noise[i, :] = rpm_noise

            noisy_rpm = rpm_i + rpm_noise
            all_rpms[i, :] = np.clip(noisy_rpm, 0.0, self.MAX_RPM)

            self.prev_predicted_trajectories[i, :] = self.dmpc_controllers[i].predicted_trajectory.flatten()

            if self.step_counter % self.PYB_STEPS_PER_CTRL == 0:
                print(f"\n--- DRONE {i} STEP {self.step_counter} (DMPC CONTROL) ---")
                print(f"Targeting (DMPC target): {np.round(target_pos_dmpc, 3)}")
                print(f"P_current (m): {np.round(state[0:3], 3)} | P_target (m): {np.round(P_target, 3)}")
                print(f"V_cap (m/s): {np.round(V_request, 3)} | V_DMPC (m/s): {np.round(V_target_DMPC, 3)}")
                print(f"U_opt (ACCEL, m/s^2): {np.round(u_opt, 3)}")
                print(f"RPMs: {np.round(all_rpms[i, :], 1)}")

        # append noise rows for this control step
        # shape: (NUM_DRONES, 6) and (NUM_DRONES, 4)
        self._noise_log.append(per_step_noise.copy())
        self._act_noise_log.append(per_step_act_noise.copy())

        return all_rpms

    ################################################################################
    def _computeReward(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0.0
        for i in range(self.NUM_DRONES):
            dist_to_target = np.linalg.norm(self.TARGET_POS[i] - states[i][0:3])
            ret += max(0.0, 2.0 - dist_to_target ** 4)
        return ret

    def _computeTerminated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist_sum = sum(np.linalg.norm(self.TARGET_POS[i] - states[i][0:3]) for i in range(self.NUM_DRONES))
        return bool(dist_sum < 0.01)

    def _computeTruncated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 6.0 or abs(states[i][1]) > 6.0 or states[i][2] > 6.0
                or abs(states[i][7]) > 2.5 or abs(states[i][8]) > 2.5):
                return True
        if self.step_counter / self.PYB_FREQ > 120:
            return True
        return False

    def _computeInfo(self):
        return {"answer": 42}
