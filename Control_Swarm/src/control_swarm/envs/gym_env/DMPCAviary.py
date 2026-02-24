# DMPCAviary.py
import numpy as np
import os
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from control_swarm.controllers.dmpc_gym.DMPCControl import DMPCControl



class DMPCAviary(BaseAviary):
    """Multi-drone environment class for DMPC swarm control."""

    def __init__(self, num_drones: int = 2, target_pos: np.ndarray = np.array([2.0, 0.0, 1.5]), **kwargs):

        # --- Handle per-drone or single target input ---
        self.NUM_DRONES = num_drones
        target_arr = np.array(target_pos, dtype=float)
        if target_arr.shape == (3,):
            # same target for all drones
            self.TARGET_POS = np.tile(target_arr.reshape(1, 3), (self.NUM_DRONES, 1))
        elif target_arr.shape == (self.NUM_DRONES, 3):
            self.TARGET_POS = target_arr.copy()
        else:
            raise ValueError("target_pos must be shape (3,) or (NUM_DRONES,3)")

        # DMPC / mission parameters (tunable)
        # self.NP = 20
        # self.D_SAFE = 0.4
        # self.MAX_ACC = 5.0
        # self.MAX_VEL = 2.0
        # self.MAX_SAFE_VEL_PID = 0.5
        # self.MISSION_DELAY_STEPS = 5

        self.NP = 50
        self.D_SAFE = 0.5
        self.MAX_ACC = 5.0
        self.MAX_VEL = 2.0
        self.MAX_SAFE_VEL_PID = 0.4
        self.MISSION_DELAY_STEPS = 5

        # call BaseAviary initializer (this sets CTRL_FREQ, PYB_FREQ, etc.)
        super().__init__(num_drones=num_drones, ctrl_freq=30, **kwargs)

        # Derived time step for DMPC
        self.DT = 1.0 / self.CTRL_FREQ

        # Prevent some MKL/OMP crashes on certain systems (keeps consistent with previous edits)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        # Low level PID controllers (one per drone)
        self.ctrl = [DSLPIDControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]

        # DMPC controllers (one per drone)
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

        # previous predicted trajectories flattened for neighbor sharing
        self.prev_predicted_trajectories = np.zeros((self.NUM_DRONES, 6 * (self.NP + 1)))

        # --- Create visual target markers (visual only, no collision) ---
        # small sphere per drone to indicate its DMPC target
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
                basePosition=[0, 0, -10],  # start below ground; moved later by update
                useMaximalCoordinates=True,
                physicsClientId=self.CLIENT,
            )
            # disable collisions for marker
            try:
                p.setCollisionFilterGroupMask(marker_id, -1, 0, 0, physicsClientId=self.CLIENT)
            except Exception:
                # some pybullet versions have different signatures; ignore if it fails
                pass
            self.target_marker_ids.append(marker_id)

    ################################################################################
    def _actionSpace(self):
        return spaces.Box(low=-1 * np.ones(1), high=+1 * np.ones(1), dtype=np.float32)

    def _observationSpace(self):
        obs_lower_bound = np.array(
            [
                [
                    -np.inf,
                    -np.inf,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -np.pi,
                    -np.pi,
                    -np.pi,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                for _ in range(self.NUM_DRONES)
            ]
        )
        obs_upper_bound = np.array(
            [
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.pi,
                    np.pi,
                    np.pi,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    self.MAX_RPM,
                    self.MAX_RPM,
                    self.MAX_RPM,
                    self.MAX_RPM,
                ]
                for _ in range(self.NUM_DRONES)
            ]
        )
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################
    def _preprocessAction(self, action):

        all_rpms = np.zeros((self.NUM_DRONES, 4))

        # --- Soft Start Stabilization (first 1.5 s) ---
        soft_start_steps = int(self.PYB_FREQ * 1.5)
        if self.step_counter < soft_start_steps:
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                cur_dmpc_state = np.hstack([state[0:3], state[10:13]])

                # Initialize DMPC predicted trajectory with current state (safe initial guess)
                self.dmpc_controllers[i].predicted_trajectory = np.tile(cur_dmpc_state, (self.NP + 1, 1))
                self.prev_predicted_trajectories[i, :] = self.dmpc_controllers[i].predicted_trajectory.flatten()

                # closed-loop hover target: small gentle lift above current/initial z
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

                # clip to safe RPM range
                all_rpms[i, :] = np.clip(rpm_i, 0.0, self.MAX_RPM)

                # optionally show marker at hover target (in GUI)
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

            return all_rpms

        # --- DMPC Control Loop ---
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            cur_dmpc_state = np.hstack([state[0:3], state[10:13]])

            # --- Target Activation Delay ---
            if self.step_counter < (soft_start_steps + self.MISSION_DELAY_STEPS * self.PYB_STEPS_PER_CTRL):
                # hold current hover position until mission delay passes
                target_pos_dmpc = state[0:3].copy()
            else:
                # per-drone target
                target_pos_dmpc = self.TARGET_POS[i].copy()

            # set DMPC target for this drone (DMPC controller reads this inside compute_control)
            self.dmpc_controllers[i].target_pos = target_pos_dmpc.reshape(3, 1)

            # update visual marker to this target (if GUI enabled)
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

            # gather neighbor flattened predicted trajectories
            neighbor_trajs_flat = np.hstack(
                [self.prev_predicted_trajectories[j, :] for j in range(self.NUM_DRONES) if j != i]
            ) if self.NUM_DRONES > 1 else np.array([])

            # --- 1. Solve DMPC (returns acceleration commands) ---
            u_opt = self.dmpc_controllers[i].compute_control(cur_dmpc_state, neighbor_trajs_flat)

            # predicted next state from DMPC
            target_dmpc_state = self.dmpc_controllers[i].predicted_trajectory[1, :]
            P_target = target_dmpc_state[0:3]
            V_target_DMPC = target_dmpc_state[3:6]

            # 2. Velocity request: prefer DMPC-predicted velocity if it's meaningful,
            # otherwise derive from position error
            pos_error = P_target - state[0:3]
            V_request = V_target_DMPC.copy()
            if np.linalg.norm(V_request) < 1e-6:
                # fallback to position-derived request
                V_request = pos_error / self.DT

            # cap velocity safely for PID
            norm = np.linalg.norm(V_request)
            if norm > self.MAX_SAFE_VEL_PID:
                V_request = (V_request / norm) * self.MAX_SAFE_VEL_PID

            # 3. Low-level PID controller: use DMPC target pos + velocity request
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

            # safety clip RPMs
            all_rpms[i, :] = np.clip(rpm_i, 0.0, self.MAX_RPM)

            # store predicted trajectory for neighbor sharing next step
            self.prev_predicted_trajectories[i, :] = self.dmpc_controllers[i].predicted_trajectory.flatten()

            # Debug prints (every control step)
            if self.step_counter % self.PYB_STEPS_PER_CTRL == 0:
                print(f"\n--- DRONE {i} STEP {self.step_counter} (DMPC CONTROL) ---")
                print(f"Targeting (DMPC target): {np.round(target_pos_dmpc, 3)}")
                print(f"P_current (m): {np.round(state[0:3], 3)} | P_target (m): {np.round(P_target, 3)}")
                print(f"V_cap (m/s): {np.round(V_request, 3)} | V_DMPC (m/s): {np.round(V_target_DMPC, 3)}")
                print(f"U_opt (ACCEL, m/s^2): {np.round(u_opt, 3)}")
                print(f"RPMs: {np.round(all_rpms[i, :], 1)}")

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
            if (abs(states[i][0]) > 3.0 or abs(states[i][1]) > 3.0 or states[i][2] > 3.0
                or abs(states[i][7]) > 1.0 or abs(states[i][8]) > 1.0):
                return True
        if self.step_counter / self.PYB_FREQ > 20:
            return True
        return False

    def _computeInfo(self):
        return {"answer": 42}
