import numpy as np
import os
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl 
from control.DMPCControl_v1 import DMPCControl 

class DMPCAviary(BaseAviary):
    """Multi-drone environment class for DMPC swarm control."""

    def __init__(self, num_drones: int=2, target_pos: np.ndarray=np.array([2.0, 0.0, 1.5]), **kwargs):
        
        self.NUM_DRONES = num_drones
        self.TARGET_POS = target_pos
        self.NP = 20
        self.D_SAFE = 0.4
        self.MAX_ACC = 5.0
        self.MAX_VEL = 2.0
        self.MAX_SAFE_VEL_PID = 0.5 
        self.MISSION_DELAY_STEPS = 5 

        super().__init__(num_drones=num_drones, ctrl_freq=30, **kwargs)

        self.DT = 1.0 / self.CTRL_FREQ
        
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.ctrl = [DSLPIDControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]

        self.dmpc_controllers = [
            DMPCControl(
                drone_id=i, 
                Np=self.NP, 
                dt=self.DT, 
                target_pos=self.TARGET_POS, 
                max_acc=self.MAX_ACC, 
                max_vel=self.MAX_VEL, 
                d_safe=self.D_SAFE,
                num_drones=self.NUM_DRONES
            ) for i in range(self.NUM_DRONES)
        ]
        
        self.prev_predicted_trajectories = np.zeros((self.NUM_DRONES, 6 * (self.NP + 1)))

    def _actionSpace(self):
        return spaces.Box(low=-1*np.ones(1), high=+1*np.ones(1), dtype=np.float32)

    def _observationSpace(self):
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0., -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0., 0., 0., 0.] for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf, np.inf, np.inf, 1., 1., 1., 1., np.pi, np.pi, np.pi, np.inf, np.inf, -np.inf, np.inf, np.inf, np.inf, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        
        all_rpms = np.zeros((self.NUM_DRONES, 4))
        
        # --- Soft Start Stabilization (1.5 seconds) ---
        if self.step_counter < (self.PYB_FREQ * 1.5): 
            
            HOVER_RPM_FOR_LIFT = self.HOVER_RPM * 1.1 
            
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                
                all_rpms[i,:] = np.array([HOVER_RPM_FOR_LIFT] * 4) 

                cur_dmpc_state = np.hstack([state[0:3], state[10:13]]) 
                self.dmpc_controllers[i].predicted_trajectory = np.tile(cur_dmpc_state, (self.NP + 1, 1))
                self.prev_predicted_trajectories[i, :] = self.dmpc_controllers[i].predicted_trajectory.flatten()
                
            return all_rpms


        # --- DMPC Control Loop ---
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            cur_dmpc_state = np.hstack([state[0:3], state[10:13]]) 
            
            # --- Target Activation Delay ---
            if self.step_counter < (self.PYB_FREQ * 1.5 + self.MISSION_DELAY_STEPS * self.PYB_STEPS_PER_CTRL):
                # Command hover
                target_pos_dmpc = state[0:3]
            else:
                # Engage mission goal
                target_pos_dmpc = self.TARGET_POS
            
            self.dmpc_controllers[i].target_pos = target_pos_dmpc.reshape(3, 1)
            # -----------------------------
            
            neighbor_trajs_flat = np.hstack([
                self.prev_predicted_trajectories[j, :] 
                for j in range(self.NUM_DRONES) if j != i
            ])
            
            # --- 1. Solve DMPC ---
            u_opt = self.dmpc_controllers[i].compute_control(cur_dmpc_state, neighbor_trajs_flat)
            
            target_dmpc_state = self.dmpc_controllers[i].predicted_trajectory[1, :]
            
            P_target = target_dmpc_state[0:3]
            V_target_DMPC = target_dmpc_state[3:6]
            
            # 2. Velocity Capping (Safety)
            pos_error = P_target - state[0:3]
            V_request = pos_error / self.DT
            
            norm = np.linalg.norm(V_request)
            if norm > self.MAX_SAFE_VEL_PID:
                V_request = (V_request / norm) * self.MAX_SAFE_VEL_PID
            
            # 3. Low-Level Control
            rpm_i, _, _ = self.ctrl[i].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=P_target,       
                target_rpy=np.array([0, 0, state[9]]), 
                target_vel=V_request       
            )
            all_rpms[i,:] = rpm_i
            
            self.prev_predicted_trajectories[i, :] = self.dmpc_controllers[i].predicted_trajectory.flatten()
            
            # Debug print
            if self.step_counter % self.PYB_STEPS_PER_CTRL == 0:
                print(f"\n--- DRONE {i} STEP {self.step_counter} (DMPC CONTROL) ---")
                print(f"Targeting: {target_pos_dmpc}")
                print(f"P_current (m): {np.round(state[0:3], 3)} | P_target (m): {np.round(P_target, 3)}")
                print(f"V_cap (m/s): {np.round(V_request, 3)} | V_DMPC (m/s): {np.round(V_target_DMPC, 3)}")
                print(f"U_opt (ACCEL, m/s^2): {np.round(u_opt, 3)}")
                print(f"RPMs: {np.round(rpm_i, 1)}")


        return all_rpms

    def _computeReward(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            dist_to_target = np.linalg.norm(self.TARGET_POS - states[i][0:3])
            ret += max(0, 2 - dist_to_target**4)
        return ret

    def _computeTerminated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist_sum = sum(np.linalg.norm(self.TARGET_POS - states[i][0:3]) for i in range(self.NUM_DRONES))
        if dist_sum < 0.01:
            return True
        else:
            return False
    
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