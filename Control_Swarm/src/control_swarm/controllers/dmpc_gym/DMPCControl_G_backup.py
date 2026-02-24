import numpy as np
from casadi import *
import time

class DMPCControl:
    """
    Distributed Model Predictive Control (DMPC) implementation with stability checks.
    """
    def __init__(self,
                 drone_id: int,   #Drone ID
                 Np: int,  #Prediction Horizon
                 dt: float, #Control Time Step
                 target_pos: np.ndarray,
                 max_acc: float,
                 max_vel: float,
                 d_safe: float,   #Minimum safe distance
                 num_drones: int
                 ):
        self.id = drone_id
        self.Np = Np
        self.dt = dt
        self.target_pos = target_pos.reshape(3, 1)
        self.d_safe = d_safe
        self.num_drones = num_drones
        
        self.max_acc = max_acc
        self.max_vel = max_vel
        
        self.predicted_trajectory = np.zeros((Np + 1, 6))
        self.init_guess = np.zeros((6 * (self.Np + 1) + 3 * self.Np))
        
        self.solver = None 
        self._setup_solver()

    def _setup_solver(self):
        
        max_acc = self.max_acc
        max_vel = self.max_vel
        
        X = MX.sym('X', 6, self.Np + 1)
        U = MX.sym('U', 3, self.Np)
        
        self.neighbor_traj_size = 6 * (self.Np + 1)   #The previous predicted state trajectory of neighboring drones
        P = MX.sym('P', 6 + (self.num_drones - 1) * self.neighbor_traj_size) #Values passed from the simulation, held constant during optimization
        X0 = P[:6]
        
        obj = 0
        g = []  #Initializes the list for all symbolic constraint expressions (equalities and inequalities).
        self.lbx = []
        self.ubx = []
        self.lbg = []
        self.ubg = []
        
        Q = diag([1e2, 1e2, 1e2, 0.1, 0.1, 0.1])
        R = diag([1e-1, 1e-1, 1e-1])
        target_state = vertcat(self.target_pos, DM.zeros(3, 1))
        
        g.append(X[:, 0] - X0)
        self.lbg.extend([0.0] * 6)
        self.ubg.extend([0.0] * 6)
        
        # Dynamic and Collision Constraints (Builds g/lbg/ubg)
        for k in range(self.Np):
            p_next = X[:3, k] + self.dt * X[3:, k] + (self.dt**2/2) * U[:, k]
            v_next = X[3:, k] + self.dt * U[:, k]
            g.append(X[:6, k+1] - vertcat(p_next, v_next))
            self.lbg.extend([0.0] * 6) 
            self.ubg.extend([0.0] * 6) 
            
            p_i_k = X[:3, k+1]
            neighbor_index = 0
            for j in range(self.num_drones):
                if j == self.id: continue
                
                traj_start_idx = 6 + neighbor_index * self.neighbor_traj_size
                p_j_k_pred = P[traj_start_idx + (k+1)*6 : traj_start_idx + (k+1)*6 + 3] 
                collision_dist_sq = dot(p_i_k - p_j_k_pred, p_i_k - p_j_k_pred)
                
                g.append(collision_dist_sq) 
                self.lbg.append(self.d_safe**2) 
                self.ubg.append(np.inf)
                neighbor_index += 1
                
            obj += (X[:6, k] - target_state).T @ Q @ (X[:6, k] - target_state)
            obj += U[:, k].T @ R @ U[:, k]

        obj += (X[:6, self.Np] - target_state).T @ Q @ (X[:6, self.Np] - target_state)

        # Decision Variable Bounds (Builds lbx/ubx)
        for k in range(self.Np + 1):
            self.lbx.extend([-np.inf, -np.inf, 0.0])
            self.ubx.extend([+np.inf, +np.inf, +np.inf])
            self.lbx.extend([-max_vel, -max_vel, -max_vel])
            self.ubx.extend([+max_vel, +max_vel, +max_vel])

        for k in range(self.Np):
            self.lbx.extend([-max_acc, -max_acc, -max_acc])
            self.ubx.extend([+max_acc, +max_acc, +max_acc])

        # Setup NLP Solver
        OPT_variables = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))) 
        
        nlp = {'f': obj, 'x': OPT_variables, 'g': vertcat(*g), 'p': P}
        opts = {'ipopt': {'max_iter': 100, 'print_level': 0, 'acceptable_tol':1e-3, 'acceptable_obj_change_tol':1e-3}, 'print_time': 0}
        
        try:
            self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        except Exception as e:
            print(f"\nFATAL ERROR: Drone {self.id} CasADi nlpsol failed to initialize the solver! Error: {e}")


    def compute_control(self, cur_state: np.ndarray, neighbor_trajectories_flat: np.ndarray):
        
        if self.solver is None:
            self.predicted_trajectory[:] = cur_state
            return np.zeros(3)

        P_val = np.hstack([cur_state, neighbor_trajectories_flat])
        
        try:
            sol = self.solver(x0=self.init_guess, p=P_val, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
            
            # 🏆 CRITICAL FIX: CHECK SOLVER CONVERGENCE (Sanity Check)
            if self.solver.stats()['success'] == False:
                # If solver fails to converge (Infeasible_Problem_Detected), force a safe hover.
                self.predicted_trajectory[:] = cur_state # Set target to current state
                return np.zeros(3)

            sol_x = sol['x'].full().flatten()
            
            u_start_idx = 6 * (self.Np + 1)
            u_opt = sol_x[u_start_idx: u_start_idx + 3] 
            
            x_opt_flat = sol_x[:u_start_idx]
            self.predicted_trajectory = x_opt_flat.reshape((self.Np + 1, 6))
            
            self._update_init_guess(sol_x)

            return u_opt
            
        except Exception as e:
            # Catch Python runtime errors
            self.predicted_trajectory[:] = cur_state
            return np.zeros(3)

    def _update_init_guess(self, sol_x: np.ndarray):
        
        x_opt_flat = sol_x[:6 * (self.Np + 1)]
        x_opt_shifted = np.concatenate([x_opt_flat[6:], np.zeros(6)])
        
        u_start_idx = 6 * (self.Np + 1)
        u_opt_flat = sol_x[u_start_idx:]
        u_opt_shifted = np.concatenate([u_opt_flat[3:], np.zeros(3)])
        
        self.init_guess = np.hstack([x_opt_shifted, u_opt_shifted])