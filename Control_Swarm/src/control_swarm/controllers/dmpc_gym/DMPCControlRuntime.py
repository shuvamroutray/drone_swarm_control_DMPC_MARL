# DMPCControl.py (REAL-TIME OPTIMIZED VERSION)

# This is an optimized version of DMPCControl built to execute in runtime.
# Variables are converted into parameters, enabling problem creation and variable generation 
# only once. With dynamic values just updated in the already defined parameters.


"""
REAL-TIME OPTIMIZED DMPC CONTROLLER

This implementation differs from the original DMPCControl in structure,
not in mathematics.

Original Version:
    - Rebuilt decision variables, constraints, objective, and cp.Problem
      inside compute_control() at every control step.
    - Triggered full CVXPY canonicalization and matrix stuffing each time.
    - Computationally expensive for multi-drone real-time execution.

Current Version:
    - Builds the optimization problem once in __init__().
    - Uses cp.Parameter for time-varying data (initial state and neighbor trajectories).
    - Only updates parameter values at runtime.
    - Reuses the same problem structure with warm_start=True.

Advantage:
    - Avoids repeated symbolic reconstruction.
    - Significantly reduces computation time.
    - Scalable for real-time multi-agent DMPC.

The MPC formulation and solver settings remain unchanged.
Only the implementation structure has been optimized.
"""



import numpy as np
import cvxpy as cp
import time


class DMPCControl:
    """
    Real-time optimized Distributed MPC using CVXPY + OSQP.

    Structure built once.
    Only parameters updated at runtime.
    """

    def __init__(self,
                 drone_id: int,
                 Np: int,
                 dt: float,
                 target_pos: np.ndarray,
                 max_acc: float,
                 max_vel: float,
                 d_safe: float,
                 num_drones: int):

        self.id = drone_id
        self.Np = Np
        self.dt = dt
        self.num_drones = num_drones
        self.d_safe = float(d_safe)

        self.max_acc = float(max_acc)
        self.max_vel = float(max_vel)

        self.target_pos = target_pos.reshape(3,)
        self.target_state = np.hstack([self.target_pos, np.zeros(3)])

        # Cost weights
        self.Q = np.diag([1e2, 1e2, 1e2, 0.1, 0.1, 0.1])
        self.R = np.diag([1e-1, 1e-1, 1e-1])

        # ===============================
        # DECISION VARIABLES (constant structure)
        # ===============================

        self.X = cp.Variable((6, self.Np + 1))
        self.U = cp.Variable((3, self.Np))

        # ===============================
        # PARAMETERS (runtime-updated)
        # ===============================

        # Initial state parameter
        self.x0_param = cp.Parameter(6)

        # Linearization reference for self (Np+1,3)
        self.self_ref_param = cp.Parameter((self.Np + 1, 3))

        # Neighbor predicted trajectories
        self.max_neighbors = self.num_drones - 1
        self.neighbor_ref_param = cp.Parameter(
            (self.max_neighbors, self.Np + 1, 3)
        )

        # ===============================
        # BUILD PROBLEM ONCE
        # ===============================

        constraints = []

        # Initial condition
        constraints += [self.X[:, 0] == self.x0_param]

        # Dynamics + constraints
        for k in range(self.Np):

            p_k = self.X[0:3, k]
            v_k = self.X[3:6, k]
            a_k = self.U[:, k]

            p_next = self.X[0:3, k + 1]
            v_next = self.X[3:6, k + 1]

            # Double integrator dynamics
            constraints += [
                p_next == p_k + self.dt * v_k + 0.5 * self.dt**2 * a_k,
                v_next == v_k + self.dt * a_k
            ]

            # Velocity bounds
            constraints += [
                self.X[3:6, k] <= self.max_vel,
                self.X[3:6, k] >= -self.max_vel
            ]

            # Altitude constraint
            constraints += [self.X[2, k] >= 0.0]

            # Acceleration bounds
            constraints += [
                self.U[:, k] <= self.max_acc,
                self.U[:, k] >= -self.max_acc
            ]

            # Linearized collision constraints
            for j in range(self.max_neighbors):

                p_i_ref = self.self_ref_param[k + 1, :]
                p_j_ref = self.neighbor_ref_param[j, k + 1, :]

                r = p_i_ref - p_j_ref
                r_norm_sq = cp.sum_squares(r)

                # Linearized inequality:
                # 2 r^T p_i >= d_safe^2 - ||r||^2 + 2 r^T p_i_ref
                lhs = 2 * r @ self.X[0:3, k + 1]
                rhs = self.d_safe**2 - r_norm_sq + 2 * r @ p_i_ref

                constraints += [lhs >= rhs]

        # Terminal constraints
        constraints += [
            self.X[3:6, self.Np] <= self.max_vel,
            self.X[3:6, self.Np] >= -self.max_vel,
            self.X[2, self.Np] >= 0.0
        ]

        # ===============================
        # COST FUNCTION
        # ===============================

        cost = 0
        for k in range(self.Np):
            cost += cp.quad_form(self.X[:, k] - self.target_state, self.Q)
            cost += cp.quad_form(self.U[:, k], self.R)

        cost += cp.quad_form(self.X[:, self.Np] - self.target_state, self.Q)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

        # Store predicted trajectory
        self.predicted_trajectory = np.zeros((self.Np + 1, 6))

    # ==========================================================
    # REAL-TIME CONTROL LOOP
    # ==========================================================

    def compute_control(self, cur_state, neighbor_trajectories_flat):

        cur_state = np.asarray(cur_state).reshape(6,)
        self.x0_param.value = cur_state

        # Self linearization reference
        if np.allclose(self.predicted_trajectory, 0.0):
            self_ref = np.tile(cur_state[:3], (self.Np + 1, 1))
        else:
            self_ref = self.predicted_trajectory[:, 0:3]

        self.self_ref_param.value = self_ref

        # Parse neighbors
        neighbors = []
        if neighbor_trajectories_flat is not None and len(neighbor_trajectories_flat) > 0:
            flat = np.asarray(neighbor_trajectories_flat).flatten()
            per = 6 * (self.Np + 1)
            n_possible = flat.size // per
            for j in range(n_possible):
                block = flat[j*per:(j+1)*per]
                traj = block.reshape(self.Np + 1, 6)
                neighbors.append(traj[:, 0:3])

        # Fill neighbor parameter
        neighbor_param = np.zeros(
            (self.max_neighbors, self.Np + 1, 3)
        )

        for j in range(min(len(neighbors), self.max_neighbors)):
            neighbor_param[j] = neighbors[j]

        self.neighbor_ref_param.value = neighbor_param

        start = time.time()

        try:
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-3,
                eps_rel=1e-3,
                max_iter=10000
            )

            if self.problem.status not in ["optimal", "optimal_inaccurate"]:
                return np.zeros(3)
            
            print(f"[Drone {self.id}] Solve time: {time.time() - start:.4f} s")

            X_opt = self.X.value
            U_opt = self.U.value

            self.predicted_trajectory = X_opt.T

            return U_opt[:, 0]

        except Exception as e:
            print(f"[DMPC][Drone {self.id}] Solver error:", e)
            return np.zeros(3)