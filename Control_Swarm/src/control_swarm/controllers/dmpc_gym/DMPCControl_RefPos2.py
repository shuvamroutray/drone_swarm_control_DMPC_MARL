# DMPCControl_Obstacle.py
# DMPCControl with Sequential Convex Programming (first-order linearization of -2 x^T Q r)
import numpy as np
import cvxpy as cp
import math
import time

class DMPCControl:
    """
    DMPC controller implemented as a convex QP with sequential convex programming (SCP)
    to linearize the bilinear term -2 x^T Q r where r is a reference position (3,)
    that the optimizer can choose (subject to reachability).

    Public interface:
      compute_control(cur_state: (6,), neighbor_trajectories_flat: np.ndarray) -> (3,) acceleration
    """

    def __init__(self,
                 drone_id: int,
                 Np: int,
                 dt: float,
                 target_pos: np.ndarray,
                 max_acc: float,
                 max_vel: float,
                 d_safe: float,
                 num_drones: int,
                 n_scp_iters: int = 2
                 ):
        self.id = drone_id
        self.Np = int(Np)
        self.dt = float(dt)
        # store target_pos as column vector (3,1) but keep numpy float vector for defaults
        self.target_pos = np.asarray(target_pos, dtype=float).reshape(3,)
        self.max_acc = float(max_acc)
        self.max_vel = float(max_vel)
        self.d_safe = float(d_safe)
        self.num_drones = int(num_drones)

        # predicted_trajectory: (Np+1, 6) rows -> [px,py,pz,vx,vy,vz]
        self.predicted_trajectory = np.zeros((self.Np + 1, 6), dtype=float)

        # initial guess: tile current pos/vel (will be overwritten on first solve)
        self.init_guess = np.tile(np.hstack([np.zeros(3), np.zeros(3)]), (self.Np + 1, 1)).reshape(-1)

        # cost weights (tunable)
        self.Q = np.diag([1e2, 1e2, 1e2, 0.1, 0.1, 0.1])   # state cost
        self.R = np.diag([1e-1, 1e-1, 1e-1])                # control cost

        # store last reference and last predicted trajectory for linearization
        # initialize r to provided target_pos
        self._last_r = self.target_pos.copy()
        # _last_x_ref shape (Np+1,6)
        self._last_x_ref = np.tile(np.hstack([self.target_pos, np.zeros(3)]), (self.Np + 1, 1))

        # SCP iteration count (how many linearize-&-solve steps per control call)
        self.n_scp_iters = int(n_scp_iters)

    def _build_prediction_ref(self, cur_state: np.ndarray):
        """
        Build an (Np+1,6) reference trajectory for linearization.
        Prefer last predicted trajectory, otherwise tile current state.
        """
        if (self.predicted_trajectory is not None) and (not np.allclose(self.predicted_trajectory, 0.0)):
            return self.predicted_trajectory.copy()
        else:
            return np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))

    def compute_control(self, cur_state: np.ndarray, neighbor_trajectories_flat: np.ndarray):
        """
        Robust DMPC QP build for OSQP (and fallback to SCS).
        cur_state: (6,) [px,py,pz,vx,vy,vz]
        neighbor_trajectories_flat: flattened blocks of (Np+1,6) per neighbor (can include static obstacles)
        """
        cur_state = np.asarray(cur_state, dtype=float).reshape(6,)

        # Build prediction reference (existing helper)
        x_ref = self._build_prediction_ref(cur_state)  # shape (Np+1, 6)

        # Parse neighbors robustly
        neighbors = []
        if neighbor_trajectories_flat is None or (hasattr(neighbor_trajectories_flat, "__len__") and len(neighbor_trajectories_flat) == 0):
            neighbors = []
        else:
            flat = np.asarray(neighbor_trajectories_flat, dtype=float).flatten()
            per = 6 * (self.Np + 1)
            n_possible = max(0, flat.size // per)
            for j in range(n_possible):
                start = j * per
                block = flat[start:start + per]
                traj = block.reshape(self.Np + 1, 6)
                neighbors.append(traj)

        # CVXPY variables
        X = cp.Variable((6, self.Np + 1))   # state (px,py,pz,vx,vy,vz) columns
        U = cp.Variable((3, self.Np))       # accelerations
        constraints = []

        # initial condition
        constraints += [X[:, 0] == cur_state]

        # Slack for collisions (if any)
        num_collision_constraints = len(neighbors) * self.Np
        if num_collision_constraints > 0:
            s = cp.Variable(num_collision_constraints)  # slack vector
            constraints += [s >= 0]
        else:
            s = None

        # Ensure Q and R are numeric and Q symmetric
        Q = np.array(self.Q, dtype=float)
        Q = 0.5 * (Q + Q.T)
        # For numerical robustness use sqrt of diagonal (works if Q is diagonal or approx diag)
        if np.allclose(Q, np.diag(np.diag(Q))):
            sqrtQ_vec = np.sqrt(np.diag(Q))
        else:
            # fallback: take absolute diag sqrt (safe heuristic)
            sqrtQ_vec = np.sqrt(np.abs(np.diag(Q)))

        R = np.array(self.R, dtype=float)
        if np.allclose(R, np.diag(np.diag(R))):
            sqrtR_vec = np.sqrt(np.diag(R))
        else:
            sqrtR_vec = np.sqrt(np.abs(np.diag(R)))

        coll_idx = 0
        for k in range(self.Np):
            p_k = X[0:3, k]
            v_k = X[3:6, k]
            a_k = U[:, k]
            p_next = X[0:3, k + 1]
            v_next = X[3:6, k + 1]

            # dynamics (affine)
            constraints += [
                p_next == p_k + self.dt * v_k + (self.dt ** 2 / 2.0) * a_k,
                v_next == v_k + self.dt * a_k
            ]

            # velocity bounds elementwise (affine)
            constraints += [X[3, k] <= self.max_vel, X[3, k] >= -self.max_vel]
            constraints += [X[4, k] <= self.max_vel, X[4, k] >= -self.max_vel]
            constraints += [X[5, k] <= self.max_vel, X[5, k] >= -self.max_vel]

            # z bounds (lower + optional upper for safety)
            constraints += [X[2, k] >= 0.0]
            # optional upper bound - tune or expose as parameter
            constraints += [X[2, k] <= 3.0]

            # acceleration bounds (optionally asymmetric)
            constraints += [U[0, k] <= self.max_acc, U[0, k] >= -self.max_acc]
            constraints += [U[1, k] <= self.max_acc, U[1, k] >= -self.max_acc]
            constraints += [U[2, k] <= min(self.max_acc, 2.0), U[2, k] >= -self.max_acc]

            # linearized collision constraints (affine)
            if len(neighbors) > 0:
                p_i_ref = x_ref[k + 1, 0:3].astype(float)
                for traj_j in neighbors:
                    p_j_ref = traj_j[k + 1, 0:3].astype(float)
                    r = (p_i_ref - p_j_ref).astype(float)  # pure numpy vector
                    r_norm_sq = float(np.dot(r, r))
                    if r_norm_sq < 1e-8:
                        # avoid degenerate direction
                        r = np.array([1e-3, 0.0, 0.0], dtype=float)
                        r_norm_sq = float(np.dot(r, r))
                    # make sure lhs is a scalar affine expression
                    # r is numpy vector, X[0:3, k+1] is cvx variable vector -> inner product is affine scalar
                    lhs = 2.0 * (r @ X[0:3, k + 1])   # scalar cvx expression
                    rhs = (self.d_safe ** 2 - r_norm_sq + 2.0 * (r @ p_i_ref))
                    if s is not None:
                        constraints += [lhs + s[coll_idx] >= rhs]
                    else:
                        constraints += [lhs >= rhs]
                    coll_idx += 1

        # terminal constraints
        constraints += [X[3, self.Np] <= self.max_vel, X[3, self.Np] >= -self.max_vel]
        constraints += [X[4, self.Np] <= self.max_vel, X[4, self.Np] >= -self.max_vel]
        constraints += [X[5, self.Np] <= self.max_vel, X[5, self.Np] >= -self.max_vel]
        constraints += [X[2, self.Np] >= 0.0]
        constraints += [X[2, self.Np] <= 3.0]

        # objective built as sum of squared (weighted) terms to ensure QP form
        obj = 0
        target_state_vec = np.hstack([self.target_pos.reshape(3,), np.zeros(3,)]).astype(float)

        for k in range(self.Np):
            xk = X[:, k]
            uk = U[:, k]

            # weighted state error using sqrtQ_vec -> sum_squares(sqrtQ_vec * (xk - target))
            diff = xk - target_state_vec
            if sqrtQ_vec.size == diff.shape[0]:
                # elementwise multiply then sum squares
                obj += cp.sum_squares(cp.multiply(sqrtQ_vec.reshape(-1, 1), diff))
            else:
                # fallback (unweighted)
                obj += cp.sum_squares(diff)

            # control cost: diagonal R assumption
            if sqrtR_vec.size == uk.shape[0]:
                obj += cp.sum_squares(cp.multiply(sqrtR_vec.reshape(-1, 1), uk))
            else:
                obj += cp.sum_squares(uk)

        # terminal cost
        xN = X[:, self.Np]
        diffN = xN - target_state_vec
        if sqrtQ_vec.size == diffN.shape[0]:
            obj += cp.sum_squares(cp.multiply(sqrtQ_vec.reshape(-1, 1), diffN))
        else:
            obj += cp.sum_squares(diffN)

        # slack penalty (heavy)
        if s is not None:
            w_slack = 1e5
            obj += w_slack * cp.sum_squares(s)

        problem = cp.Problem(cp.Minimize(obj), constraints)

        # Try to solve with OSQP first (fast QP). If CVXPY cannot reduce to QP
        # try SCS (conic) as a fallback (install scs via pip if you want this fallback).
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                        eps_abs=1e-3, eps_rel=1e-3, max_iter=10000)
        except Exception as e:
            # Try a conic solver if available (SCS). This is slower but works if reduction failed.
            try:
                print(f"[DMPC][Drone {self.id}] OSQP solve failed ({e}), trying SCS fallback...")
                problem.solve(solver=cp.SCS, verbose=False, eps=1e-3, max_iters=2500)
            except Exception as e2:
                print(f"[DMPC][Drone {self.id}] SCS fallback also failed: {e2}")
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

        # Check status
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            # fallback: safe hover
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)

        # extract solution
        X_opt = X.value
        U_opt = U.value
        if X_opt is None or U_opt is None:
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)

        self.predicted_trajectory = X_opt.T
        u0 = U_opt[:, 0].reshape(3,)
        self._last_x_ref = X_opt.T.copy()
        return u0
