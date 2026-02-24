# DMPCControl.py
import numpy as np
import time
import cvxpy as cp


class DMPCControl:
    """
    Distributed Model Predictive Control (DMPC) implemented as a QP using CVXPY/OSQP.

    - Linear double-integrator dynamics:
        p_{k+1} = p_k + dt * v_k + 0.5 * dt^2 * a_k
        v_{k+1} = v_k + dt * a_k
      where control is acceleration a_k (3-vector).

    - Quadratic cost in states and controls.
    - Box constraints on velocities and accelerations.
    - Linearized collision avoidance constraints based on neighbor predicted trajectories.
    """

    def __init__(self,
                 drone_id: int,
                 Np: int,
                 dt: float,
                 target_pos: np.ndarray,
                 max_acc: float,
                 max_vel: float,
                 d_safe: float,
                 num_drones: int
                 ):
        self.id = drone_id
        self.Np = Np
        self.dt = dt
        self.target_pos = target_pos.reshape(3, 1).astype(float)
        self.d_safe = float(d_safe)
        self.num_drones = num_drones

        self.max_acc = float(max_acc)
        self.max_vel = float(max_vel)

        # predicted_trajectory: (Np+1, 6) rows -> [px,py,pz,vx,vy,vz]
        self.predicted_trajectory = np.zeros((Np + 1, 6))
        # initialize predicted trajectory with zeros (or will be set on first call)
        self.init_guess = np.tile(np.hstack([np.zeros(3), np.zeros(3)]), (Np + 1, 1)).reshape(-1)

        # cost weights (tunable)
        self.Q = np.diag([1e2, 1e2, 1e2, 0.1, 0.1, 0.1])   # state cost
        self.R = np.diag([1e-1, 1e-1, 1e-1])                # control cost

        # For linearized collisions we need a previous reference trajectory for self (p_i_ref)
        # initialize as current pos repeated
        self._last_x_ref = None

    def _build_prediction_ref(self, cur_state: np.ndarray):
        """
        Returns a reference state trajectory (Np+1, 6) used to linearize collisions.
        If we have self.predicted_trajectory from previous solves, use that; otherwise tile current state.
        cur_state: array-like of length 6 (p (3), v (3))
        """
        if (self.predicted_trajectory is not None) and (not np.allclose(self.predicted_trajectory, 0.0)):
            # use previous predicted traj as reference
            return self.predicted_trajectory.copy()
        else:
            # tile current state
            return np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))

    def compute_control(self, cur_state: np.ndarray, neighbor_trajectories_flat: np.ndarray):
        """
        Solve a QP using CVXPY/OSQP.

        cur_state: (6,) array -> [px,py,pz,vx,vy,vz]
        neighbor_trajectories_flat: flattened array containing predicted trajectories of other drones:
            layout: [drone_j_state_traj ...] where each drone has 6*(Np+1) entries, and there are (num_drones - 1) such blocks.
        """

        # Ensure types
        cur_state = np.asarray(cur_state).astype(float).reshape(6,)
        # If no solver available or cvxpy not installed, fallback
        try:
            import cvxpy  # already imported above, just sanity
        except Exception as e:
            # fallback: return zero acceleration and set predicted trajectory to current state
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)

        # Build reference for linearization
        x_ref = self._build_prediction_ref(cur_state)  # shape (Np+1, 6)

        # Parse neighbor trajectories
        # neighbor_trajectories_flat expected len = (num_drones-1) * 6 * (Np+1)
        neighbors = []
        if neighbor_trajectories_flat is None or len(neighbor_trajectories_flat) == 0:
            # no neighbors
            neighbors = []
        else:
            flat = np.asarray(neighbor_trajectories_flat).astype(float).flatten()
            per = 6 * (self.Np + 1)
            expected = (self.num_drones - 1) * per
            if flat.size != expected:
                # If sizing mismatch, try to be robust: reshape as many full neighbors as possible
                n_possible = flat.size // per
                per_use = per
            else:
                n_possible = (self.num_drones - 1)
                per_use = per
            for j in range(n_possible):
                start = j * per_use
                block = flat[start:start + per_use]
                traj = block.reshape(self.Np + 1, 6)
                neighbors.append(traj)

        # CVXPY variables
        X = cp.Variable((6, self.Np + 1))   # state over horizon (each column a 6-vector)
        U = cp.Variable((3, self.Np))       # control (accelerations) over horizon

        constraints = []

        # initial condition
        constraints += [X[:, 0] == cur_state]

        # dynamics constraints
        for k in range(self.Np):
            # p_next = p_k + dt * v_k + 0.5*dt^2 * a_k
            p_k = X[0:3, k]
            v_k = X[3:6, k]
            a_k = U[:, k]
            p_next = X[0:3, k + 1]
            v_next = X[3:6, k + 1]

            constraints += [
                p_next == p_k + self.dt * v_k + (self.dt ** 2 / 2.0) * a_k,
                v_next == v_k + self.dt * a_k
            ]

            # velocity bounds elementwise
            constraints += [X[3, k] <= self.max_vel, X[3, k] >= -self.max_vel]
            constraints += [X[4, k] <= self.max_vel, X[4, k] >= -self.max_vel]
            constraints += [X[5, k] <= self.max_vel, X[5, k] >= -self.max_vel]

            # z position lower bound (no negative altitude)
            constraints += [X[2, k] >= 0.0]

            # acceleration bounds
            constraints += [U[0, k] <= self.max_acc, U[0, k] >= -self.max_acc]
            constraints += [U[1, k] <= self.max_acc, U[1, k] >= -self.max_acc]
            constraints += [U[2, k] <= self.max_acc, U[2, k] >= -self.max_acc]

            # linearized collision constraints w.r.t. neighbors
            # For each neighbor j, use neighbor predicted position at time k+1 (if available)
            if len(neighbors) > 0:
                p_i_ref = x_ref[k + 1, 0:3]  # reference position of this drone at k+1
                for traj_j in neighbors:
                    p_j_ref = traj_j[k + 1, 0:3]
                    r = p_i_ref - p_j_ref   # vector (3,)
                    r_norm_sq = np.dot(r, r)
                    # If r is almost zero, skip linearization to avoid degenerate constraint
                    if r_norm_sq < 1e-6:
                        # to avoid numerical issues, create a soft constraint pushing them apart in any direction
                        # skip this specific linearization (or add a small margin)
                        continue
                    # Linearization of ||p_i - p_j||^2 >= d_safe^2 around p_i_ref:
                    # 2 * r^T * (p_i - p_i_ref) + ||r||^2 >= d_safe^2
                    # => 2 r^T p_i >= d_safe^2 - ||r||^2 + 2 r^T p_i_ref
                    lhs = 2.0 * (r.reshape(1, 3) @ X[0:3, k + 1])  # shape (1,1) affine
                    rhs = (self.d_safe ** 2 - r_norm_sq + 2.0 * (r @ p_i_ref))
                    # Add as inequality (lhs >= rhs) -> cp enforces lhs >= rhs via lhs >= rhs => -lhs <= -rhs for default style
                    constraints += [lhs >= rhs]

        # terminal state bounds (velocity bound and z >= 0)
        constraints += [X[3, self.Np] <= self.max_vel, X[3, self.Np] >= -self.max_vel]
        constraints += [X[4, self.Np] <= self.max_vel, X[4, self.Np] >= -self.max_vel]
        constraints += [X[5, self.Np] <= self.max_vel, X[5, self.Np] >= -self.max_vel]
        constraints += [X[2, self.Np] >= 0.0]

        # objective: sum_k (X_k - target_state)^T Q (X_k - target_state) + U_k^T R U_k
        obj = 0
        # target state vector (6,) at each stage: [targetpos; zeros(3)]
        target_state_vec = np.hstack([self.target_pos.reshape(3,), np.zeros(3,)]).astype(float)

        for k in range(self.Np):
            xk = X[:, k]
            uk = U[:, k]
            # cvxpy quad_form uses symmetric PSD matrices
            obj += cp.quad_form(xk - target_state_vec, self.Q) + cp.quad_form(uk, self.R)

        # terminal cost
        xN = X[:, self.Np]
        obj += cp.quad_form(xN - target_state_vec, self.Q)

        problem = cp.Problem(cp.Minimize(obj), constraints)

        try:
            # Solve with OSQP (fast QP solver). Warm start and verbose False.
            problem.solve(solver=cp.OSQP,
                          warm_start=True,
                          verbose=False,
                          eps_abs=1e-3,
                          eps_rel=1e-3,
                          max_iter=10000)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # solver failed -> fallback safe hover
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

            X_opt = X.value  # shape (6, Np+1)
            U_opt = U.value  # shape (3, Np)

            if X_opt is None or U_opt is None:
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

            # Save predicted trajectory in same shape as original (Np+1, 6)
            self.predicted_trajectory = X_opt.T

            # extract first control
            u0 = U_opt[:, 0].reshape(3,)

            # store last reference
            self._last_x_ref = X_opt.T.copy()

            return u0

        except Exception as e:
            # Infeasible or solver error: return zero acceleration & keep predicted trajectory at current
            print(f"[DMPC][Drone {self.id}] QP solver error: {e}")
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)
