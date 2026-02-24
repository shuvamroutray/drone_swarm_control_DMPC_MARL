# DMPCControl_Obstacle.py
import numpy as np
import cvxpy as cp


class DMPCControl:
    """
    DMPC QP (cvxpy/OSQP) with a free terminal reference variable `r`.

    New additions:
      - r : cp.Variable(3)   # reference position chosen by optimizer
      - objective adds:
          w_ref  * || p_N - r ||^2
          w_term * || r - target_pos ||^2

    This lets the optimizer choose a reachable intermediate reference while still
    penalizing deviation from the true mission target.
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
        # initial guess (flat)
        self.init_guess = np.tile(np.hstack([np.zeros(3), np.zeros(3)]), (Np + 1, 1)).reshape(-1)

        # cost weights (tunable)
        self.Q = np.diag([1e2, 1e2, 1e2, 0.1, 0.1, 0.1])   # state cost
        self.R = np.diag([1e-1, 1e-1, 1e-1])                # control cost

        # New hyper-parameters for reference handling
        self.w_ref = 1e3    # weight for ||p_N - r||^2  (how strongly final state must match r)
        self.w_term = 1e3   # weight for ||r - p_target||^2 (how strongly r must be near actual target)
        # You can reduce w_term to allow r to be farther from final target when unreachable.

        # For linearized collisions we can keep last x_ref
        self._last_x_ref = None

    def _build_prediction_ref(self, cur_state: np.ndarray):
        """
        Returns a reference state trajectory (Np+1, 6) used to linearize collisions.
        If we have self.predicted_trajectory from previous solves, use that; otherwise tile current state.
        cur_state: array-like of length 6 (p (3), v (3))
        """
        if (self.predicted_trajectory is not None) and (not np.allclose(self.predicted_trajectory, 0.0)):
            return self.predicted_trajectory.copy()
        else:
            return np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))

    def compute_control(self, cur_state: np.ndarray, neighbor_trajectories_flat: np.ndarray):
        """
        Solve a QP using CVXPY/OSQP.

        cur_state: (6,) array -> [px,py,pz,vx,vy,vz]
        neighbor_trajectories_flat: flattened neighbor blocks (each block = (Np+1,6))
        """
        cur_state = np.asarray(cur_state).astype(float).reshape(6,)

        # Build reference for linearization
        x_ref = self._build_prediction_ref(cur_state)  # shape (Np+1, 6)

        # Parse neighbors robustly
        neighbors = []
        if neighbor_trajectories_flat is None or len(neighbor_trajectories_flat) == 0:
            neighbors = []
        else:
            flat = np.asarray(neighbor_trajectories_flat).astype(float).flatten()
            per = 6 * (self.Np + 1)
            n_possible = flat.size // per
            for j in range(n_possible):
                start = j * per
                block = flat[start:start + per]
                traj = block.reshape(self.Np + 1, 6)
                neighbors.append(traj)

        # CVXPY variables
        X = cp.Variable((6, self.Np + 1))   # states (columns are time steps)
        U = cp.Variable((3, self.Np))       # controls (accelerations)
        r = cp.Variable(3)                  # new: terminal reference position (free)

        constraints = []

        # initial condition
        constraints += [X[:, 0] == cur_state]

        # pre-count collision constraints to allow slack if desired (we keep same approach)
        num_collision_constraints = len(neighbors) * self.Np
        if num_collision_constraints > 0:
            s = cp.Variable(num_collision_constraints)  # slack >= 0
            constraints += [s >= 0]
        else:
            s = None

        coll_idx = 0
        for k in range(self.Np):
            p_k = X[0:3, k]
            v_k = X[3:6, k]
            a_k = U[:, k]
            p_next = X[0:3, k + 1]
            v_next = X[3:6, k + 1]

            # dynamics
            constraints += [
                p_next == p_k + self.dt * v_k + (self.dt ** 2 / 2.0) * a_k,
                v_next == v_k + self.dt * a_k
            ]

            # velocity bounds elementwise (k)
            constraints += [X[3, k] <= self.max_vel, X[3, k] >= -self.max_vel]
            constraints += [X[4, k] <= self.max_vel, X[4, k] >= -self.max_vel]
            constraints += [X[5, k] <= self.max_vel, X[5, k] >= -self.max_vel]

            # z position lower bound
            constraints += [X[2, k] >= 0.0]

            # acceleration bounds
            constraints += [U[0, k] <= self.max_acc, U[0, k] >= -self.max_acc]
            constraints += [U[1, k] <= self.max_acc, U[1, k] >= -self.max_acc]
            constraints += [U[2, k] <= self.max_acc, U[2, k] >= -self.max_acc]

            # linearized collision constraints relative to neighbors at step k+1
            if len(neighbors) > 0:
                p_i_ref = x_ref[k + 1, 0:3]
                for traj_j in neighbors:
                    p_j_ref = traj_j[k + 1, 0:3]
                    rvec = p_i_ref - p_j_ref
                    r_norm_sq = np.dot(rvec, rvec)
                    if r_norm_sq < 1e-8:
                        # avoid degeneracy
                        rvec = np.array([1e-3, 0.0, 0.0])
                        r_norm_sq = np.dot(rvec, rvec)
                    lhs = 2.0 * (rvec.reshape(1, 3) @ X[0:3, k + 1])  # scalar affine
                    rhs = (self.d_safe ** 2 - r_norm_sq + 2.0 * (rvec @ p_i_ref))
                    if s is not None:
                        constraints += [lhs + s[coll_idx] >= rhs]
                    else:
                        constraints += [lhs >= rhs]
                    coll_idx += 1

        # terminal bounds for velocities and z
        constraints += [X[3, self.Np] <= self.max_vel, X[3, self.Np] >= -self.max_vel]
        constraints += [X[4, self.Np] <= self.max_vel, X[4, self.Np] >= -self.max_vel]
        constraints += [X[5, self.Np] <= self.max_vel, X[5, self.Np] >= -self.max_vel]
        constraints += [X[2, self.Np] >= 0.0]

        # Optional: bound r to workspace box (helps numerical stability).
        # You can tune these box values to your workspace; kept reasonably large here.
        big = 10.0
        constraints += [r[0] <= big, r[0] >= -big, r[1] <= big, r[1] >= -big, r[2] <= big, r[2] >= 0.0]

        # Objective: stage cost + control cost
        obj = 0
        target_state_vec = np.hstack([self.target_pos.reshape(3,), np.zeros(3,)]).astype(float)
        for k in range(self.Np):
            xk = X[:, k]
            uk = U[:, k]
            obj += cp.quad_form(xk - target_state_vec, self.Q) + cp.quad_form(uk, self.R)

        # terminal state cost (traditional)
        xN = X[:, self.Np]
        obj += cp.quad_form(xN - target_state_vec, self.Q)

        # ---------- new reference-related costs ----------
        # encourage final predicted position p_N to be close to r
        obj += self.w_ref * cp.sum_squares(X[0:3, self.Np] - r)
        # encourage r to be close to true mission target
        obj += self.w_term * cp.sum_squares(r - self.target_pos.reshape(3,))

        # penalize slack heavily if used
        if s is not None:
            w_slack = 1e5
            obj += w_slack * cp.sum_squares(s)

        problem = cp.Problem(cp.Minimize(obj), constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                          eps_abs=1e-3, eps_rel=1e-3, max_iter=10000)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # fallback: hover
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

            X_opt = X.value
            U_opt = U.value
            r_opt = r.value if r is not None else None

            if X_opt is None or U_opt is None:
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

            # Save predicted trajectory
            self.predicted_trajectory = X_opt.T

            # store last reference (useful for debugging / visualization)
            try:
                self._last_x_ref = X_opt.T.copy()
                self._last_r_opt = r_opt.copy() if r_opt is not None else None
            except Exception:
                self._last_r_opt = None

            # return first control
            u0 = U_opt[:, 0].reshape(3,)
            return u0

        except Exception as e:
            print(f"[DMPC][Drone {self.id}] QP solver error: {e}")
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)
