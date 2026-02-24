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
        # in DMPCControl.__init__
        self.solver_wall_times = []     # float seconds
        self.solver_cpu_times = []
        self.solver_status = []         # string
        self.solver_iters = []          # int or np.nan
        self.solver_obj = []            # float or np.nan
        self.solver_slack_norm = []     # float or 0
        self.solver_problem_size = []   # (n_vars, n_constraints)
        self.solver_mem_rss = []        # bytes (psutil) or np.nan

        



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
            layout: [drone_j_state_traj ...] where each drone has 6*(Np+1) entries, and there are (num_neighbors) blocks.
        """

        cur_state = np.asarray(cur_state).astype(float).reshape(6,)

        # Build reference
        x_ref = self._build_prediction_ref(cur_state)  # shape (Np+1, 6)

        # Parse neighbors robustly (same code you had)
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
        X = cp.Variable((6, self.Np + 1))
        U = cp.Variable((3, self.Np))

        constraints = []
        # initial condition
        constraints += [X[:, 0] == cur_state]

        # Pre-count linearized collision constraints to create slack vector
        num_collision_constraints = len(neighbors) * self.Np  # one per neighbor per step (k+1)
        # If you expect many obstacles, allow slack size accordingly
        if num_collision_constraints > 0:
            s = cp.Variable(num_collision_constraints)  # slack >= 0
            constraints += [s >= 0]
        else:
            # placeholder: zero-length slack vector (CVXPY doesn't like shape (0,), so skip)
            s = None

        # Build dynamics/box constraints and linearized collisions
        coll_idx = 0
        for k in range(self.Np):
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

            # z position lower bound
            constraints += [X[2, k] >= 0.0]

            # acceleration bounds
            constraints += [U[0, k] <= self.max_acc, U[0, k] >= -self.max_acc]
            constraints += [U[1, k] <= self.max_acc, U[1, k] >= -self.max_acc]
            constraints += [U[2, k] <= self.max_acc, U[2, k] >= -self.max_acc]

            # linearized collision constraints: one per neighbor j at time k+1
            if len(neighbors) > 0:
                p_i_ref = x_ref[k + 1, 0:3]
                for traj_j in neighbors:
                    p_j_ref = traj_j[k + 1, 0:3]
                    r = p_i_ref - p_j_ref
                    r_norm_sq = np.dot(r, r)
                    if r_norm_sq < 1e-8:
                        # If references overlap, replace with a small fixed separation direction to avoid degeneracy:
                        # push along world x (or neighbor->self small offset)
                        r = np.array([1e-3, 0.0, 0.0])
                        r_norm_sq = np.dot(r, r)
                    lhs = 2.0 * (r.reshape(1, 3) @ X[0:3, k + 1])  # 1x1 affine
                    rhs = (self.d_safe ** 2 - r_norm_sq + 2.0 * (r @ p_i_ref))
                    if s is not None:
                        constraints += [lhs + s[coll_idx] >= rhs]
                    else:
                        constraints += [lhs >= rhs]
                    coll_idx += 1

        # terminal bounds
        constraints += [X[3, self.Np] <= self.max_vel, X[3, self.Np] >= -self.max_vel]
        constraints += [X[4, self.Np] <= self.max_vel, X[4, self.Np] >= -self.max_vel]
        constraints += [X[5, self.Np] <= self.max_vel, X[5, self.Np] >= -self.max_vel]
        constraints += [X[2, self.Np] >= 0.0]

        # objective
        obj = 0
        target_state_vec = np.hstack([self.target_pos.reshape(3,), np.zeros(3,)]).astype(float)
        for k in range(self.Np):
            xk = X[:, k]
            uk = U[:, k]
            obj += cp.quad_form(xk - target_state_vec, self.Q) + cp.quad_form(uk, self.R)
        xN = X[:, self.Np]
        obj += cp.quad_form(xN - target_state_vec, self.Q)

        # slack penalty (heavy)
        if s is not None:
            w_slack = 1e5  # tune this big enough to avoid collisions but allow feasibility
            obj += w_slack * cp.sum_squares(s)

        problem = cp.Problem(cp.Minimize(obj), constraints)


        import time
        import numpy as np
        try:
            import psutil
            _HAS_PSUTIL = True
        except Exception:
            _HAS_PSUTIL = False

        # --- just before calling problem.solve(...) ---
        t_wall_start = time.time()
        t_cpu_start = time.process_time()
        mem_start = psutil.Process().memory_info().rss if _HAS_PSUTIL else None

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                        eps_abs=1e-3, eps_rel=1e-3, max_iter=10000)
            # post-solve measurements
            t_wall = time.time() - t_wall_start
            t_cpu = time.process_time() - t_cpu_start
            mem_end = psutil.Process().memory_info().rss if _HAS_PSUTIL else None
            mem_delta = (mem_end - mem_start) if _HAS_PSUTIL else np.nan

            # CVXPY solver stats (defensive)
            try:
                stats = problem.solver_stats
                solve_time_reported = getattr(stats, "solve_time", np.nan)
                num_iters = getattr(stats, "num_iters", np.nan)
            except Exception:
                solve_time_reported = np.nan
                num_iters = np.nan

            status = problem.status if hasattr(problem, "status") else "unknown"
            obj = problem.value if hasattr(problem, "value") else np.nan

            # slack norm (if s exists)
            try:
                if s is not None and hasattr(s, "value"):
                    slack_norm = float(np.linalg.norm(s.value))
                else:
                    slack_norm = 0.0
            except Exception:
                slack_norm = np.nan

            # problem size (rough)
            try:
                n_vars = sum([v.size for v in problem.variables()])
                n_cons = len(problem.constraints)
            except Exception:
                n_vars = np.nan
                n_cons = np.nan

            # store into controller lists
            self.solver_wall_times.append(float(t_wall))
            self.solver_cpu_times.append(float(t_cpu))
            self.solver_status.append(str(status))
            self.solver_iters.append(int(num_iters) if not np.isnan(num_iters) else np.nan)
            self.solver_obj.append(float(obj) if obj is not None else np.nan)
            self.solver_slack_norm.append(float(slack_norm))
            self.solver_problem_size.append((int(n_vars) if not np.isnan(n_vars) else np.nan,
                                            int(n_cons) if not np.isnan(n_cons) else np.nan))
            self.solver_mem_rss.append(int(mem_delta) if _HAS_PSUTIL else np.nan)

        except Exception as e:
            # record failure
            t_wall = time.time() - t_wall_start
            t_cpu = time.process_time() - t_cpu_start
            mem_end = psutil.Process().memory_info().rss if _HAS_PSUTIL else None
            mem_delta = (mem_end - mem_start) if _HAS_PSUTIL else np.nan

            self.solver_wall_times.append(float(t_wall))
            self.solver_cpu_times.append(float(t_cpu))
            self.solver_status.append(f"error: {type(e).__name__}")
            self.solver_iters.append(np.nan)
            self.solver_obj.append(np.nan)
            self.solver_slack_norm.append(np.nan)
            self.solver_problem_size.append((np.nan, np.nan))
            self.solver_mem_rss.append(int(mem_delta) if _HAS_PSUTIL else np.nan)
            print(f"[DMPC][Drone {self.id}] Solver exception: {e}")
            # existing fallback behavior below...

        








        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                        eps_abs=1e-3, eps_rel=1e-3, max_iter=10000)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

            X_opt = X.value
            U_opt = U.value
            if X_opt is None or U_opt is None:
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

            self.predicted_trajectory = X_opt.T
            u0 = U_opt[:, 0].reshape(3,)
            self._last_x_ref = X_opt.T.copy()
            return u0

        except Exception as e:
            print(f"[DMPC][Drone {self.id}] QP solver error: {e}")
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)
