# DMPCControl.py
import numpy as np
import time
import cvxpy as cp

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


class DMPCControl:
    """
    DMPC controller (QP via cvxpy OSQP). Lightweight robust instrumentation included.
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
        self.id = int(drone_id)
        self.Np = int(Np)
        self.dt = float(dt)
        self.target_pos = np.asarray(target_pos).reshape(3, 1).astype(float)
        self.d_safe = float(d_safe)
        self.num_drones = int(num_drones)

        self.max_acc = float(max_acc)
        self.max_vel = float(max_vel)

        self.predicted_trajectory = np.zeros((self.Np + 1, 6))

        # tuning: state & control weights
        self.Q = np.diag([200., 200., 200., 0.05, 0.05, 0.05])
        self.R = np.diag([0.5, 0.5, 0.5])

        # instrumentation
        self.solver_wall_times = []
        self.solver_cpu_times = []
        self.solver_status = []
        self.solver_iters = []
        self.solver_obj = []
        self.solver_slack_norm = []
        self.solver_problem_size = []
        self.solver_mem_rss = []

        self._last_x_ref = None

    def _build_prediction_ref(self, cur_state: np.ndarray):
        if (self.predicted_trajectory is not None) and (not np.allclose(self.predicted_trajectory, 0.0)):
            return self.predicted_trajectory.copy()
        else:
            return np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))

    def compute_control(self, cur_state: np.ndarray, neighbor_trajectories_flat: np.ndarray):
        cur_state = np.asarray(cur_state).astype(float).reshape(6,)

        x_ref = self._build_prediction_ref(cur_state)

        # parse neighbors
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
                try:
                    traj = block.reshape(self.Np + 1, 6)
                    neighbors.append(traj)
                except Exception:
                    continue

        # cvx variables
        X = cp.Variable((6, self.Np + 1))
        U = cp.Variable((3, self.Np))

        constraints = [X[:, 0] == cur_state]

        num_collision_constraints = len(neighbors) * self.Np
        if num_collision_constraints > 0:
            s = cp.Variable(num_collision_constraints)
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

            constraints += [
                p_next == p_k + self.dt * v_k + (self.dt ** 2 / 2.0) * a_k,
                v_next == v_k + self.dt * a_k
            ]

            # vel bounds
            constraints += [X[3, k] <= self.max_vel, X[3, k] >= -self.max_vel]
            constraints += [X[4, k] <= self.max_vel, X[4, k] >= -self.max_vel]
            constraints += [X[5, k] <= self.max_vel, X[5, k] >= -self.max_vel]

            # z floor
            constraints += [X[2, k] >= 0.0]

            # accel bounds
            constraints += [U[0, k] <= self.max_acc, U[0, k] >= -self.max_acc]
            constraints += [U[1, k] <= self.max_acc, U[1, k] >= -self.max_acc]
            constraints += [U[2, k] <= self.max_acc, U[2, k] >= -self.max_acc]

            if len(neighbors) > 0:
                p_i_ref = x_ref[k + 1, 0:3]
                for traj_j in neighbors:
                    p_j_ref = traj_j[k + 1, 0:3]
                    r = p_i_ref - p_j_ref
                    r_norm_sq = np.dot(r, r)
                    if r_norm_sq < 1e-8:
                        r = np.array([1e-3, 0.0, 0.0])
                        r_norm_sq = np.dot(r, r)
                    lhs = 2.0 * (r.reshape(1, 3) @ X[0:3, k + 1])
                    rhs = (self.d_safe ** 2 - r_norm_sq + 2.0 * (r @ p_i_ref))
                    if s is not None:
                        constraints += [lhs + s[coll_idx] >= rhs]
                    else:
                        constraints += [lhs >= rhs]
                    coll_idx += 1

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

        if s is not None:
            obj += 1e5 * cp.sum_squares(s)

        problem = cp.Problem(cp.Minimize(obj), constraints)

        t_wall_start = time.time()
        t_cpu_start = time.process_time()
        mem_start = psutil.Process().memory_info().rss if _HAS_PSUTIL else None

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                          eps_abs=1e-2, eps_rel=1e-2, max_iter=20000)
        except Exception:
            try:
                problem.solve(solver=cp.OSQP, warm_start=False, verbose=False,
                              eps_abs=1e-1, eps_rel=1e-1, max_iter=10000)
            except Exception as e:
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
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)

        t_wall = time.time() - t_wall_start
        t_cpu = time.process_time() - t_cpu_start
        mem_end = psutil.Process().memory_info().rss if _HAS_PSUTIL else None
        mem_delta = (mem_end - mem_start) if _HAS_PSUTIL else np.nan

        try:
            stats = problem.solver_stats
        except Exception:
            stats = None

        num_iters = np.nan
        if stats is not None:
            try:
                raw = getattr(stats, "num_iters", None)
                if raw is not None:
                    num_iters = float(raw)
            except Exception:
                num_iters = np.nan

        status = problem.status if hasattr(problem, "status") else "unknown"
        try:
            obj_val = float(problem.value) if hasattr(problem, "value") and problem.value is not None else np.nan
        except Exception:
            obj_val = np.nan

        try:
            if s is not None and hasattr(s, "value") and s.value is not None:
                slack_norm = float(np.linalg.norm(np.asarray(s.value)))
            else:
                slack_norm = 0.0
        except Exception:
            slack_norm = np.nan

        try:
            n_vars = sum([int(v.size) for v in problem.variables()])
            n_cons = int(len(problem.constraints))
        except Exception:
            n_vars = np.nan
            n_cons = np.nan

        # append instrumentation safely
        try:
            self.solver_wall_times.append(float(t_wall))
        except Exception:
            self.solver_wall_times.append(np.nan)
        try:
            self.solver_cpu_times.append(float(t_cpu))
        except Exception:
            self.solver_cpu_times.append(np.nan)
        self.solver_status.append(str(status))
        try:
            self.solver_iters.append(int(num_iters) if not np.isnan(num_iters) else np.nan)
        except Exception:
            self.solver_iters.append(np.nan)
        try:
            self.solver_obj.append(float(obj_val) if (obj_val is not None and np.isfinite(obj_val)) else np.nan)
        except Exception:
            self.solver_obj.append(np.nan)
        try:
            self.solver_slack_norm.append(float(slack_norm) if (slack_norm is not None and np.isfinite(slack_norm)) else np.nan)
        except Exception:
            self.solver_slack_norm.append(np.nan)
        try:
            self.solver_problem_size.append((int(n_vars) if np.isfinite(n_vars) else np.nan,
                                            int(n_cons) if np.isfinite(n_cons) else np.nan))
        except Exception:
            self.solver_problem_size.append((np.nan, np.nan))
        try:
            self.solver_mem_rss.append(int(mem_delta) if _HAS_PSUTIL and (mem_delta is not None) else np.nan)
        except Exception:
            self.solver_mem_rss.append(np.nan)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)

        try:
            X_opt = X.value
            U_opt = U.value
            if X_opt is None or U_opt is None:
                self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
                return np.zeros(3)
            self.predicted_trajectory = X_opt.T
            u0 = U_opt[:, 0].reshape(3,)
            self._last_x_ref = X_opt.T.copy()
            return u0
        except Exception:
            self.predicted_trajectory[:] = np.tile(cur_state.reshape(1, 6), (self.Np + 1, 1))
            return np.zeros(3)
