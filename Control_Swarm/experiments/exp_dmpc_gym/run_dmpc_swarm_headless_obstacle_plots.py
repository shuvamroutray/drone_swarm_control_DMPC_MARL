#!/usr/bin/env python3
"""
run_dmpc_swarm_headless.py

Headless DMPC run that records waypoints and saves them to a .npz file.

Saves:
  - times:      (T,) array of simulation times (seconds)
  - pos:        (T, NUM_DRONES, 3) actual drone base positions
  - quat:       (T, NUM_DRONES, 4) actual base quaternions
  - vel:        (T, NUM_DRONES, 3) linear velocities
  - ang_vel:    (T, NUM_DRONES, 3) angular velocities
  - rpms:       (T, NUM_DRONES, 4) last-applied RPMs
  - dmpc_preds: (T, NUM_DRONES, Np+1, 6) predicted trajectories (if available)
  - metadata:   dict with simulation params and collected controller instrumentation (if available)
Usage:
    python run_dmpc_swarm_headless.py
"""
import time
import numpy as np
import os
from pathlib import Path

from gym_pybullet_drones.envs.DMPCAviary_Obstacle_plots import DMPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_get_attr_list(obj, name):
    """
    Safely read attribute 'name' from obj. If missing -> return [].
    If present and array-like -> convert to python list when possible so it pickles cleanly.
    """
    val = getattr(obj, name, None)
    if val is None:
        return []
    # If it's already a list, return as-is
    if isinstance(val, list):
        return val
    # If numpy array or other sequence, try to convert to list
    try:
        return list(val)
    except Exception:
        # fallback: return as single-element list
        return [val]


def collect_controller_metrics(dmpc_controllers, env=None):
    """
    For each DMPC controller in dmpc_controllers, try to collect instrumentation lists.
    Returns a metadata-dict fragment with per-drone lists.
    Keys returned:
      - solver_wall_times_per_drone
      - solver_cpu_times_per_drone
      - solver_status_per_drone
      - solver_iters_per_drone
      - solver_obj_per_drone
      - solver_slack_norm_per_drone
      - solver_mem_rss_per_drone
      - solver_problem_size_per_drone

    If controllers do not expose any attribute the corresponding per-drone entry will be [].
    """
    metrics = {
        "solver_wall_times_per_drone": [],
        "solver_cpu_times_per_drone": [],
        "solver_status_per_drone": [],
        "solver_iters_per_drone": [],
        "solver_obj_per_drone": [],
        "solver_slack_norm_per_drone": [],
        "solver_mem_rss_per_drone": [],
        "solver_problem_size_per_drone": []
    }

    for ctrl in dmpc_controllers:
        metrics["solver_wall_times_per_drone"].append(_safe_get_attr_list(ctrl, "solver_wall_times"))
        metrics["solver_cpu_times_per_drone"].append(_safe_get_attr_list(ctrl, "solver_cpu_times"))
        metrics["solver_status_per_drone"].append(_safe_get_attr_list(ctrl, "solver_status"))
        metrics["solver_iters_per_drone"].append(_safe_get_attr_list(ctrl, "solver_iters"))
        metrics["solver_obj_per_drone"].append(_safe_get_attr_list(ctrl, "solver_obj"))
        metrics["solver_slack_norm_per_drone"].append(_safe_get_attr_list(ctrl, "solver_slack_norm"))
        metrics["solver_mem_rss_per_drone"].append(_safe_get_attr_list(ctrl, "solver_mem_rss"))
        metrics["solver_problem_size_per_drone"].append(_safe_get_attr_list(ctrl, "solver_problem_size"))


    if env is not None:
        try:
            env_times = getattr(env, "_dmpc_solve_walltimes", None)
            if env_times is not None and len(env_times) == len(dmpc_controllers):
                metrics["solver_wall_times_per_drone"] = [list(x) for x in env_times]
        except Exception:
            pass

    return metrics


def run_and_record(filename=None, sim_time_s=20.0, num_drones=2, gui=False):
    if filename is None:
        filename = Path(OUTPUT_DIR) / f"dmpc_waypoints_{int(time.time())}.npz"
    else:
        filename = Path(filename)

    NUM_DRONES = num_drones
    # INIT_XYZS = np.array([
    #     [0, -1.5, 1.0],
    #     [0,  1.5, 1.0]
    # ])[:NUM_DRONES]

    # TARGETS = np.array([
    #     [0.0,  1.5, 1.0],   # target for drone 0
    #     [0.0, -1.5, 1.0]    # target for drone 1
    # ])[:NUM_DRONES]

    #6-drone initial positions (left/right columns, symmetric) ----------
    # If you pass num_drones < 6 this will be sliced automatically.
    INIT_XYZS = np.array([
        [-1.5, -1.0, 1.00],   # drone 0 (left, lower)
        [-1.5,  0.0, 1.00],   # drone 1 (left, middle)
        [-1.5,  1.0, 1.00],   # drone 2 (left, upper)
        [ 1.5, -1.0, 1.00],   # drone 3 (right, lower)
        [ 1.5,  0.0, 1.00],   # drone 4 (right, middle)
        [ 1.5,  1.0, 1.00],   # drone 5 (right, upper)
    ], dtype=float)[:NUM_DRONES, :]

    # ---------- Per-drone targets chosen to cross while avoiding collisions ----------
    TARGETS = np.array([
        [ 1.5, -1.0, 1.00],   # drone 0 -> right slightly lower than drone 3
        [ 1.5,  0.0, 1.00],   # drone 1
        [ 1.5,  1.0, 1.00],   # drone 2
        [-1.5, -1.0, 1.00],   # drone 3
        [-1.5,  0.0, 1.00],   # drone 4
        [-1.5,  1.0, 1.00],   # drone 5
    ], dtype=float)[:NUM_DRONES, :]


    OBSTACLES = [
        np.array([0.0, 1.0, 0.5]),   # center obstacle
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, -1.0, 0.5])
    ]

    env = DMPCAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB_GND_DRAG_DW,
        gui=gui,
        record=False,
        target_pos=TARGETS,
        obstacles=OBSTACLES    # pass the list here
    )

    obs, info = env.reset()

    # How many control steps to run:
    T_ctrl = int(sim_time_s * env.CTRL_FREQ)

    # Storage pre-alloc (we don't know exact T if early termination, so use list then stack)
    times = []
    pos_list = []
    quat_list = []
    vel_list = []
    angvel_list = []
    rpms_list = []
    dmpc_preds_list = []  # store predicted trajectories for each drone (Np+1,6)

    start_time = time.time()
    DUMMY_ACTION = np.zeros((NUM_DRONES, 1))

    for t in range(T_ctrl):
        # step at CONTROL frequency (env.step does internal pyb steps)
        obs, reward, terminated, truncated, info = env.step(DUMMY_ACTION)

        # record measurements at each control step (after env._updateAndStoreKinematicInformation)
        times.append((time.time() - start_time))
        pos_list.append(env.pos.copy())          # shape (NUM_DRONES, 3)
        quat_list.append(env.quat.copy())        # shape (NUM_DRONES, 4)
        vel_list.append(env.vel.copy())          # shape (NUM_DRONES, 3)
        angvel_list.append(env.ang_v.copy())     # shape (NUM_DRONES, 3)
        rpms_list.append(env.last_clipped_action.copy())  # shape (NUM_DRONES, 4)

        # predicted trajectories (for each drone) -> (NUM_DRONES, Np+1, 6)
        preds = np.zeros((NUM_DRONES, env.NP + 1, 6))
        for i in range(NUM_DRONES):
            try:
                preds[i, :, :] = env.dmpc_controllers[i].predicted_trajectory.copy()
            except Exception:
                preds[i, :, :] = np.tile(np.hstack([env.pos[i, :], env.vel[i, :]]), (env.NP + 1, 1))
        dmpc_preds_list.append(preds)

        if terminated or truncated:
            print(f"Simulation terminated early at step {t} (terminated={terminated}, truncated={truncated})")
            break

    elapsed = time.time() - start_time
    print(f"Headless simulation finished. elapsed={elapsed:.2f}s, saved to {filename}")

    # Stack into arrays
    times = np.array(times)
    pos = np.stack(pos_list, axis=0)       # (T, NUM_DRONES, 3)
    quat = np.stack(quat_list, axis=0)     # (T, NUM_DRONES, 4)
    vel = np.stack(vel_list, axis=0)       # (T, NUM_DRONES, 3)
    ang_vel = np.stack(angvel_list, axis=0)
    rpms = np.stack(rpms_list, axis=0)
    dmpc_preds = np.stack(dmpc_preds_list, axis=0)  # (T, NUM_DRONES, Np+1, 6)

    # collect controller metrics (if controllers expose instrumentation attributes)
    try:
        ctrl_metrics = collect_controller_metrics(env.dmpc_controllers, env=env)
    except Exception as e:
        # fallback: empty metrics
        print("[run_and_record] warning: failed to collect controller metrics:", e)
        ctrl_metrics = {
            "solver_wall_times_per_drone": [],
            "solver_cpu_times_per_drone": [],
            "solver_status_per_drone": [],
            "solver_iters_per_drone": [],
            "solver_obj_per_drone": [],
            "solver_slack_norm_per_drone": [],
            "solver_mem_rss_per_drone": [],
            "solver_problem_size_per_drone": []
        }

    metadata = {
        "num_drones": NUM_DRONES,
        "ctrl_freq": env.CTRL_FREQ,
        "pyb_freq": env.PYB_FREQ,
        "dt_ctrl": env.CTRL_TIMESTEP,
        "np_horizon": env.NP,
        # store per-drone targets (list-of-lists) for replay/inspection
        "targets_per_drone": env.TARGET_POS.tolist(),
        "obstacles": [pos.tolist() for pos in getattr(env, "OBSTACLE_POSITIONS", [])],
        # add controller metrics inside metadata
        **ctrl_metrics
    }

    np.savez_compressed(
        filename,
        times=times,
        pos=pos,
        quat=quat,
        vel=vel,
        ang_vel=ang_vel,
        rpms=rpms,
        dmpc_preds=dmpc_preds,
        metadata=metadata
    )

    env.close()
    return filename


if __name__ == "__main__":
    out = run_and_record(sim_time_s=20.0, num_drones=2, gui=False)
    print("Saved:", out)
