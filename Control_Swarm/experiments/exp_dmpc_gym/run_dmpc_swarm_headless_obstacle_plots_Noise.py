#!/usr/bin/env python3
"""
run_dmpc_swarm_headless_obstacle_plots_Noise.py

Headless runner that saves DMPC recordings + noise logs.
"""
import time
import numpy as np
import os
from pathlib import Path

from gym_pybullet_drones.envs.DMPCAviary_Obstacle_plots_Noise import DMPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_get_attr_list(obj, name):
    val = getattr(obj, name, None)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        return list(val)
    except Exception:
        return [val]


def collect_controller_metrics(dmpc_controllers):
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

    return metrics


def run_and_record(filename=None, sim_time_s=20.0, num_drones=2, gui=False, noise_sensor_enable=True,
                   noise_sensor_pos_std=0.005, noise_sensor_vel_std=0.01,
                   noise_actuation_enable=False, noise_actuation_rpm_std=5.0):
    if filename is None:
        filename = Path(OUTPUT_DIR) / f"dmpc_waypoints_{int(time.time())}.npz"
    else:
        filename = Path(filename)

    NUM_DRONES = num_drones
    INIT_XYZS = np.array([
        [0, -1.5, 1.0],
        [0,  1.5, 1.0]
    ])[:NUM_DRONES]

    TARGETS = np.array([
        [0.0,  1.5, 1.0],
        [0.0, -1.5, 1.0]
    ])[:NUM_DRONES]

    OBSTACLES = []

    env = DMPCAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB_GND_DRAG_DW,
        gui=gui,
        record=False,
        target_pos=TARGETS,
        obstacles=OBSTACLES,
        noise_sensor_enable=noise_sensor_enable,
        noise_sensor_pos_std=noise_sensor_pos_std,
        noise_sensor_vel_std=noise_sensor_vel_std,
        noise_actuation_enable=noise_actuation_enable,
        noise_actuation_rpm_std=noise_actuation_rpm_std
    )

    obs, info = env.reset()

    T_ctrl = int(sim_time_s * env.CTRL_FREQ)

    times = []
    pos_list = []
    quat_list = []
    vel_list = []
    angvel_list = []
    rpms_list = []
    dmpc_preds_list = []
    noise_list = []
    act_noise_list = []

    start_time = time.time()
    DUMMY_ACTION = np.zeros((NUM_DRONES, 1))

    for t in range(T_ctrl):
        obs, reward, terminated, truncated, info = env.step(DUMMY_ACTION)

        times.append((time.time() - start_time))
        pos_list.append(env.pos.copy())
        quat_list.append(env.quat.copy())
        vel_list.append(env.vel.copy())
        angvel_list.append(env.ang_v.copy())
        rpms_list.append(env.last_clipped_action.copy())

        preds = np.zeros((NUM_DRONES, env.NP + 1, 6))
        for i in range(NUM_DRONES):
            try:
                preds[i, :, :] = env.dmpc_controllers[i].predicted_trajectory.copy()
            except Exception:
                preds[i, :, :] = np.tile(np.hstack([env.pos[i, :], env.vel[i, :]]), (env.NP + 1, 1))
        dmpc_preds_list.append(preds)

        # noise logs (env holds them)
        try:
            noise_now = env._noise_log[-1] if len(env._noise_log) > 0 else np.zeros((NUM_DRONES, 6))
            noise_list.append(noise_now.copy())
        except Exception:
            noise_list.append(np.zeros((NUM_DRONES, 6)))

        try:
            act_noise_now = env._act_noise_log[-1] if len(env._act_noise_log) > 0 else np.zeros((NUM_DRONES, 4))
            act_noise_list.append(act_noise_now.copy())
        except Exception:
            act_noise_list.append(np.zeros((NUM_DRONES, 4)))

        if terminated:
            print(f"Simulation terminated early at step {t} (terminated={terminated})")
            break
        if truncated:
            print(f"[run_and_record] Warning: env reported truncated=True at step {t}. Continuing to gather data for debugging.")
            # optionally break if you want to stop, but for debugging we continue

    elapsed = time.time() - start_time
    print(f"Headless simulation finished. elapsed={elapsed:.2f}s, saved to {filename}")

    times = np.array(times)
    pos = np.stack(pos_list, axis=0)
    quat = np.stack(quat_list, axis=0)
    vel = np.stack(vel_list, axis=0)
    ang_vel = np.stack(angvel_list, axis=0)
    rpms = np.stack(rpms_list, axis=0)
    dmpc_preds = np.stack(dmpc_preds_list, axis=0)
    noise_arr = np.stack(noise_list, axis=0) if len(noise_list) > 0 else np.zeros((pos.shape[0], NUM_DRONES, 6))
    act_noise_arr = np.stack(act_noise_list, axis=0) if len(act_noise_list) > 0 else np.zeros((pos.shape[0], NUM_DRONES, 4))

    # collect controller metrics
    try:
        ctrl_metrics = collect_controller_metrics(env.dmpc_controllers)
    except Exception as e:
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
        "targets_per_drone": env.TARGET_POS.tolist(),
        "obstacles": [pos.tolist() for pos in getattr(env, "OBSTACLE_POSITIONS", [])],
        # env-level instrumentation
        "env_solver_wall_times_per_drone": getattr(env, "_dmpc_solve_walltimes", []),
        "env_solver_cpu_times_per_drone": getattr(env, "_dmpc_solve_cpu_times", []),
        "env_solver_status_per_drone": getattr(env, "_dmpc_solve_status", []),
        "env_solver_iters_per_drone": getattr(env, "_dmpc_solve_iters", []),
        # controller-level metrics
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
        noise_sensor_log=noise_arr,
        actuation_noise_log=act_noise_arr,
        metadata=metadata
    )

    env.close()
    return filename


if __name__ == "__main__":
    out = run_and_record(sim_time_s=20.0, num_drones=2, gui=False,
                         noise_sensor_enable=True, noise_sensor_pos_std=0.1, noise_sensor_vel_std=0.1,
                         noise_actuation_enable=False, noise_actuation_rpm_std=5.0)
    print("Saved:", out)
