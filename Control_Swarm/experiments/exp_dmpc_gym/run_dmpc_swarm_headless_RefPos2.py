# run_dmpc_swarm_headless.py
"""
Headless DMPC run that records waypoints and saves them to a .npz file.

Saves:
  - times:      (T,) array of simulation times (seconds)
  - pos:        (T, NUM_DRONES, 3) actual drone base positions
  - quat:       (T, NUM_DRONES, 4) actual base quaternions
  - vel:        (T, NUM_DRONES, 3) linear velocities
  - ang_vel:    (T, NUM_DRONES, 3) angular velocities
  - rpms:       (T, NUM_DRONES, 4) last-applied RPMs
  - dmpc_pred:  (T, NUM_DRONES, Np+1, 6) predicted trajectories (if available)
  - info:       optional metadata
Usage:
    python run_dmpc_swarm_headless.py
"""
import time
import numpy as np
import os
from pathlib import Path

from gym_pybullet_drones.envs.DMPCAviary_RefPos2 import DMPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_and_record(filename=None, sim_time_s=20.0, num_drones=2, gui=False):
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
    [0.0,  1.5, 1.0],   # target for drone 0
    [0.0, -1.5, 1.0]    # target for drone 1
])


    # INIT_XYZS = np.array([
    #         [0, -2.7, 2.7],
    #         [0,  2.7, 1.0]
    #     ])[:NUM_DRONES]

    # TARGETS = np.array([
    #     [0.0,  2.7, 1.0],   # target for drone 0
    #     [0.0, -2.7, 2.7]    # target for drone 1
    # ])


#     INIT_XYZS = np.array([
#         [0, -0.5, 1.0],
#         [0,  0.5, 1.0]
#     ])[:NUM_DRONES]

#     TARGETS = np.array([
#     [2.0, -0.3, 1.2],   # target for drone 0
#     [2.0,  0.3, 1.8]    # target for drone 1
# ])

   
    # ---------- 6-drone initial positions (left/right columns, symmetric) ----------
    # # If you pass num_drones < 6 this will be sliced automatically.
    # INIT_XYZS = np.array([
    #     [-1.5, -1.0, 1.00],   # drone 0 (left, lower)
    #     [-1.5,  0.0, 1.00],   # drone 1 (left, middle)
    #     [-1.5,  1.0, 1.00],   # drone 2 (left, upper)
    #     [ 1.5, -1.0, 1.00],   # drone 3 (right, lower)
    #     [ 1.5,  0.0, 1.00],   # drone 4 (right, middle)
    #     [ 1.5,  1.0, 1.00],   # drone 5 (right, upper)
    # ], dtype=float)[:NUM_DRONES, :]

    # # ---------- Per-drone targets chosen to cross while avoiding collisions ----------
    # TARGETS = np.array([
    #     [ 1.5, -1.0, 1.00],   # drone 0 -> right slightly lower than drone 3
    #     [ 1.5,  0.0, 1.00],   # drone 1
    #     [ 1.5,  1.0, 1.00],   # drone 2
    #     [-1.5, -1.0, 1.00],   # drone 3
    #     [-1.5,  0.0, 1.00],   # drone 4
    #     [-1.5,  1.0, 1.00],   # drone 5
    # ], dtype=float)[:NUM_DRONES, :]

    OBSTACLES = [
    np.array([0.0, 0.0, 0.5])   # center obstacle
    #np.array([0.0, 0.0, 0.0])    # another obstacle if desired
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

    metadata = {
        "num_drones": NUM_DRONES,
        "ctrl_freq": env.CTRL_FREQ,
        "pyb_freq": env.PYB_FREQ,
        "dt_ctrl": env.CTRL_TIMESTEP,
        "np_horizon": env.NP,
        # store per-drone targets (list-of-lists) for replay/inspection
        "targets_per_drone": env.TARGET_POS.tolist(),
        "obstacles": [pos.tolist() for pos in getattr(env, "OBSTACLE_POSITIONS", [])],
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
