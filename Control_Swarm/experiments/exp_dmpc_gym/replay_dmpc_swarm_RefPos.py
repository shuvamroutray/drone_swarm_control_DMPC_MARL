# replay_dmpc_swarm.py
"""
Replay a .npz waypoint file with GUI on.

This script:
 - creates a DMPCAviary with GUI=True
 - loads a .npz file produced by run_dmpc_swarm_headless.py
 - teleports each drone to the recorded poses at each control step (visual-only replay)
 - optionally updates per-drone visual target markers if targets were saved

Usage:
    python replay_dmpc_swarm.py recordings/dmpc_waypoints_<ts>.npz --speed 1.0

Arguments:
    file   : path to the .npz produced by the headless recorder
    --speed: replay speed multiplier (1.0 = real-time; >1 faster, <1 slower)

Notes:
 - The replay teleports drone bases (resetBasePositionAndOrientation) so physics is overridden.
 - The GUI remains open after replay finishes so 414you can inspect the final scene.
"""
import time
import numpy as np
import argparse
import pybullet as p

from gym_pybullet_drones.envs.DMPCAviary_RefPos import DMPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def replay(npz_path: str, speed: float = 1.0):
    # Load npz (robust to allow_pickle)
    data = np.load(npz_path, allow_pickle=True)

    # Mandatory arrays saved by the headless recorder
    times = data["times"]         # (T,)
    pos = data["pos"]             # (T, NUM_DRONES, 3)
    quat = data["quat"]           # (T, NUM_DRONES, 4)
    vel = data["vel"]             # (T, NUM_DRONES, 3)
    rpms = data["rpms"]           # (T, NUM_DRONES, 4)

    # Optional: predicted trajectories, targets and metadata
    dmpc_preds = data.get("dmpc_preds", None)              # (T, NUM_DRONES, Np+1, 6) or None
    metadata = data.get("metadata", None)
    if metadata is not None and isinstance(metadata, np.ndarray) and metadata.size == 1:
        # sometimes metadata saved as an object array -> extract
        try:
            metadata = metadata.item()
        except Exception:
            metadata = None

    # Per-step or static targets if available
    targets_over_time = data.get("targets_over_time", None)   # (T, NUM_DRONES, 3) optional
    targets_static = None
    if metadata is not None:
        targets_static = metadata.get("targets_per_drone", None) or metadata.get("target_pos", None)

    # Convert to numpy arrays / canonical shapes
    pos = np.array(pos)
    quat = np.array(quat)
    vel = np.array(vel)
    rpms = np.array(rpms)
    times = np.array(times)

    if targets_over_time is not None:
        targets_over_time = np.array(targets_over_time)
    elif targets_static is not None:
        try:
            targets_static = np.array(targets_static, dtype=float)
            # If single 3-vector was saved, broadcast to per-drone later
        except Exception:
            targets_static = None

    # Determine number of drones and initial poses
    T = pos.shape[0]
    NUM_DRONES = pos.shape[1]
    INIT_XYZS = pos[0, :, :]

    # Create GUI environment for replay
        # extract obstacles from metadata if available
    obstacles = None
    if metadata is not None:
        # metadata might be a dict; try both dict and object array forms
        if isinstance(metadata, dict):
            obstacles = metadata.get("obstacles", None)
        else:
            try:
                md = dict(metadata) if hasattr(metadata, "item") else metadata
                obstacles = md.get("obstacles", None)
            except Exception:
                obstacles = None

    # build env with same obstacles (if available)
    env = DMPCAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,   # simpler physics for visual replay
        gui=True,
        record=False,
        obstacles=obstacles
    )

    # center camera (optional)
    try:
        # pick a target point around middle of initial positions (avg pos)
        cam_target = np.mean(INIT_XYZS, axis=0).tolist()
        p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                     cameraYaw=-30,
                                     cameraPitch=-30,
                                     cameraTargetPosition=cam_target,
                                     physicsClientId=env.getPyBulletClient())
    except Exception:
        pass


    # Short pause to allow GUI to initialize
    time.sleep(0.8)

    client = env.getPyBulletClient()
    ctrl_dt = 1.0 / env.CTRL_FREQ

    print(f"Loaded {npz_path}: T={T}, NUM_DRONES={NUM_DRONES}, ctrl_freq={env.CTRL_FREQ}Hz")
    print(f"Replaying at speed={speed}x (real-time period per step = {ctrl_dt/speed:.3f}s)")

    # Set initial markers if static targets exist
    if targets_over_time is not None:
        # use first timestamp's targets for initial placement
        try:
            first_targets = targets_over_time[0]  # (NUM_DRONES,3)
            for i in range(NUM_DRONES):
                if hasattr(env, "target_marker_ids") and i < len(env.target_marker_ids):
                    p.resetBasePositionAndOrientation(
                        env.target_marker_ids[i],
                        first_targets[i].tolist(),
                        [0, 0, 0, 1],
                        physicsClientId=client
                    )
        except Exception:
            pass
    elif targets_static is not None:
        # If saved as (NUM_DRONES,3) or (3,) handle both
        try:
            arr = np.array(targets_static, dtype=float)
            if arr.shape == (3,):
                arr = np.tile(arr.reshape(1, 3), (NUM_DRONES, 1))
            for i in range(NUM_DRONES):
                if hasattr(env, "target_marker_ids") and i < len(env.target_marker_ids):
                    p.resetBasePositionAndOrientation(
                        env.target_marker_ids[i],
                        arr[i].tolist(),
                        [0, 0, 0, 1],
                        physicsClientId=client
                    )
        except Exception:
            pass

    # Main replay loop: teleport drones & optionally update markers per-step
    start_time = time.time()
    for k in range(T):
        for i in range(NUM_DRONES):
            # Teleport drone base pose (visual)
            try:
                p.resetBasePositionAndOrientation(
                    env.DRONE_IDS[i],
                    pos[k, i, :].tolist(),
                    quat[k, i, :].tolist(),
                    physicsClientId=client
                )
                # Set linear velocity for smoother visuals (optional)
                p.resetBaseVelocity(
                    env.DRONE_IDS[i],
                    linearVelocity=vel[k, i, :].tolist(),
                    angularVelocity=[0, 0, 0],
                    physicsClientId=client
                )
            except Exception:
                # If any reset fails, continue gracefully
                pass

            # Update per-drone marker if per-step targets are available
            if targets_over_time is not None:
                try:
                    if hasattr(env, "target_marker_ids") and i < len(env.target_marker_ids):
                        p.resetBasePositionAndOrientation(
                            env.target_marker_ids[i],
                            targets_over_time[k, i, :].tolist(),
                            [0, 0, 0, 1],
                            physicsClientId=client
                        )
                except Exception:
                    pass

        # Sleep to maintain timing derived from recorded times and requested speed.
        # Use recorded times array so we preserve irregular timing if present.
        # target_time = start_time + (recorded_time / speed)
        try:
            target_time = start_time + (times[k] / float(speed))
            to_sleep = target_time - time.time()
            if to_sleep > 0:
                time.sleep(to_sleep)
            else:
                # already late; continue without sleep to catch up
                pass
        except Exception:
            # fallback: fixed-step sleep based on ctrl dt and speed
            time.sleep(max(0.0, ctrl_dt / float(speed)))

    print("Replay finished. GUI will remain open until you close it manually.")
    # Keep GUI open for inspection. Do not call env.close() here.
    # If user wants automatic close after N seconds, they can modify this script.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay DMPC headless recording (.npz) in GUI")
    parser.add_argument("file", help="path to waypoint .npz produced by headless recorder")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier (1.0 = real-time)")
    args = parser.parse_args()
    replay(args.file, speed=args.speed)










###################### Version 2 With Target Markers################################


# replay_dmpc_swarm.py
"""
Replay a .npz waypoint file with GUI on.

This script:
 - creates a DMPCAviary with GUI=True (so you can see the visualization)
 - loads a .npz file saved by run_dmpc_swarm_headless.py
 - for every recorded control time-step:
     - force the positions and orientations of each drone using pybullet.resetBasePositionAndOrientation
     - optionally set velocities using resetBaseVelocity for nicer visualization
     - sleep to match real-time at CTRL_FREQ (or faster/slower if you want)
Notes: This is a visual replay only. It overrides physics by teleporting bases to saved poses.
"""
import time
import numpy as np
import argparse
import pybullet as p

from gym_pybullet_drones.envs.DMPCAviary import DMPCAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def replay(npz_path, speed=1.0):
    data = np.load(npz_path, allow_pickle=True)
    times = data["times"]                # (T,)
    pos = data["pos"]                    # (T, NUM_DRONES, 3)
    quat = data["quat"]                  # (T, NUM_DRONES, 4)
    vel = data["vel"]                    # (T, NUM_DRONES, 3)
    rpms = data["rpms"]                  # (T, NUM_DRONES, 4)
    metadata = data["metadata"].item() if "metadata" in data else {}

    NUM_DRONES = metadata.get("num_drones", pos.shape[1])
    INIT_XYZS = pos[0, :, :]

    env = DMPCAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,    # use simple pybullet for replay
        gui=True,
        record=False
    )

    # small pause to load GUI
    time.sleep(1.0)

    client = env.getPyBulletClient()
    T = pos.shape[0]
    ctrl_dt = 1.0 / env.CTRL_FREQ
    print(f"Replaying {T} steps at CTRL_FREQ={env.CTRL_FREQ}Hz (speed={speed}x)")

    # Place markers if metadata contains a target_pos
    metadata_target = metadata.get("target_pos", None)
    if metadata_target is not None:
        # robustly convert to numpy array or leave as-is if already array-like
        try:
            metadata_target_arr = np.array(metadata_target, dtype=float).reshape(3,)
        except Exception:
            metadata_target_arr = None

        if metadata_target_arr is not None:
            for i in range(NUM_DRONES):
                if hasattr(env, "target_marker_ids") and i < len(env.target_marker_ids):
                    # reset marker position to metadata target
                    p.resetBasePositionAndOrientation(env.target_marker_ids[i],
                                                      metadata_target_arr.tolist(),
                                                      [0, 0, 0, 1],
                                                      physicsClientId=client)

    # start replay loop
    start = time.time()
    for k in range(T):
        for i in range(NUM_DRONES):
            # teleport drone base pose
            try:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[i],
                                                  pos[k, i, :].tolist(),
                                                  quat[k, i, :].tolist(),
                                                  physicsClientId=client)
                # set velocity (for nicer visuals; optional)
                p.resetBaseVelocity(env.DRONE_IDS[i],
                                    linearVelocity=vel[k, i, :].tolist(),
                                    angularVelocity=[0, 0, 0],
                                    physicsClientId=client)
            except Exception:
                # if any reset fails, skip it (robust replay)
                pass

            # also move the visual target marker for this drone (if markers exist)
            if hasattr(env, "target_marker_ids") and i < len(env.target_marker_ids):
                try:
                    # if you want the marker to follow the DMPC predicted target saved in metadata
                    # we already placed them at start; alternatively, place marker at current drone's next target (if available)
                    pass
                except Exception:
                    pass

        # render via time.sleep to match recorded timing
        target_time = start + (times[k] / speed)
        now = time.time()
        to_sleep = target_time - now
        if to_sleep > 0:
            time.sleep(to_sleep)

    print("Replay finished. GUI will remain open until you close it manually.")
    # Keep GUI open (do not call env.close()), allow user to inspect scene manually.
    # If you want the script to close the GUI automatically after some seconds, uncomment next line:
    # time.sleep(5); env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to waypoint .npz produced by headless recorder")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier (1.0 = real-time)")
    args = parser.parse_args()
    replay(args.file, speed=args.speed)







#########***********Version 1 without target Markers*************#########################


# # replay_dmpc_swarm.py
# """
# Replay a .npz waypoint file with GUI on.

# This script:
#  - creates a DMPCAviary with GUI=True (so you can see the visualization)
#  - loads a .npz file saved by run_dmpc_swarm_headless.py
#  - for every recorded control time-step:
#      - force the positions and orientations of each drone using pybullet.resetBasePositionAndOrientation
#      - optionally set velocities using resetBaseVelocity for nicer visualization
#      - sleep to match real-time at CTRL_FREQ (or faster/slower if you want)
# Notes: This is a visual replay only. It overrides physics by teleporting bases to saved poses.
# """
# import time
# import numpy as np
# import argparse

# from gym_pybullet_drones.envs.DMPCAviary import DMPCAviary
# from gym_pybullet_drones.utils.enums import DroneModel, Physics
# import pybullet as p

# def replay(npz_path, speed=1.0):
#     data = np.load(npz_path, allow_pickle=True)
#     times = data["times"]                # (T,)
#     pos = data["pos"]                    # (T, NUM_DRONES, 3)
#     quat = data["quat"]                  # (T, NUM_DRONES, 4)
#     vel = data["vel"]                    # (T, NUM_DRONES, 3)
#     rpms = data["rpms"]                  # (T, NUM_DRONES, 4)
#     metadata = data["metadata"].item() if "metadata" in data else {}

#     NUM_DRONES = metadata.get("num_drones", pos.shape[1])
#     INIT_XYZS = pos[0, :, :]

#     env = DMPCAviary(
#         drone_model=DroneModel.CF2X,
#         num_drones=NUM_DRONES,
#         initial_xyzs=INIT_XYZS,
#         physics=Physics.PYB,    # use simple pybullet for replay
#         gui=True,
#         record=False
#     )

#         # If metadata contains a per-drone target or a global target, place markers there
#     metadata_target = metadata.get("target_pos", None)
#     if metadata_target is not None:
#         # Place all drone markers at same target (common case)
#         for i in range(NUM_DRONES):
#             if hasattr(env, "target_marker_ids") and i < len(env.target_marker_ids):
#                 p.resetBasePositionAndOrientation(env.target_marker_ids[i],
#                                                   metadata_target.tolist(),
#                                                   [0, 0, 0, 1],
#                                                   physicsClientId=env.getPyBulletClient())











#     # small pause to load GUI
#     time.sleep(1.0)

#     client = env.getPyBulletClient()
#     T = pos.shape[0]
#     ctrl_dt = 1.0 / env.CTRL_FREQ
#     print(f"Replaying {T} steps at CTRL_FREQ={env.CTRL_FREQ}Hz (speed={speed}x)")

#     start = time.time()
#     for k in range(T):
#         for i in range(NUM_DRONES):
#             # teleport drone base pose
#             p.resetBasePositionAndOrientation(env.DRONE_IDS[i],
#                                               pos[k, i, :].tolist(),
#                                               quat[k, i, :].tolist(),
#                                               physicsClientId=client)

#             # set velocity (for nicer visuals; optional)
#             p.resetBaseVelocity(env.DRONE_IDS[i],
#                                 linearVelocity=vel[k, i, :].tolist(),
#                                 angularVelocity=[0, 0, 0],
#                                 physicsClientId=client)

#         # render via time.sleep to match recorded timing
#         target_time = start + (times[k] / speed)
#         now = time.time()
#         to_sleep = target_time - now
#         if to_sleep > 0:
#             time.sleep(to_sleep)

#     print("Replay finished. Keep GUI open until you close it manually.")
#     # do not call env.close() to keep GUI visible; user may close window when done.

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("file", help="path to waypoint .npz produced by headless recorder")
#     parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier (1.0 = real-time)")
#     args = parser.parse_args()
#     replay(args.file, speed=args.speed)
