#!/usr/bin/env python3
"""
generate_all_figures.py

Usage:
    python generate_all_figures.py recordings/dmpc_waypoints_176...npz --outdir figs/
    python generate_all_figures.py recordings/*.npz --outdir figs/ --all

Description:
    Loads one or more .npz files (recordings produced by run_dmpc_swarm_headless.py)
    and generates a set of standard diagnostic / presentation plots for each file:

    - topdown (x-y) trajectories
    - 3D trajectories
    - predicted vs actual (snapshot)
    - tracking error vs time + final error bar
    - min pairwise distance vs time (with dashed D_SAFE if available)
    - control effort (from dmpc_preds or rpms)
    - prediction tracking error (mean prediction error at lookahead)
    - optionally solver time histogram / slack plots if those keys exist

    Outputs: PNG files named <basename>__<plotname>.png in the output directory.

Dependencies:
    numpy, matplotlib
    (optional) imageio if you enable GIF creation

Author: ChatGPT (adapted for your DMPC recordings)
"""
import os
import sys
import glob
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
# optional: import imageio if making gifs (not used by default)

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)



def get_times(rec):
    t = rec.get("times", None)
    if t is None:
        # fallback: uniform time using dt_ctrl or assume 30 Hz
        T = rec["pos"].shape[0]
        dt = rec.get("metadata", {}).get("dt_ctrl", 1.0/30.0)
        return np.arange(T) * dt
    return np.array(t)


def load_recording(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    def maybe(key):
        return data.get(key, None)
    # fix metadata possibly saved as length-1 ndarray
    metadata = maybe("metadata")
    if metadata is not None and isinstance(metadata, np.ndarray) and metadata.size == 1:
        try:
            metadata = metadata.item()
        except Exception:
            pass
    rec = {
        "times": maybe("times"),
        "pos": maybe("pos"),
        "quat": maybe("quat"),
        "vel": maybe("vel"),
        "rpms": maybe("rpms"),
        "dmpc_preds": maybe("dmpc_preds"),
        "metadata": metadata,
        # optional diagnostic arrays that user may save
        "solver_times": maybe("solver_times"),
        "slack": maybe("slack"),
    }
    return rec

def pairwise_min_distance(positions_t):
    # positions_t: (N,3)
    if positions_t is None:
        return np.nan
    N = positions_t.shape[0]
    if N <= 1:
        return np.nan
    # vectorized computation
    diffs = positions_t.reshape(N,1,3) - positions_t.reshape(1,N,3)
    dists = np.linalg.norm(diffs, axis=2)
    # ignore diagonal zeros
    i_upper = np.triu_indices(N, k=1)
    if i_upper[0].size == 0:
        return np.nan
    return float(np.min(dists[i_upper]))

def tracking_error_per_drone(pos, targets):
    # pos: (T,N,3), targets: (N,3) or (T,N,3)
    T, N, _ = pos.shape
    targets = np.array(targets)
    if targets.ndim == 2 and targets.shape == (N,3):
        return np.linalg.norm(pos - targets.reshape(1, N, 3), axis=2)  # (T,N)
    elif targets.ndim == 3 and targets.shape[0] == T:
        return np.linalg.norm(pos - targets, axis=2)
    else:
        raise ValueError("targets shape not compatible")

def safe_get_meta(rec, key, default=None):
    md = rec.get("metadata", None)
    if md is None:
        return default
    if isinstance(md, dict):
        return md.get(key, default)
    try:
        return getattr(md, key, default) if not isinstance(md, np.ndarray) else default
    except Exception:
        return default

# ----------------------------
# Plot functions
# ----------------------------



# ----------------------------
# DMPC computational cost plots
# ----------------------------


def plot_one_step_prediction_error(rec, outpath, title=None):
    """
    Plots the 1-step-ahead prediction error:
      error[t,i] = || actual_pos[t+1,i] - predicted_pos[t,i,1] ||
    Requires rec['pos'] and rec['dmpc_preds'].
    """
    pos = rec.get("pos", None)
    preds = rec.get("dmpc_preds", None)
    if pos is None or preds is None:
        print("[one_step_pred] pos or dmpc_preds missing, skipping")
        return

    T_pred, N, Pplus1, _ = preds.shape
    T_pos = pos.shape[0]
    # we can compute errors up to min(T_pred, T_pos-1)
    T = min(T_pred, T_pos - 1)
    if T <= 0:
        print("[one_step_pred] not enough length to compute one-step errors")
        return

    errors = np.full((T, N), np.nan)
    # times from recording (align to t=0..T-1)
    times = get_times(rec)
    times_arr = np.array(times)
    for t in range(T):
        for i in range(N):
            try:
                pred_pos = preds[t, i, 1, 0:3]   # predicted next-step position
                actual_pos = pos[t+1, i, :]
                errors[t, i] = np.linalg.norm(actual_pos - pred_pos)
            except Exception:
                errors[t, i] = np.nan

    fig, ax = plt.subplots(2,1, figsize=(9,6), gridspec_kw={'height_ratios':[2,1]})
    for i in range(N):
        ax[0].plot(times_arr[:T], errors[:, i], label=f"dr{i}")
    ax[0].set_ylabel("1-step prediction error (m)")
    ax[0].set_title(title or "1-step prediction error vs time")
    ax[0].grid(True)
    ax[0].legend(fontsize='small')

    ax[1].boxplot([errors[:, i][~np.isnan(errors[:, i])] for i in range(N)], labels=[f"dr{i}" for i in range(N)])
    ax[1].set_ylabel("error (m)")
    ax[1].set_xlabel("drone id")
    ax[1].set_title("1-step prediction error distribution")
    ax[1].grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)



def plot_turbulence_forces(rec, outpath, title=None):
    """
    Plot applied turbulence force magnitude ||F|| for each drone vs time.
    Expects 'turb_forces' in rec or saved in npz.
    """
    turb = rec.get("turb_forces", None) or rec.get("turb_forces", None)
    if turb is None:
        # also check under metadata if stored there accidentally
        turb = safe_get_meta(rec, "turb_forces", None)
    if turb is None:
        print("[turb_forces] not found; skipping")
        return

    turb = np.asarray(turb)
    if turb.ndim != 3 or turb.shape[2] != 3:
        print("[turb_forces] unexpected shape; expected (T,N,3), got", turb.shape)
        return

    T, N, _ = turb.shape
    times = get_times(rec)
    if len(times) > T:
        times = times[:T]

    mags = np.linalg.norm(turb, axis=2)  # (T,N)

    fig, ax = plt.subplots(2,1, figsize=(9,6), gridspec_kw={'height_ratios':[2,1]})
    for i in range(N):
        ax[0].plot(times, mags[:, i], label=f"dr{i}")
    ax[0].set_ylabel("||F_turb|| (N)")
    ax[0].set_title(title or "Turbulence force magnitude vs time")
    ax[0].grid(True)
    ax[0].legend(fontsize='small')

    ax[1].boxplot([mags[:, i] for i in range(N)], labels=[f"dr{i}" for i in range(N)])
    ax[1].set_ylabel("||F_turb|| (N)")
    ax[1].set_xlabel("drone id")
    ax[1].set_title("Turbulence force distribution")
    ax[1].grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_single_step_pred_error(rec, outpath, title=None):
    """
    Computes and plots single-step prediction error:
        error(t,i) = || pos[t+1,i] - pred[t,i,1] ||

    rec: loaded .npz dict with keys 'pos' and 'dmpc_preds'
    outpath: path to save PNG
    """

    import numpy as np
    import matplotlib.pyplot as plt

    pos = rec.get("pos", None)
    preds = rec.get("dmpc_preds", None)

    if pos is None or preds is None:
        print("[single_step_pred_error] missing pos or dmpc_preds — skipping")
        return

    pos = np.asarray(pos)               # (T, N, 3)
    preds = np.asarray(preds)           # (T, N, Hp1, 6)

    T, N, _ = pos.shape
    _, _, Hp1, _ = preds.shape

    if Hp1 < 2:
        print("[single_step_pred_error] DMPC preds horizon too small")
        return

    # horizon step 1 = next-step prediction
    errors = np.zeros((T-1, N))

    for t in range(T-1):
        for i in range(N):
            pred_pos = preds[t, i, 1, 0:3]      # DMPC-predicted next position
            actual_pos = pos[t+1, i, :]         # actual next pos
            errors[t, i] = np.linalg.norm(actual_pos - pred_pos)

    times = rec.get("times", np.arange(T-1) * rec.get("metadata",{}).get("dt_ctrl", 0.0333))
    times = np.asarray(times)
    if times.shape[0] > errors.shape[0]:
        times = times[:errors.shape[0]]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9,5))

    for i in range(N):
        ax.plot(times, errors[:, i], label=f"dr{i}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title(title or "Predicted Position vs Actual Position Error")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_actual_vs_pred_error(rec, outpath, max_horizon=None, title=None):
    """
    Plots trajectory prediction error:
        error(t, h, i) = || pos[t+h, i] - dmpc_preds[t, i, h] ||

    rec: loaded .npz dict (must contain 'pos' and 'dmpc_preds')
    outpath: path to save PNG
    max_horizon: limit horizon plotted (default = full horizon)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    pos = rec.get("pos", None)
    preds = rec.get("dmpc_preds", None)

    if pos is None or preds is None:
        print("[actual_vs_pred_error] pos or dmpc_preds missing — skipping")
        return

    pos = np.asarray(pos)                   # shape (T, N, 3)
    preds = np.asarray(preds)               # shape (T, N, H+1, 6)

    T, N, _ = pos.shape
    _, _, Hp1, _ = preds.shape
    H = Hp1 - 1                              # prediction horizon

    if max_horizon is not None:
        H = min(H, max_horizon)

    # Times
    times = rec.get("times", np.arange(T) * rec.get("metadata",{}).get("dt_ctrl", 0.0333))

    # Main error array: shape (T-H, N, H)
    errors = np.zeros((T - H, N, H))

    for t in range(T - H):
        for i in range(N):
            for h in range(1, H+1):
                pred = preds[t, i, h, :3]            # predicted position at step h
                actual = pos[t + h, i, :]            # actual future position
                errors[t, i, h-1] = np.linalg.norm(actual - pred)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))

    horizon_labels = [f"h={h}" for h in range(1, H+1)]

    # Average error per horizon (across all drones)
    mean_err = np.mean(errors, axis=1)               # (T-H, H)

    for h in range(H):
        ax.plot(times[:T-H], mean_err[:, h], label=f"h={h+1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prediction error (m)")
    ax.set_title(title or "Prediction Error vs Time (Actual vs DMPC Predicted)")
    ax.grid(True)
    ax.legend(title="Horizon")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)



def _flatten_metric_per_drone(metric_list_per_drone):
    # metric_list_per_drone: list of lists (per-drone lists of numbers)
    # returns: list of arrays (each array: M_i values), plus pooled array
    per = []
    for l in metric_list_per_drone:
        try:
            arr = np.asarray(l, dtype=float)
        except Exception:
            arr = np.array([])
        per.append(arr)
    pooled = np.concatenate([a for a in per if a.size>0]) if any(a.size>0 for a in per) else np.array([])
    return per, pooled

def plot_solver_time_series(rec, outpath, key="solver_wall_times_per_drone", title=None):
    md = rec.get("metadata", {}) or {}
    st_per = md.get(key, None)
    if st_per is None:
        # try top-level key fallback
        st_per = rec.get("solver_wall_times", None)
    if st_per is None:
        print("[solver_time_series] solver times missing; skipping")
        return
    per, pooled = _flatten_metric_per_drone(st_per)
    N = len(per)
    fig, ax = plt.subplots(figsize=(9,4))
    for i, arr in enumerate(per):
        if arr.size==0: continue
        ax.plot(np.arange(arr.size), arr, label=f"dr{i}")
    ax.set_ylabel("solver time (s)")
    ax.set_xlabel("solve index")
    ax.set_title(title or "Solver wall time per solve (per drone)")
    ax.grid(True); ax.legend(fontsize='small')
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_solver_time_histogram(rec, outpath, key="solver_wall_times_per_drone", title=None):
    md = rec.get("metadata", {}) or {}
    st_per = md.get(key, None) or rec.get("solver_wall_times", None)
    if st_per is None:
        print("[solver_time_hist] solver times missing; skipping")
        return
    _, pooled = _flatten_metric_per_drone(st_per)
    fig, ax = plt.subplots(figsize=(6,4))
    if pooled.size == 0:
        print("[solver_time_hist] no data")
        return
    ax.hist(pooled[~np.isnan(pooled)], bins=50)
    ax.set_xlabel("solver time (s)"); ax.set_ylabel("count")
    ax.set_title(title or "Solver time histogram")
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_solver_time_percentiles(rec, outpath, key="solver_wall_times_per_drone", dt_ctrl_key="dt_ctrl", title=None):
    md = rec.get("metadata", {}) or {}
    st_per = md.get(key, None) or rec.get("solver_wall_times", None)
    dt_ctrl = md.get(dt_ctrl_key, None) or rec.get("metadata", {}).get("dt_ctrl", None) or 1.0/30.0
    if st_per is None:
        print("[solver_time_percentiles] missing; skipping")
        return
    _, pooled = _flatten_metric_per_drone(st_per)
    pooled = pooled[~np.isnan(pooled)]
    if pooled.size == 0:
        print("[solver_time_percentiles] no data")
        return
    pct = [50, 90, 95, 99]
    vals = np.percentile(pooled, pct)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar([str(p) for p in pct], vals)
    ax.axhline(dt_ctrl, linestyle='--', label=f"dt_ctrl={dt_ctrl:.3f}s")
    ax.set_ylabel("solver time (s)")
    ax.set_title(title or "Solver time percentiles")
    ax.legend(fontsize='small'); ax.grid(True)
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_realtime_violations(rec, outpath, key="solver_wall_times_per_drone", dt_ctrl_key="dt_ctrl", title=None):
    md = rec.get("metadata", {}) or {}
    st_per = md.get(key, None) or rec.get("solver_wall_times", None)
    dt_ctrl = md.get(dt_ctrl_key, None) or rec.get("metadata", {}).get("dt_ctrl", None) or 1.0/30.0
    if st_per is None:
        print("[rt_violations] missing; skipping")
        return
    per, _ = _flatten_metric_per_drone(st_per)
    N = len(per)
    # compute fraction exceeding, and plot stacked bar of fraction
    fracs = []
    for arr in per:
        if arr.size == 0:
            fracs.append(np.nan)
            continue
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            fracs.append(np.nan)
            continue
        fracs.append(np.mean(arr > dt_ctrl))
    fig, ax = plt.subplots(figsize=(6,3))
    ids = np.arange(N)
    ax.bar(ids, [0 if np.isnan(x) else x for x in fracs])
    ax.set_xticks(ids); ax.set_xticklabels([f"dr{i}" for i in ids])
    ax.set_ylim(0,1); ax.set_ylabel("fraction > dt_ctrl")
    ax.set_title(title or f"Fraction of solves exceeding dt_ctrl={dt_ctrl:.3f}s")
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_solver_time_vs_problem_size(rec, outpath, time_key="solver_wall_times_per_drone", size_key="solver_problem_size_per_drone", title=None):
    md = rec.get("metadata", {}) or {}
    times_per = md.get(time_key, None)
    size_per = md.get(size_key, None)
    if times_per is None or size_per is None:
        print("[time_vs_size] missing keys; skipping")
        return
    # flatten and pair; size_per entries might be lists of tuples (nvars,ncons)
    times_flat = []
    nvars_flat = []
    for drone_times, drone_sizes in zip(times_per, size_per):
        # ensure same length
        L = min(len(drone_times), len(drone_sizes))
        for i in range(L):
            try:
                t = float(drone_times[i])
                nvars = float(drone_sizes[i][0]) if hasattr(drone_sizes[i], '__len__') else np.nan
            except Exception:
                continue
            times_flat.append(t); nvars_flat.append(nvars)
    if len(times_flat) == 0:
        print("[time_vs_size] no paired data")
        return
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(nvars_flat, times_flat, s=6)
    ax.set_xlabel("n_vars"); ax.set_ylabel("solver time (s)")
    ax.set_title(title or "Solver time vs n_vars (scatter)")
    ax.grid(True)
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)


def plot_velocity_magnitude(rec, outpath, title=None):
    """
    Plot velocity magnitude (||v||) vs time for each drone.
    """
    vel = rec.get("vel", None)
    if vel is None:
        # try to derive from dmpc_preds if present (pred contains velocities in cols 3:6)
        preds = rec.get("dmpc_preds", None)
        if preds is not None:
            # use predicted initial velocity at each step as proxy (preds[t,i,0,3:6])
            try:
                T, N, _, _ = preds.shape
                vel = np.zeros((T, N, 3))
                for t in range(T):
                    for i in range(N):
                        vel[t, i, :] = preds[t, i, 0, 3:6]
            except Exception:
                vel = None

    if vel is None:
        print("[vel_mag] velocity data not found, skipping")
        return

    times = get_times(rec)
    # ensure shapes: vel -> (T,N,3)
    vel = np.asarray(vel)
    if vel.ndim != 3 or vel.shape[2] != 3:
        print("[vel_mag] unexpected vel shape, skipping")
        return
    T, N, _ = vel.shape

    mags = np.linalg.norm(vel, axis=2)  # (T,N)

    fig, ax = plt.subplots(2,1, figsize=(9,6), gridspec_kw={'height_ratios':[2,1]})
    for i in range(N):
        ax[0].plot(times, mags[:, i], label=f"dr{i}")
    ax[0].set_ylabel("||v|| (m/s)")
    ax[0].set_title(title or "Velocity magnitude vs time")
    ax[0].grid(True)
    ax[0].legend(fontsize='small')

    ax[1].boxplot([mags[:, i] for i in range(N)], labels=[f"dr{i}" for i in range(N)])
    ax[1].set_ylabel("||v|| (m/s)")
    ax[1].set_xlabel("drone id")
    ax[1].set_title("Velocity distribution")
    ax[1].grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_acceleration_magnitude(rec, outpath, title=None):
    """
    Compute acceleration from velocities (finite difference) and plot ||a|| vs time.
    acceleration times are aligned to the later timestamp (i.e., times[1:]) by default.
    """
    vel = rec.get("vel", None)
    if vel is None:
        # try to approximate from dmpc_preds velocities if available
        preds = rec.get("dmpc_preds", None)
        if preds is not None:
            try:
                T, N, _, _ = preds.shape
                vel = np.zeros((T, N, 3))
                for t in range(T):
                    for i in range(N):
                        vel[t, i, :] = preds[t, i, 0, 3:6]
            except Exception:
                vel = None

    if vel is None:
        print("[acc_mag] velocity data not found, skipping")
        return

    times = get_times(rec)
    vel = np.asarray(vel)
    if vel.ndim != 3 or vel.shape[2] != 3:
        print("[acc_mag] unexpected vel shape, skipping")
        return

    # compute dt array (robust to variable dt)
    times_arr = np.array(times)
    if times_arr.size < 2:
        print("[acc_mag] not enough time samples to compute acceleration, skipping")
        return
    dt_arr = np.diff(times_arr)  # length T-1
    # guard against zeros
    dt_arr[dt_arr == 0] = np.finfo(float).eps

    # finite difference for acceleration: a[t] = (v[t] - v[t-1]) / dt[t-1]
    # we'll compute accel at indices 1..T-1 and align times_acc = times[1:]
    T, N, _ = vel.shape
    acc = np.zeros((T-1, N))
    for t in range(1, T):
        dv = vel[t, :, :] - vel[t-1, :, :]  # shape (N,3)
        dt = dt_arr[t-1]
        a_vec = dv / dt
        acc[t-1, :] = np.linalg.norm(a_vec, axis=1)

    times_acc = times_arr[1:]

    fig, ax = plt.subplots(2,1, figsize=(9,6), gridspec_kw={'height_ratios':[2,1]})
    for i in range(N):
        ax[0].plot(times_acc, acc[:, i], label=f"dr{i}")
    ax[0].set_ylabel("||a|| (m/s²)")
    ax[0].set_title(title or "Acceleration magnitude vs time")
    ax[0].grid(True)
    ax[0].legend(fontsize='small')

    ax[1].boxplot([acc[:, i] for i in range(N)], labels=[f"dr{i}" for i in range(N)])
    ax[1].set_ylabel("||a|| (m/s²)")
    ax[1].set_xlabel("drone id")
    ax[1].set_title("Acceleration distribution")
    ax[1].grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_topdown_trajs(rec, outpath, title=None):
    pos = rec["pos"]
    if pos is None:
        print("[topdown] pos missing, skipping")
        return
    T, N, _ = pos.shape
    fig, ax = plt.subplots(figsize=(7,7))
    for i in range(N):
        traj = pos[:, i, :2]
        ax.plot(traj[:,0], traj[:,1], label=f"dr{i}")
        ax.scatter(traj[0,0], traj[0,1], marker='o', s=30)
        ax.scatter(traj[-1,0], traj[-1,1], marker='x', s=30)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title or "Top-down trajectories (x-y)")
    ax.grid(True); ax.axis('equal'); ax.legend(loc='best', fontsize='small')
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_3d_trajs(rec, outpath, title=None):
    pos = rec["pos"]
    if pos is None:
        print("[3d] pos missing, skipping")
        return
    T, N, _ = pos.shape
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        traj = pos[:, i, :]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label=f"dr{i}")
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], marker='o')
        ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], marker='x')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title or "3D trajectories")
    ax.legend(loc='best', fontsize='small')
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_predicted_vs_actual(rec, outpath, step_idx=0, title=None):
    pos = rec["pos"]
    dmpc_preds = rec["dmpc_preds"]
    if pos is None:
        print("[pred_vs_actual] pos missing, skipping")
        return
    if dmpc_preds is None:
        print("[pred_vs_actual] dmpc_preds missing, skipping")
        return
    T, N, _, _ = dmpc_preds.shape
    step_idx = min(max(0, step_idx), pos.shape[0]-1)
    fig, ax = plt.subplots(figsize=(7,7))
    for i in range(N):
        actual = pos[step_idx:, i, :2]
        ax.plot(actual[:,0], actual[:,1], '-', label=f"actual dr{i}")
        try:
            pred = dmpc_preds[step_idx, i, :, 0:2]
            ax.plot(pred[:,0], pred[:,1], '--', linewidth=1, label=f"pred dr{i}")
            ax.scatter(pred[0,0], pred[0,1], marker='s', s=20)
        except Exception:
            pass
        ax.scatter(pos[step_idx, i, 0], pos[step_idx, i, 1], marker='o', s=20)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title or f"Predicted vs actual (step {step_idx})")
    ax.axis('equal'); ax.grid(True); ax.legend(loc='best', fontsize='small')
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_tracking_error(rec, outpath, title=None):
    pos = rec["pos"]
    if pos is None:
        print("[tracking_error] pos missing, skipping")
        return
    T, N, _ = pos.shape
    times = get_times(rec)

    md = rec.get("metadata", {}) or {}
    targets = None
    if isinstance(md, dict):
        targets = md.get("targets_per_drone", md.get("target_pos", None))

    if targets is None:
        print("[tracking_error] targets not found, skipping")
        return

    targets = np.array(targets)
    errs = tracking_error_per_drone(pos, targets)

    fig, ax = plt.subplots(2,1, figsize=(9,6), gridspec_kw={'height_ratios':[2,1]})

    for i in range(N):
        ax[0].plot(times, errs[:,i], label=f"dr{i}")

    ax[0].set_ylabel("Position Error (m)")
    ax[0].set_title(title or "Tracking Error vs time")
    ax[0].grid(True)
    ax[0].legend(fontsize='small')

    ax[1].bar(np.arange(N), errs[-1])
    ax[1].set_xlabel("drone id")
    ax[1].set_ylabel("final error (m)")
    ax[1].grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_min_pairwise_distance(rec, outpath, title=None):
    """
    Plots minimum pairwise distance over time.
    Includes a hardcoded D_SAFE horizontal line.
    """

    # ---- Hardcode your D_SAFE here ----
    D_SAFE = 0.5    # <<< CHANGE THIS VALUE ANYTIME >>>

    pos = rec.get("pos", None)
    if pos is None:
        print("[min_pairwise] `pos` missing — skipping")
        return

    T, N, _ = pos.shape
    times = get_times(rec)

    mins = np.zeros(T)
    for t in range(T):
        mins[t] = pairwise_min_distance(pos[t])

    fig, ax = plt.subplots(figsize=(8,4))

    # main curve
    ax.plot(times, mins, linewidth=2, label="Interdrone Distance")

    # D_SAFE horizontal line
    ax.axhline(D_SAFE, color='r', linestyle='--', linewidth=1.8,
               label=f"D_safe = {D_SAFE:.2f} m")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(title or "Interdrone Distance vs Time")
    ax.grid(True)
    ax.legend(fontsize='small')

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)




def plot_control_effort(rec, outpath, title=None):
    dmpc = rec.get("dmpc_preds", None)
    rpms = rec.get("rpms", None)
    times = get_times(rec)

    if dmpc is None and rpms is None:
        print("[control_effort] missing data")
        return

    if dmpc is not None:
        T, N, _, _ = dmpc.shape
        dt = safe_get_meta(rec, "dt_ctrl", safe_get_meta(rec, "CTRL_TIMESTEP", 1.0/30.0))
        efforts = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                try:
                    v0 = dmpc[t, i, 0, 3:6]
                    v1 = dmpc[t, i, 1, 3:6]
                    a = (v1 - v0) / dt
                    efforts[t,i] = np.sum(a*a)
                except:
                    #efforts[t,i] = 0.0
                    efforts[t,i] = efforts[t-1,i-1]
    else:
        rpms = np.asarray(rpms)
        T, N, _ = rpms.shape
        efforts = np.sum(rpms**2, axis=2)

    fig, ax = plt.subplots(2,1, figsize=(9,6), gridspec_kw={'height_ratios':[2,1]})

    for i in range(N):
        ax[0].plot(times, efforts[:,i], label=f"dr{i}")

    ax[0].set_title(title or "Control effort (proxy) vs time")
    ax[0].set_ylabel("effort")
    ax[0].grid(True)
    ax[0].legend(fontsize='small')

    ax[1].boxplot([efforts[:,i] for i in range(N)], labels=[f"dr{i}" for i in range(N)])
    ax[1].set_ylabel("effort")
    ax[1].set_title("effort distribution")
    ax[1].grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_prediction_tracking_error(rec, outpath, lookahead=5, title=None):
    pos = rec.get("pos", None)
    preds = rec.get("dmpc_preds", None)
    if pos is None or preds is None:
        print("[prediction_tracking_error] missing, skipping")
        return

    times = get_times(rec)
    T, N, Pplus1, _ = preds.shape

    errors = []
    t_axis = []

    for t in range(0, T - lookahead):
        err_t = []
        for i in range(N):
            try:
                pred_pos = preds[t, i, lookahead, 0:3]
                actual_pos = pos[t+lookahead, i, :]
                err_t.append(np.linalg.norm(actual_pos - pred_pos))
            except:
                err_t.append(np.nan)
        errors.append(np.nanmean(err_t))
        t_axis.append(times[t])

    errors = np.array(errors)
    t_axis = np.array(t_axis)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(t_axis, errors)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"mean pred error (+{lookahead} steps)")
    ax.set_title(title or "Prediction accuracy vs time")
    ax.grid(True)

    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_solver_times(rec, outpath, key="solver_times", title=None):
    st = rec.get(key, None) or safe_get_meta(rec, key, None)
    if st is None:
        print(f"[solver_times] key '{key}' not found, skipping")
        return
    arr = np.asarray(st).flatten()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(arr[~np.isnan(arr)], bins=50)
    ax.set_xlabel("solver time (s)"); ax.set_ylabel("count")
    ax.set_title(title or "Solver time histogram")
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

def plot_slack_usage(rec, outpath, key="slack", title=None):
    slack = rec.get(key, None)
    if slack is None:
        slack = safe_get_meta(rec, key, None)
    if slack is None:
        print("[slack] not present, skipping")
        return
    slack = np.asarray(slack)
    if slack.ndim == 2:
        summed = np.sum(slack, axis=1)
    else:
        summed = np.sum(slack, axis=tuple(range(1, slack.ndim)))
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(summed); ax.set_xlabel("step"); ax.set_ylabel("sum slack"); ax.set_title(title or "Slack usage over time"); ax.grid(True)
    fig.savefig(outpath, bbox_inches='tight', dpi=200); plt.close(fig)

# ----------------------------
# Runner
# ----------------------------
DEFAULT_SET = [
    ("topdown", plot_topdown_trajs),
    ("3d", plot_3d_trajs),
    ("pred_vs_actual", plot_predicted_vs_actual),
    ("tracking_error", plot_tracking_error),
    ("min_pairwise", plot_min_pairwise_distance),
    ("control_effort", plot_control_effort),
    ("velocity_magnitude", plot_velocity_magnitude),     # <-- added
    ("acceleration_magnitude", plot_acceleration_magnitude),  # <-- added
    ("prediction_error", plot_prediction_tracking_error),
    # optional:
    ("solver_times", plot_solver_times),
    ("slack", plot_slack_usage),
]

DEFAULT_SET += [
    ("solver_time_series", plot_solver_time_series),
    ("solver_time_hist", plot_solver_time_histogram),
    ("solver_time_pct", plot_solver_time_percentiles),
    ("rt_violations", plot_realtime_violations),
    ("time_vs_size", plot_solver_time_vs_problem_size),
    ("actual_vs_pred_error", plot_actual_vs_pred_error),
    ("single_step_pred_error", plot_single_step_pred_error),
    ("turb_forces", plot_turbulence_forces),
    ("one_step_pred", plot_one_step_prediction_error)

]

import inspect
import traceback

def _rec_summary(rec):
    summary = {}
    for k, v in (rec.items() if isinstance(rec, dict) else []):
        try:
            arr = np.asarray(v)
            summary[k] = {"dtype": str(arr.dtype), "shape": arr.shape, "size": int(arr.size)}
        except Exception:
            summary[k] = {"type": type(v).__name__}
    return summary

def process_file(npz_path, outdir, do_all=False):
    rec = load_recording(npz_path)
    # small summary to help debugging
    print("\n=== Processing file:", npz_path)
    md = rec.get("metadata", {}) or {}
    print(" Metadata keys:", list(md.keys()) if isinstance(md, dict) else type(md))
    rec_summary = _rec_summary(rec)
    print(" Rec contents summary:")
    for k,v in rec_summary.items():
        print("  ", k, v)

    # compute dt_ctrl defensively
    dt_ctrl = md.get("dt_ctrl", None) if isinstance(md, dict) else None
    if dt_ctrl is None:
        try:
            dt_ctrl = rec.get("metadata", {}).get("dt_ctrl", 1.0/30.0)
        except Exception:
            dt_ctrl = 1.0/30.0

    # print solver-time quick stats if present
    times_md = md.get("solver_wall_times_per_drone", None) if isinstance(md, dict) else None
    if times_md is not None:
        try:
            pooled = np.concatenate([np.asarray(x) for x in times_md if len(x) > 0]) if any(len(x) > 0 for x in times_md) else np.array([])
            if pooled.size > 0:
                def pct(x): return np.percentile(pooled, x)
                print("Solver time stats (s): mean {:.4f}, median {:.4f}, p95 {:.4f}, max {:.4f}".format(
                    float(pooled.mean()), float(np.median(pooled)), float(pct(95)), float(pooled.max())))
                print("Fraction solves > dt_ctrl ({:.3f}s): {:.2%}".format(dt_ctrl, float(np.mean(pooled > dt_ctrl))))
            else:
                print("No solver time data found in metadata (arrays empty).")
        except Exception as e:
            print("Error while computing solver-time stats:", e)
    else:
        print("metadata has no solver_wall_times_per_drone")

    basename = Path(npz_path).stem
    print(f"Processing {npz_path} -> {basename} ...")
    ensure_dir(outdir)

    for shortname, fn in DEFAULT_SET:
        outpath = os.path.join(outdir, f"{basename}__{shortname}.png")
        print(f"\n-> Running plot: {shortname} -> {outpath}")
        # decide kwargs to pass by inspecting function signature
        try:
            sig = inspect.signature(fn)
            params = sig.parameters.keys()
            kwargs = {}

            # common optional params your plot functions use
            if "step_idx" in params:
                kwargs["step_idx"] = 0
            if "lookahead" in params:
                kwargs["lookahead"] = 5
            if "title" in params:
                kwargs["title"] = None
            if "key" in params and shortname in ("solver_times", "slack"):
                kwargs["key"] = "solver_times" if shortname == "solver_times" else "slack"
            if "max_horizon" in params:
                # for single-step we don't want horizon > 1; default to 1 if that is the user's intent,
                # but many functions use max_horizon differently — set to None here so fn uses default if any.
                kwargs["max_horizon"] = None

            # Attempt call: prefer calling with (rec, outpath, **kwargs)
            try:
                fn(rec, outpath, **kwargs)
                print("  wrote:", outpath)
            except TypeError as te:
                # fallback: try without kwargs (some functions have only (rec,outpath))
                print("  TypeError calling with kwargs, retrying without kwargs:", te)
                try:
                    fn(rec, outpath)
                    print("  wrote:", outpath)
                except Exception as e2:
                    print("  Failed calling function without kwargs. Traceback:")
                    traceback.print_exc()
            except Exception:
                print("  Exception while running plot function; full traceback:")
                traceback.print_exc()

        except Exception as e:
            print(f"  error preparing/running {shortname}: {e}")
            traceback.print_exc()

    print("\nDone processing", basename)




# def process_file(npz_path, outdir, do_all=False):
#     rec = load_recording(npz_path)
#     # after rec = load_recording(npz_path)
#     md = rec.get("metadata", {}) or {}
#     dt_ctrl = md.get("dt_ctrl", rec.get("metadata",{}).get("dt_ctrl", 1.0/30.0))
#     times = md.get("solver_wall_times_per_drone", None)
#     if times is not None:
#         pooled = np.concatenate([np.asarray(x) for x in times if len(x)>0]) if any(len(x)>0 for x in times) else np.array([])
#         if pooled.size>0:
#             #import numpy as np
#             def pct(x): return np.percentile(pooled, x)
#             print("Solver time stats (s): mean {:.4f}, median {:.4f}, p95 {:.4f}, max {:.4f}".format(
#                 pooled.mean(), np.median(pooled), pct(95), pooled.max()))
#             print("Fraction solves > dt_ctrl ({:.3f}s): {:.2%}".format(dt_ctrl, np.mean(pooled>dt_ctrl)))
#         else:
#             print("No solver time data found in metadata.")
#     else:
#         print("metadata has no solver_wall_times_per_drone")


#     basename = Path(npz_path).stem
#     print(f"Processing {npz_path} -> {basename} ...")
#     ensure_dir(outdir)
#     # plot main set
#     for shortname, fn in DEFAULT_SET:
#         # skip optional ones if data missing and not do_all
#         outpath = os.path.join(outdir, f"{basename}__{shortname}.png")
#         try:
#             if shortname == "pred_vs_actual":
#                 # default snapshot near start (safe)
#                 fn(rec, outpath, step_idx=0, title=None)
#             elif shortname == "prediction_error":
#                 fn(rec, outpath, lookahead=5, title=None)
#             elif shortname == "solver_times":
#                 # only run if solver times exist or do_all True
#                 if rec.get("solver_times", None) is None and not do_all:
#                     print("  solver_times not found; skipping")
#                     continue
#                 fn(rec, outpath, key="solver_times")
#             elif shortname == "slack":
#                 if rec.get("slack", None) is None and not do_all:
#                     print("  slack not found; skipping")
#                     continue
#                 fn(rec, outpath, key="slack")

#             elif shortname == "actual_vs_pred_error":
#                 fn(rec, outpath, max_horizon=10)

#             else:
#                 fn(rec, outpath)
#             print("  wrote:", outpath)
#         except Exception as e:
#             print(f"  error generating {shortname}: {e}")

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate plots from DMPC .npz recordings")
    parser.add_argument("files", nargs="+", help="one or more .npz files (glob allowed)")
    parser.add_argument("--outdir", default="figs", help="output directory for PNGs")
    parser.add_argument("--all", action="store_true", help="force generation of optional plots even if keys missing")
    args = parser.parse_args()
    # expand globs
    files = []
    for f in args.files:
        files.extend(sorted(glob.glob(f)))
    if len(files) == 0:
        print("No files found. Provide .npz file paths (globs allowed).")
        sys.exit(1)
    ensure_dir(args.outdir)
    for f in files:
        process_file(f, args.outdir, do_all=args.all)
    print("Done.")

if __name__ == "__main__":
    main()
