#!/usr/bin/env python3
import numpy as np, sys
from pathlib import Path

def summarize_array(name, a):
    try:
        arr = np.asarray(a)
    except Exception as e:
        print(f"  {name}: cannot convert to ndarray ({e})")
        return
    print(f"  {name}: dtype={arr.dtype}, shape={arr.shape}, size={arr.size}")
    if arr.size > 0 and arr.size < 2000:
        print("    sample:", arr.flatten()[:10])
    if arr.size > 0:
        try:
            print("    min/max/nan_count:", np.nanmin(arr), np.nanmax(arr), np.sum(np.isnan(arr)))
        except Exception:
            pass

def load_and_inspect(path):
    print("Loading:", path)
    data = np.load(path, allow_pickle=True)
    print("Keys in .npz:", list(data.keys()))
    keys_of_interest = ["times", "pos", "dmpc_preds", "vel", "rpms", "metadata", "solver_times",
                        "solver_wall_times_per_drone", "solver_cpu_times_per_drone",
                        "solver_status_per_drone", "solver_iters_per_drone", "slack"]
    for k in keys_of_interest:
        if k in data:
            print(f"\nFound key: {k}")
            summarize_array(k, data[k])
        else:
            # try metadata field inside 'metadata' if present
            pass

    # Inspect metadata contents if present
    if "metadata" in data:
        md = data["metadata"]
        print("\nRaw metadata type:", type(md), "repr:")
        try:
            print(repr(md)[:1000])
        except Exception:
            pass
        # try to unpack if it's 0-d array
        if isinstance(md, np.ndarray) and md.size == 1:
            try:
                md0 = md.item()
                print("-> metadata.item() is dict? ", isinstance(md0, dict))
                if isinstance(md0, dict):
                    for kk, vv in md0.items():
                        print(f"   meta {kk}: type={type(vv)}, try summarize")
                        summarize_array(f"metadata[{kk}]", vv)
            except Exception as e:
                print("  metadata.item() failed:", e)
    else:
        print("\nNo metadata in file.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: inspect_recording.py <file.npz>")
        sys.exit(1)
    load_and_inspect(sys.argv[1])
