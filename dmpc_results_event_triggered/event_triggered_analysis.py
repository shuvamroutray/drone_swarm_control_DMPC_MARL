#!/usr/bin/env python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = "./"

# =====================================================
# LOAD DATA
# =====================================================

records = []

expected_scenarios = 100
expected_runs = expected_scenarios * 2

for scenario_dir in sorted(glob.glob("scenario_*")):

    benchmark_files = glob.glob(
        os.path.join(
            scenario_dir,
            "*_benchmark.txt"
        )
    )

    for filename in benchmark_files:

        data = {}

        with open(filename, "r") as f:

            for line in f:

                line = line.strip()

                if "=" not in line:
                    continue

                key, value = line.split("=")

                try:
                    data[key] = float(value)
                except:
                    data[key] = value

        if len(data) > 0:
            records.append(data)

successful_runs = len(records)

# =====================================================
# SUCCESS RATE
# =====================================================

success_rate = (
    successful_runs /
    expected_runs
) * 100.0

print("\n====================================")
print("DATASET SUMMARY")
print("====================================")
print(f"Expected runs : {expected_runs}")
print(f"Successful runs : {successful_runs}")
print(f"Success rate : {success_rate:.2f}%")

# =====================================================
# CONVERT TO NUMPY
# =====================================================

def get_metric(metric):

    return np.array([
        r[metric]
        for r in records
        if metric in r
    ])

mission_time = get_metric("mission_time")
solve_count = get_metric("solve_count")
avg_solve_ms = get_metric("avg_solve_ms")
total_opt_ms = get_metric("total_optimization_time_ms")
path_length = get_metric("path_length")
min_separation = get_metric("min_separation")

goal_x = get_metric("goal_x")
goal_y = get_metric("goal_y")

goal_distance = np.sqrt(
    goal_x**2 +
    goal_y**2
)

# =====================================================
# SUMMARY TABLE
# =====================================================

metrics = {
    "mission_time": mission_time,
    "solve_count": solve_count,
    "avg_solve_ms": avg_solve_ms,
    "total_optimization_time_ms": total_opt_ms,
    "path_length": path_length,
    "min_separation": min_separation,
}

print("\n====================================")
print("STATISTICS")
print("====================================")

for name, values in metrics.items():

    print(f"\n{name}")

    print(
        f"Mean = {np.mean(values):.4f}"
    )

    print(
        f"Std  = {np.std(values):.4f}"
    )

    print(
        f"Min  = {np.min(values):.4f}"
    )

    print(
        f"Max  = {np.max(values):.4f}"
    )

# =====================================================
# SAVE REPORT
# =====================================================

with open("summary_report.txt", "w") as f:

    f.write("DMPC EVENT-TRIGGERED ANALYSIS\n\n")

    f.write(
        f"Expected Runs: {expected_runs}\n"
    )

    f.write(
        f"Successful Runs: {successful_runs}\n"
    )

    f.write(
        f"Success Rate: {success_rate:.2f}%\n\n"
    )

    for name, values in metrics.items():

        f.write(f"{name}\n")

        f.write(
            f"Mean={np.mean(values):.4f}\n"
        )

        f.write(
            f"Std={np.std(values):.4f}\n"
        )

        f.write(
            f"Min={np.min(values):.4f}\n"
        )

        f.write(
            f"Max={np.max(values):.4f}\n\n"
        )

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================

os.makedirs(
    "analysis_plots",
    exist_ok=True
)

# =====================================================
# HISTOGRAMS
# =====================================================

def save_histogram(
    data,
    title,
    filename,
    xlabel
):

    plt.figure(figsize=(7,5))

    plt.hist(
        data,
        bins=20
    )

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            "analysis_plots",
            filename
        )
    )

    plt.close()


save_histogram(
    mission_time,
    "Mission Time Distribution",
    "mission_time_hist.png",
    "Mission Time (s)"
)

save_histogram(
    solve_count,
    "Solve Count Distribution",
    "solve_count_hist.png",
    "Solve Count"
)

save_histogram(
    total_opt_ms,
    "Total Optimization Time",
    "optimization_time_hist.png",
    "Optimization Time (ms)"
)

save_histogram(
    path_length,
    "Path Length Distribution",
    "path_length_hist.png",
    "Path Length (m)"
)

# =====================================================
# SAFETY PLOT
# =====================================================

plt.figure(figsize=(7,5))

plt.hist(
    min_separation,
    bins=20
)

plt.axvline(
    0.5,
    linestyle='--',
    linewidth=2,
    label='Safety Limit'
)

plt.title(
    "Minimum Separation Distribution"
)

plt.xlabel(
    "Minimum Separation (m)"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "analysis_plots/min_separation_hist.png"
)

plt.close()

# =====================================================
# BOXPLOTS
# =====================================================

for name, data in metrics.items():

    plt.figure(figsize=(5,5))

    plt.boxplot(data)

    plt.title(name)

    plt.tight_layout()

    plt.savefig(
        f"analysis_plots/{name}_boxplot.png"
    )

    plt.close()

# =====================================================
# SCATTER:
# Solve Count vs Mission Time
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(
    solve_count,
    mission_time
)

plt.xlabel(
    "Solve Count"
)

plt.ylabel(
    "Mission Time (s)"
)

plt.title(
    "Mission Time vs Solve Count"
)

plt.tight_layout()

plt.savefig(
    "analysis_plots/solve_vs_time.png"
)

plt.close()

# =====================================================
# SCATTER:
# Goal Distance vs Solve Count
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(
    goal_distance,
    solve_count
)

plt.xlabel(
    "Goal Distance (m)"
)

plt.ylabel(
    "Solve Count"
)

plt.title(
    "Goal Distance vs Solve Count"
)

plt.tight_layout()

plt.savefig(
    "analysis_plots/goal_distance_vs_solves.png"
)

plt.close()

print("\n====================================")
print("Analysis complete.")
print("Plots saved in:")
print("analysis_plots/")
print("Summary saved as:")
print("summary_report.txt")
print("====================================")