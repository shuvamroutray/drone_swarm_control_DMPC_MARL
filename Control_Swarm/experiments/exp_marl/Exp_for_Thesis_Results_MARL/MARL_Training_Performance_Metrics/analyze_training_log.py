import re
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

LOG_FILE = "training_progress.log"
SMOOTH_WINDOW = 20

# =====================================================
# PARSE LOG
# =====================================================

pattern = re.compile(
    r"Iteration:\s*(\d+)\s*\|\s*"
    r"Mean Reward:\s*([-+]?\d*\.?\d+)\s*\|\s*"
    r"Episode Length:\s*([-+]?\d*\.?\d+)\s*\|\s*"
    r"Env Steps:\s*([-+]?\d*\.?\d+)"
)

raw_data = []

with open(LOG_FILE, "r") as f:
    for line in f:

        match = pattern.search(line)

        if match:

            iteration = int(match.group(1))
            reward = float(match.group(2))
            ep_len = float(match.group(3))
            env_steps = float(match.group(4))

            raw_data.append(
                (
                    iteration,
                    reward,
                    ep_len,
                    env_steps
                )
            )

print(f"\nTotal log entries found: {len(raw_data)}")

# =====================================================
# SPLIT INTO TRAINING RUNS
# =====================================================

runs = []
current_run = []

prev_iter = -1

for row in raw_data:

    iteration = row[0]

    if iteration < prev_iter:
        runs.append(current_run)
        current_run = []

    current_run.append(row)

    prev_iter = iteration

if current_run:
    runs.append(current_run)

print(f"Detected {len(runs)} training run(s)")


# =====================================================
# USE RUN WITH MAX ITERATIONS
# =====================================================

run_max_iters = [
    max(row[0] for row in run)
    for run in runs
]

largest_run_idx = np.argmax(run_max_iters)

data = runs[largest_run_idx]

print("\nRun summary:")

for i, run in enumerate(runs):

    start_iter = run[0][0]
    end_iter = run[-1][0]

    print(
        f"Run {i+1}: "
        f"start={start_iter}, "
        f"end={end_iter}, "
        f"entries={len(run)}"
    )

print(
    f"\nUsing Run {largest_run_idx+1} "
    f"(highest iteration reached = "
    f"{run_max_iters[largest_run_idx]})"
)
# =====================================================
# CONVERT TO ARRAYS
# =====================================================

iterations = np.array([x[0] for x in data])
rewards = np.array([x[1] for x in data])
episode_lengths = np.array([x[2] for x in data])
env_steps = np.array([x[3] for x in data])

# =====================================================
# MOVING AVERAGE
# =====================================================

def moving_average(x, window):

    return np.convolve(
        x,
        np.ones(window) / window,
        mode="valid"
    )

reward_smooth = moving_average(
    rewards,
    SMOOTH_WINDOW
)

episode_smooth = moving_average(
    episode_lengths,
    SMOOTH_WINDOW
)

smooth_iters = iterations[
    SMOOTH_WINDOW - 1:
]

# =====================================================
# STATISTICS
# =====================================================

initial_reward = rewards[0]
final_reward = rewards[-1]

max_reward = np.max(rewards)
min_reward = np.min(rewards)

reward_improvement = (
    (final_reward - initial_reward)
    / abs(initial_reward)
) * 100

reward_mean_final = np.mean(
    rewards[-100:]
)

reward_std_final = np.std(
    rewards[-100:]
)

initial_ep_len = episode_lengths[0]
final_ep_len = episode_lengths[-1]

episode_reduction = (
    (initial_ep_len - final_ep_len)
    / initial_ep_len
) * 100

total_env_steps = env_steps[-1]

# =====================================================
# CONVERGENCE ITERATION
# =====================================================

threshold = 0.95 * max_reward

convergence_iteration = None

for i in range(len(rewards)):

    if rewards[i] >= threshold:
        convergence_iteration = iterations[i]
        break

# =====================================================
# PRINT REPORT
# =====================================================

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"Iterations              : {len(iterations)}")

print(f"\nInitial Reward          : {initial_reward:.2f}")
print(f"Final Reward            : {final_reward:.2f}")
print(f"Maximum Reward          : {max_reward:.2f}")
print(f"Minimum Reward          : {min_reward:.2f}")

print(
    f"\nReward Improvement (%)  : "
    f"{reward_improvement:.2f}"
)

print(
    f"\nFinal Reward Mean       : "
    f"{reward_mean_final:.2f}"
)

print(
    f"Final Reward Std        : "
    f"{reward_std_final:.2f}"
)

print(
    f"\nInitial Episode Length  : "
    f"{initial_ep_len:.2f}"
)

print(
    f"Final Episode Length    : "
    f"{final_ep_len:.2f}"
)

print(
    f"Episode Reduction (%)   : "
    f"{episode_reduction:.2f}"
)

print(
    f"\nTotal Env Steps         : "
    f"{total_env_steps:,.0f}"
)

print(
    f"Convergence Iteration   : "
    f"{convergence_iteration}"
)

print("=" * 60)

# =====================================================
# REWARD CURVE
# =====================================================

# plt.figure(figsize=(12, 6))

# plt.plot(
#     iterations,
#     rewards,
#     color='tab:blue',
#     alpha=0.7,
#     linewidth=1.5,
#     label="Raw Reward"
# )

# plt.plot(
#     smooth_iters,
#     reward_smooth,
#     color='darkorange',
#     linewidth=3,
#     label=f"{SMOOTH_WINDOW}-Iter Moving Avg"
# )

# plt.xlabel("Training Iteration")
# plt.ylabel("Mean Episode Reward")
# plt.title("PPO Reward Convergence")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     "reward_convergence.png",
#     dpi=300
# )


plt.figure(figsize=(12,6))

plt.plot(
    iterations,
    rewards,
    color='tab:blue',
    alpha=0.8,
    linewidth=1.2,
    label='Mean Reward'
)

plt.plot(
    smooth_iters,
    reward_smooth,
    color='darkorange',
    linewidth=3.5,
    label=f'{SMOOTH_WINDOW}-Iteration Moving Average'
)

plt.xlabel("Training Iteration", fontsize=14)
plt.ylabel("Mean Episode Reward", fontsize=14)
plt.title("PPO Training Reward Convergence", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(fontsize=12)

plt.tight_layout()

plt.savefig(
    "reward_convergence.png",
    dpi=600,
    bbox_inches='tight'
)

# =====================================================
# EPISODE LENGTH CURVE
# =====================================================

plt.figure(figsize=(12, 6))

plt.plot(
    iterations,
    episode_lengths,
    alpha=0.25,
    linewidth=1,
    label="Raw Length"
)

plt.plot(
    smooth_iters,
    episode_smooth,
    linewidth=3,
    label=f"{SMOOTH_WINDOW}-Iter Moving Avg"
)

plt.xlabel("Training Iteration")
plt.ylabel("Episode Length")
plt.title("Episode Length Convergence")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "episode_length_convergence.png",
    dpi=300
)

# =====================================================
# REWARD VS STEPS
# =====================================================

plt.figure(figsize=(12, 6))

plt.plot(
    env_steps,
    rewards,
    alpha=0.25,
    linewidth=1
)

plt.xlabel("Environment Steps")
plt.ylabel("Mean Reward")
plt.title("Reward vs Environment Interactions")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reward_vs_steps.png",
    dpi=300
)

# =====================================================
# REWARD DISTRIBUTION
# =====================================================

plt.figure(figsize=(10, 5))

plt.hist(
    rewards,
    bins=30
)

plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution")

plt.tight_layout()

plt.savefig(
    "reward_distribution.png",
    dpi=300
)

# =====================================================
# STABILITY PLOT
# =====================================================

rolling_std = []

window = 50

for i in range(window, len(rewards)):

    rolling_std.append(
        np.std(
            rewards[i-window:i]
        )
    )

plt.figure(figsize=(12, 6))

plt.plot(
    iterations[window:],
    rolling_std
)

plt.xlabel("Training Iteration")
plt.ylabel("Reward Std Dev")
plt.title("Reward Stability During Training")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reward_stability.png",
    dpi=300
)

plt.show()