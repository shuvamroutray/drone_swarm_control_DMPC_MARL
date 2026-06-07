import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ray
import torch
import numpy as np
import matplotlib.pyplot as plt

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from control_swarm.envs.marl_env.swarm_env_rllib_4 import SwarmEnvRLlib

# ============================================================
# CONFIG
# ============================================================

NUM_EPISODES = 100

CHECKPOINT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "checkpoints"
    )
)

print("Checkpoint path:")
print(CHECKPOINT_PATH)

# ============================================================
# RAY
# ============================================================

ray.init(ignore_reinit_error=True, num_gpus=1)

# ============================================================
# ENV
# ============================================================

env_name = "swarm_env"

register_env(
    env_name,
    lambda config: SwarmEnvRLlib(config)
)

config = (
    PPOConfig()
    .environment(
        env=env_name,
        env_config={
            "grid_size": 6,
            "n_agents": 2,
            "max_steps": 100,
        }
    )
    .resources(num_gpus=1)
    .training(
        model={
            "fcnet_hiddens": [256,256],
            "fcnet_activation": "relu",
        }
    )
    .framework("torch")
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda aid,*a,**k:"shared_policy"
    )
)

algo = config.build_algo()

algo.restore(CHECKPOINT_PATH)

module = algo.get_module("shared_policy")

device = next(module.parameters()).device

print("Using device:", device)

# ============================================================
# STORAGE
# ============================================================

all_t100 = []

all_redundancy = []

all_avg_separation = []

all_coverage_curves = []

mission_lengths = []

# ============================================================
# EPISODES
# ============================================================

for ep in range(NUM_EPISODES):

    env = SwarmEnvRLlib({
        "grid_size":6,
        "n_agents":2,
        "max_steps":100,
    })

    obs,_ = env.reset()

    coverage_history = []

    separation_history = []

    cell_visit_counter = {}

    total_visits = 0

    repeated_visits = 0

    t100 = None

    # --------------------------------------------------------
    # EPISODE LOOP
    # --------------------------------------------------------

    for step in range(env.max_steps):

        actions = {}

        for agent_id, agent_obs in obs.items():

            input_tensor = torch.from_numpy(
                np.array([agent_obs], dtype=np.float32)
            ).to(device)

            with torch.no_grad():

                output = module.forward_inference(
                    {"obs": input_tensor}
                )

            if "actions" in output:

                action = output["actions"][0].item()

            else:

                logits = output["action_dist_inputs"]

                action = torch.argmax(
                    logits,
                    dim=1
                )[0].item()

            actions[agent_id] = action

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # =====================================================
        # COVERAGE
        # =====================================================

        explored_cells = np.sum(
            (env.grid == 1) |
            (env.grid == 3) |
            (env.grid == 4)
        )

        coverage = (
            explored_cells /
            (env.grid_size * env.grid_size)
        ) * 100

        coverage_history.append(coverage)

        if coverage >= 100 and t100 is None:
            t100 = step + 1

        # =====================================================
        # REDUNDANCY
        # =====================================================

        for agent in env.agents:

            pos = env.agent_pos[agent]

            total_visits += 1

            if pos in cell_visit_counter:

                repeated_visits += 1

                cell_visit_counter[pos] += 1

            else:

                cell_visit_counter[pos] = 1

        # =====================================================
        # SEPARATION
        # =====================================================

        agents = list(env.agents)

        x1, y1 = env.agent_pos[agents[0]]
        x2, y2 = env.agent_pos[agents[1]]

        dist = abs(x1 - x2) + abs(y1 - y2)

        separation_history.append(dist)

        # =====================================================
        # TERMINATION
        # =====================================================

        if terminateds["__all__"]:

            mission_lengths.append(step + 1)

            break

        if truncateds["__all__"]:

            mission_lengths.append(step + 1)

            break

    # =========================================================
    # EPISODE METRICS
    # =========================================================

    if t100 is None:
        t100 = env.max_steps

    redundancy = (
        repeated_visits /
        max(total_visits,1)
    ) * 100

    avg_sep = np.mean(separation_history)

    all_t100.append(t100)

    all_redundancy.append(redundancy)

    all_avg_separation.append(avg_sep)

    all_coverage_curves.append(
        coverage_history
    )

    print(
        f"Episode {ep+1:03d} | "
        f"T100={t100:.1f} | "
        f"Redundancy={redundancy:.2f}% | "
        f"AvgSep={avg_sep:.2f}"
    )

# ============================================================
# ALIGN CURVES
# ============================================================

max_len = max(
    len(curve)
    for curve in all_coverage_curves
)

aligned_curves = []

for curve in all_coverage_curves:

    padded = curve + [curve[-1]] * (
        max_len - len(curve)
    )

    aligned_curves.append(padded)

aligned_curves = np.array(aligned_curves)

mean_curve = np.mean(
    aligned_curves,
    axis=0
)

std_curve = np.std(
    aligned_curves,
    axis=0
)

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("="*60)
print("EXPLORATION EFFICIENCY ANALYSIS")
print("="*60)

print(
    f"Mean T100               : "
    f"{np.mean(all_t100):.2f}"
)

print(
    f"Std T100                : "
    f"{np.std(all_t100):.2f}"
)

print(
    f"Mean Redundancy (%)     : "
    f"{np.mean(all_redundancy):.2f}"
)

print(
    f"Std Redundancy (%)      : "
    f"{np.std(all_redundancy):.2f}"
)

print(
    f"Mean Agent Separation   : "
    f"{np.mean(all_avg_separation):.2f}"
)

print(
    f"Std Agent Separation    : "
    f"{np.std(all_avg_separation):.2f}"
)

print(
    f"Mean Mission Length     : "
    f"{np.mean(mission_lengths):.2f}"
)

print("="*60)

# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "exploration_metrics.txt",
    "w"
) as f:

    f.write(
        f"Mean T100: {np.mean(all_t100):.4f}\n"
    )

    f.write(
        f"Std T100: {np.std(all_t100):.4f}\n"
    )

    f.write(
        f"Mean Redundancy: "
        f"{np.mean(all_redundancy):.4f}\n"
    )

    f.write(
        f"Mean Agent Separation: "
        f"{np.mean(all_avg_separation):.4f}\n"
    )

# ============================================================
# COVERAGE CURVE
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    mean_curve,
    linewidth=3,
    label="Mean Coverage"
)

plt.fill_between(
    np.arange(max_len),
    mean_curve - std_curve,
    mean_curve + std_curve,
    alpha=0.2
)

plt.xlabel("Time Step")
plt.ylabel("Coverage (%)")
plt.title("Coverage Progress During Exploration")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "coverage_progress_curve.png",
    dpi=600
)

plt.show()

ray.shutdown()


def select_action(obs, policy_type):

    actions = {}

    if policy_type == "random":

        for agent_id in obs.keys():
            actions[agent_id] = np.random.randint(5)

        return actions

    elif policy_type == "ppo":

        for agent_id, agent_obs in obs.items():

            input_tensor = torch.from_numpy(
                np.array([agent_obs], dtype=np.float32)
            ).to(device)

            with torch.no_grad():
                output = module.forward_inference(
                    {"obs": input_tensor}
                )

            if "actions" in output:
                action = output["actions"][0].item()

            else:
                logits = output["action_dist_inputs"]
                action = torch.argmax(
                    logits,
                    dim=1
                )[0].item()

            actions[agent_id] = action

        return actions
    

def evaluate_policy(policy_type, num_episodes=100):

        all_t100 = []
        all_redundancy = []
        all_avg_sep = []
        all_mission_lengths = []
        all_coverage_curves = []

        for ep in range(num_episodes):

            np.random.seed(ep)

            env = SwarmEnvRLlib({
                "grid_size": 6,
                "n_agents": 2,
                "max_steps": 100,
            })

            obs, _ = env.reset(seed=ep)

            coverage_history = []
            separation_history = []

            cell_visit_counter = {}

            total_visits = 0
            repeated_visits = 0

            t100 = None

            for step in range(env.max_steps):

                actions = select_action(
                    obs,
                    policy_type
                )

                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                # =================================================
                # COVERAGE
                # =================================================

                explored_cells = np.sum(
                    (env.grid == 1) |
                    (env.grid == 3) |
                    (env.grid == 4)
                )

                coverage = (
                    explored_cells /
                    (env.grid_size * env.grid_size)
                ) * 100

                coverage_history.append(coverage)

                if coverage >= 100 and t100 is None:
                    t100 = step + 1

                # =================================================
                # REDUNDANCY
                # =================================================

                for agent in env.agents:

                    pos = env.agent_pos[agent]

                    total_visits += 1

                    if pos in cell_visit_counter:
                        repeated_visits += 1
                        cell_visit_counter[pos] += 1
                    else:
                        cell_visit_counter[pos] = 1

                # =================================================
                # SEPARATION
                # =================================================

                agents = list(env.agents)

                x1, y1 = env.agent_pos[agents[0]]
                x2, y2 = env.agent_pos[agents[1]]

                dist = abs(x1 - x2) + abs(y1 - y2)

                separation_history.append(dist)

                if (
                    terminateds["__all__"]
                    or truncateds["__all__"]
                ):
                    break

            if t100 is None:
                t100 = env.max_steps

            redundancy = (
                repeated_visits /
                max(total_visits, 1)
            ) * 100

            all_t100.append(t100)

            all_redundancy.append(redundancy)

            all_avg_sep.append(
                np.mean(separation_history)
            )

            all_mission_lengths.append(
                len(coverage_history)
            )

            all_coverage_curves.append(
                coverage_history
            )

        return {
            "t100": np.array(all_t100),
            "redundancy": np.array(all_redundancy),
            "avg_sep": np.array(all_avg_sep),
            "mission_length": np.array(all_mission_lengths),
            "coverage_curves": all_coverage_curves,
        }

print("\nEvaluating PPO...")
ppo = evaluate_policy("ppo", 100)

print("\nEvaluating Random...")
rnd = evaluate_policy("random", 100)

print("\n")
print("="*80)
print("PPO vs RANDOM")
print("="*80)

print(
    f"T100: "
    f"{np.mean(ppo['t100']):.2f} ± {np.std(ppo['t100']):.2f}"
    f" | "
    f"{np.mean(rnd['t100']):.2f} ± {np.std(rnd['t100']):.2f}"
)

print(
    f"Redundancy: "
    f"{np.mean(ppo['redundancy']):.2f}% ± {np.std(ppo['redundancy']):.2f}"
    f" | "
    f"{np.mean(rnd['redundancy']):.2f}% ± {np.std(rnd['redundancy']):.2f}"
)

print(
    f"Avg Separation: "
    f"{np.mean(ppo['avg_sep']):.2f} ± {np.std(ppo['avg_sep']):.2f}"
    f" | "
    f"{np.mean(rnd['avg_sep']):.2f} ± {np.std(rnd['avg_sep']):.2f}"
)

print(
    f"Mission Length: "
    f"{np.mean(ppo['mission_length']):.2f} ± {np.std(ppo['mission_length']):.2f}"
    f" | "
    f"{np.mean(rnd['mission_length']):.2f} ± {np.std(rnd['mission_length']):.2f}"
)


def mean_curve(curves):

    max_len = max(len(c) for c in curves)

    padded = []

    for c in curves:
        padded.append(
            c + [c[-1]]*(max_len-len(c))
        )

    padded = np.array(padded)

    return (
        np.mean(padded, axis=0),
        np.std(padded, axis=0)
    )

ppo_mean, ppo_std = mean_curve(
    ppo["coverage_curves"]
)

rnd_mean, rnd_std = mean_curve(
    rnd["coverage_curves"]
)

plt.figure(figsize=(10,6))

plt.plot(
    ppo_mean,
    linewidth=3,
    label="PPO"
)

plt.fill_between(
    np.arange(len(ppo_mean)),
    ppo_mean-ppo_std,
    ppo_mean+ppo_std,
    alpha=0.2
)

plt.plot(
    rnd_mean,
    linewidth=3,
    label="Random"
)

plt.fill_between(
    np.arange(len(rnd_mean)),
    rnd_mean-rnd_std,
    rnd_mean+rnd_std,
    alpha=0.2
)

plt.xlabel("Time Step")
plt.ylabel("Coverage (%)")
plt.title("Coverage Progress Comparison")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "ppo_vs_random_coverage.png",
    dpi=600
)
    

