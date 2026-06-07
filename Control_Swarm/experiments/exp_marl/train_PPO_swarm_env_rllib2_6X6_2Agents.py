# ============================================================
# TRAIN_SWARM_PPO.py
# RLlib PPO training for:
# Exploration + RL-only anomaly response
# GPU1 Optimized (L40S)
# ============================================================

import os

# ------------------------------------------------------------
# FORCE GPU1 ONLY (physical GPU1 becomes logical cuda:0)
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from control_swarm.envs.marl_env.swarm_env_rllib_2 import SwarmEnvRLlib

import torch

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
# ============================================================
# INIT RAY
# ============================================================
ray.init(
    ignore_reinit_error=True,
    num_gpus=1,   # only GPU1 visible
)


env_name = "swarm_env"

register_env(
    env_name,
    lambda config: SwarmEnvRLlib(config)
)


# ============================================================
# PPO CONFIG
# ============================================================
config = (
    PPOConfig()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------
    .environment(
        env=env_name,
        env_config={
            "grid_size": 6,
            "n_agents": 2,
            "max_steps": 100,
        }
    )

    # --------------------------------------------------------
    # Parallel rollout workers
    # --------------------------------------------------------
    .env_runners(
        num_env_runners=8,          # more CPU rollout workers
        num_envs_per_env_runner=4,  # better throughput
        rollout_fragment_length=100,
    )


    # --------------------------------------------------------
    # Resource Allocation
    # --------------------------------------------------------
    .resources(num_gpus=1)

    # --------------------------------------------------------
    # PPO Hyperparameters
    # --------------------------------------------------------
    .training(
        train_batch_size=16000,
        minibatch_size=2048,
        num_sgd_iter=10,
        lr=3e-4,

        gamma=0.99,
        lambda_=0.95,

        clip_param=0.2,

        vf_loss_coeff=0.5,
        entropy_coeff=0.01,

        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
    )
    # --------------------------------------------------------
    # Torch backend
    # --------------------------------------------------------
    .framework("torch")

    # --------------------------------------------------------
    # Shared policy
    # --------------------------------------------------------
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
)


# ============================================================
# BUILD ALGORITHM
# ============================================================
algo = config.build_algo()


# ============================================================
# CHECKPOINT DIR
# ============================================================
checkpoint_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "checkpoints")
)

os.makedirs(checkpoint_dir, exist_ok=True)

print("Saving checkpoints to:", checkpoint_dir)


# ============================================================
# TRAIN LOOP
# ============================================================
for i in range(500):

    result = algo.train()

    reward = (
        result.get("episode_return_mean")
        or result.get("episode_reward_mean")
        or result.get("env_runners", {}).get("episode_return_mean")
        or result.get("env_runners", {}).get("episode_reward_mean")
    )

    print(f"\n==============================")
    print(f"Iteration: {i}")
    print(f"Mean Reward: {reward}")
    print(
        "Env Steps:",
        result.get("num_env_steps_sampled_lifetime")
    )
    print(f"==============================")

    if i > 0 and i % 20 == 0:
        checkpoint = algo.save(checkpoint_dir)
        print(f"Checkpoint saved at: {checkpoint}")


# ============================================================
# FINAL SAVE
# ============================================================
final_checkpoint = algo.save(checkpoint_dir)

print(f"\nFINAL CHECKPOINT: {final_checkpoint}")

ray.shutdown()