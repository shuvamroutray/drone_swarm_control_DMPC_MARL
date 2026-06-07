# ============================================================
# TRAIN_SWARM_PPO.py
# RLlib PPO training for:
# Exploration + RL-only anomaly response
# GPU1 Optimized (L40S)
# ============================================================

import os
import logging

# ------------------------------------------------------------
# FORCE GPU1 ONLY (physical GPU1 becomes logical cuda:0)
# ------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from control_swarm.envs.marl_env.swarm_env_rllib_4 import SwarmEnvRLlib

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
        train_batch_size=32000,
        minibatch_size=4096,
        num_sgd_iter=10,
        lr=3e-4,

        gamma=0.99,
        lambda_=0.95,

        clip_param=0.2,

        vf_loss_coeff=0.5,
        entropy_coeff=0.02,

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


log_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "training_progress.log")
)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

print("Training log file:", log_file)
logging.info("==== TRAINING STARTED ====")

logging.info(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

logging.info("Environment: grid=6, agents=2, max_steps=100")
logging.info("Training: batch=32000, minibatch=4096, entropy=0.02")


# ============================================================
# TRAIN LOOP
# ============================================================

try:

    for i in range(1000):

        result = algo.train()

        reward = (
            result.get("episode_return_mean")
            or result.get("episode_reward_mean")
            or result.get("env_runners", {}).get("episode_return_mean")
            or result.get("env_runners", {}).get("episode_reward_mean")
        )

        
        episode_len = (
            result.get("episode_len_mean")
            or result.get("env_runners", {}).get("episode_len_mean")
            
        )

        reward = reward if reward is not None else "N/A"
        episode_len = episode_len if episode_len is not None else "N/A"

        msg = (
            f"Iteration: {i} | "
            f"Mean Reward: {reward} | "
            f"Episode Length: {episode_len} | "
            f"Env Steps: {result.get('num_env_steps_sampled_lifetime')}"
        )

        print("\n==============================")
        print(msg)
        print("==============================")

        logging.info(msg)


        if i > 0 and i % 20 == 0:
            checkpoint = algo.save(checkpoint_dir)
            cp_msg = f"Checkpoint saved at: {checkpoint}"
            print(cp_msg)
            logging.info(cp_msg)  

except Exception as e:

    logging.exception(f"TRAINING CRASHED: {e}")
    raise    

      
        


# ============================================================
# FINAL SAVE
# ============================================================
final_checkpoint = algo.save(checkpoint_dir)

print(f"\nFINAL CHECKPOINT: {final_checkpoint}")

logging.info(f"FINAL CHECKPOINT: {final_checkpoint}")
logging.info("==== TRAINING COMPLETE ====")



ray.shutdown()