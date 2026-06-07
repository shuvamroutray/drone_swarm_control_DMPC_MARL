# ============================================================
# RUN_TRAINED_SWARM.py
# FINAL FIXED VERSION
# Load trained PPO policy on GPU1 and visualize full grid episode
# ============================================================

import os

# ============================================================
# FORCE GPU1 ONLY (set BEFORE torch/ray import)
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import ray
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from control_swarm.envs.marl_env.swarm_env_rllib_4 import SwarmEnvRLlib


# ============================================================
# GPU CHECK
# ============================================================
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))


# ============================================================
# INIT RAY
# ============================================================
ray.init(
    ignore_reinit_error=True,
    num_gpus=1,
)


# ============================================================
# REGISTER ENV
# ============================================================
env_name = "swarm_env"

register_env(
    env_name,
    lambda config: SwarmEnvRLlib(config)
)


# ============================================================
# PPO CONFIG
# MUST MATCH TRAINING ARCHITECTURE
# ============================================================
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
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        }
    )

    .framework("torch")

    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
)


# ============================================================
# BUILD ALGO
# ============================================================
algo = config.build_algo()


# ============================================================
# LOAD CHECKPOINT
# ============================================================
checkpoint_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "checkpoints")
)

print("\nLoading checkpoint from:")
print(checkpoint_path)

algo.restore(checkpoint_path)

print(f"\nLoaded checkpoint successfully from: {checkpoint_path}")


# ============================================================
# GET SHARED POLICY MODULE
# ============================================================
module = algo.get_module("shared_policy")


# ============================================================
# IMPORTANT:
# FORCE MODULE DEVICE CONSISTENCY
# ============================================================
device = next(module.parameters()).device

print("Module Device:", device)


# ============================================================
# CREATE ENV
# ============================================================
env = SwarmEnvRLlib({
    "grid_size": 6,
    "n_agents": 2,
    "max_steps": 100,
})


# ============================================================
# RESET
# ============================================================
obs, _ = env.reset()


# ============================================================
# GRID DISPLAY FUNCTION
# ============================================================
def display_grid(env):
    """
    Grid legend:
    0 = unexplored
    1 = explored
    2 = hidden anomaly
    3 = detected anomaly
    4 = resolved anomaly
    9 = agent_0
    10 = agent_1
    """

    grid = env.grid.copy()

    # Mark drone positions
    for idx, (agent, (x, y)) in enumerate(env.agent_pos.items()):
        grid[x, y] = 9 + idx

    print("\nGrid State:")
    print(grid)

    print("\nLegend:")
    print("0 = Unexplored")
    print("1 = Explored")
    print("2 = Hidden Anomaly")
    print("3 = Detected Anomaly")
    print("4 = Resolved Anomaly")
    print("9 = Agent_0")
    print("10 = Agent_1")

    print("\nAgent Positions:")
    print(env.agent_pos)

    if hasattr(env, "anomaly_pos"):
        print("True Anomaly Position (debug):", env.anomaly_pos)


# ============================================================
# START
# ============================================================
print("\n========== STARTING POLICY RUN ==========\n")

display_grid(env)


# ============================================================
# RUN ONE EPISODE
# ============================================================
for step in range(100):

    actions = {}

    # --------------------------------------------------------
    # COMPUTE ACTIONS
    # --------------------------------------------------------
    for agent_id, agent_obs in obs.items():

        # ----------------------------------------------------
        # CRITICAL FIX:
        # Send obs to SAME device as PPO module
        # ----------------------------------------------------
        input_tensor = torch.from_numpy(
            np.array([agent_obs], dtype=np.float32)
        ).to(device)

        input_dict = {
            "obs": input_tensor
        }

        with torch.no_grad():

            output = module.forward_inference(input_dict)

        # ----------------------------------------------------
        # RLlib compatibility
        # ----------------------------------------------------
        if "actions" in output:

            action = output["actions"][0].item()

        else:

            logits = output["action_dist_inputs"]

            # IMPORTANT:
            # Keep logits on module device for argmax
            action = torch.argmax(logits, dim=1)[0].item()

        actions[agent_id] = action

    # --------------------------------------------------------
    # ENV STEP
    # --------------------------------------------------------
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    # --------------------------------------------------------
    # PRINT STEP INFO
    # --------------------------------------------------------
    print("\n==================================================")
    print(f"STEP: {step}")
    print("ACTIONS:", actions)
    print("REWARDS:", rewards)
    print("TERMINATED:", terminateds["__all__"])
    print("TRUNCATED:", truncateds["__all__"])
    print("==================================================")

    # --------------------------------------------------------
    # DISPLAY GRID
    # --------------------------------------------------------
    display_grid(env)

    # Optional animation delay
    time.sleep(0.3)

    # --------------------------------------------------------
    # STOP CONDITIONS
    # --------------------------------------------------------
    if terminateds["__all__"]:

        print("\n========== MISSION COMPLETE ==========\n")
        break

    if truncateds["__all__"]:

        print("\n========== EPISODE TRUNCATED (TIMEOUT) ==========\n")
        break


# ============================================================
# CLEANUP
# ============================================================
ray.shutdown()

print("\n========== RUN FINISHED ==========\n")