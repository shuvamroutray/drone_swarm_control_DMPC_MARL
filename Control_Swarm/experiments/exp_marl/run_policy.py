import ray
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from control_swarm.envs.marl_env.swarm_env_rllib import SwarmEnvRLlib


# -------------------------
# Init Ray
# -------------------------
ray.init()

env_name = "swarm_env"

register_env(env_name, lambda config: SwarmEnvRLlib(config))


# -------------------------
# Same config as training
# -------------------------
config = (
    PPOConfig()
    .environment(env_name, env_config={
        "grid_size": 6,
        "n_agents": 2,
        "max_steps": 50,
    })
    .framework("torch")
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
)

algo = config.build_algo()

# -------------------------
# LOAD YOUR CHECKPOINT
# -------------------------
# Replace this path with your latest checkpoint path
checkpoint_path = "/home/shuvam/6_Drone_Swarm_DMPC_MARL/Control_Swarm/checkpoints"

algo.restore(checkpoint_path)


# -------------------------
# Create env
# -------------------------
env = SwarmEnvRLlib({
    "grid_size": 6,
    "n_agents": 2,
    "max_steps": 50
})

obs, _ = env.reset()

print("\n=== STARTING POLICY RUN ===\n")


# -------------------------
# Run one episode
# -------------------------
for step in range(50):

    actions = {}

    import torch
    import numpy as np

    module = algo.get_module("shared_policy")

    for agent_id, agent_obs in obs.items():
        input_dict = {
            "obs": torch.from_numpy(np.array([agent_obs])).float()
        }

        output = module.forward_inference(input_dict)

        # Handle both RLlib cases
        if "actions" in output:
            action = output["actions"][0].item()
        else:
            # fallback: sample from logits
            logits = output["action_dist_inputs"]
            action = torch.argmax(logits, dim=1)[0].item()

        actions[agent_id] = action
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    print(f"\nStep {step}")
    print("Actions:", actions)
    print("Rewards:", rewards)

    # Simple visualization
    grid = env.grid.copy()
    for agent, (x, y) in env.agent_pos.items():
        grid[x, y] = 9

    print("Grid:\n", grid)

    if terminateds["__all__"]:
        print("\n=== EPISODE DONE ===")
        break