#!/usr/bin/env python3

import numpy as np
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from control_swarm.envs.marl_env.swarm_env_rllib_4 import SwarmEnvRLlib


class PolicyLoader:

    def __init__(self, checkpoint_path, grid_size, n_agents):

        env_name = "swarm_env"

        register_env(env_name, lambda config: SwarmEnvRLlib(config))

        config = (
            PPOConfig()
            .environment(env_name, env_config={
                "grid_size": grid_size,
                "n_agents": n_agents
            })
            .framework("torch")
            .multi_agent(
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
            )
        )

        self.algo = config.build_algo()
        self.algo.restore(checkpoint_path)

        self.module = self.algo.get_module("shared_policy")

    def predict(self, obs):

        input_dict = {
            "obs": torch.from_numpy(np.array([obs])).float()
        }

        output = self.module.forward_inference(input_dict)

        if "actions" in output:
            return output["actions"][0].item()

        logits = output["action_dist_inputs"]
        return torch.argmax(logits, dim=1)[0].item()