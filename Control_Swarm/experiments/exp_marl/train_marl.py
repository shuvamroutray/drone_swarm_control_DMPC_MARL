import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from control_swarm.envs.marl_env.swarm_env_rllib import SwarmEnvRLlib
import os

ray.init()    # distributed computing framework. worker management + distributed execution

env_name = "swarm_env"

register_env(env_name, lambda config: SwarmEnvRLlib(config))

"""
PPO Configuration
What world?
How many workers?
How to train?
Which framework?
How many policies?


Overall structure 
config = (
        PPOConfig()--> Creates PPO algorithm template.Policy gradient + clipping + value function + entropy + optimizer defaults
        .environment(...)--> What world PPO trains in. env_config is passed onto env_name which is SwarmEnvRllib
        .env_runners(...)--> Defines data collection parallelism. 
                num_env_runners=4 --> Create 4 rollout workers. 
                    Each worker:
                        runs environments
                        collects trajectories

                num_envs_per_env_runner=2
                    Each worker runs 2 env copies

            Total environments: 4X2=8 simultaneous environments

        .training(...)
            train_batch_size=4000 -->Before PPO updates, collect 4000 timesteps total
            minibatch_size=256 --> splits into chunks of 250 for stochastic gradient descent
            num_sgd_iter=10 --> Use same batch. 10 optimization passes(loops over batch of 256 for 10 times)
            lr= lr=3e-4--> learning rate


        .framework(...) --> "torch" use pytorch backend

        .multi_agent(...) --> for multiagent PPO
            policy_mapping_fn--> One policy exist. Policy shared among all the agents.
    )
    

"""
config = (
    PPOConfig()
    .environment(env_name, env_config={
        "grid_size": 6,
        "n_agents": 2,
        "max_steps": 100,
    })
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=2
    )
    .training(
        train_batch_size=4000,
        minibatch_size=256,     
        num_sgd_iter=10,
        lr=3e-4
    )
    .framework("torch")
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
)

algo = config.build() # creates PPO algorithm. Create environment runners (4X2=8 envs). Setup the entire PPO pipeline


for i in range(100): # run for 100 training iterations
    result = algo.train()

    print(f"Iteration {i}")

    # Debug: print available keys
    print("Available keys:", result.keys())

    # Try safe access
    if "env_runners" in result:
        print("Env runner keys:", result["env_runners"].keys())

    # Try multiple possible reward keys
    reward = (
        result.get("episode_return_mean")
        or result.get("episode_reward_mean")
        or result.get("env_runners", {}).get("episode_return_mean")
        or result.get("env_runners", {}).get("episode_reward_mean")
    )

    print("Reward:", reward)

    if i % 10 == 0:
   

        checkpoint_dir = os.path.abspath("checkpoints")

        checkpoint = algo.save(checkpoint_dir)
        print("Checkpoint saved at", checkpoint)