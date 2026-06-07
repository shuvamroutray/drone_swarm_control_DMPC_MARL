import random
from control_swarm.envs.marl_env.grid_swarm_env import SwarmEnv


env = SwarmEnv(grid_size=6, n_agents=2)

obs = env.reset()

for step in range(20):
    actions = {i: random.randint(0, 4) for i in range(2)}

    obs, rewards, done, _ = env.step(actions)

    print(f"\nStep {step}")
    print("Actions:", actions)
    print("Rewards:", rewards)

    env.render()

    if done:
        print("\nEpisode finished!")
        break