import numpy as np


class SwarmEnv:
    """
    Grid-based multi-agent environment for MARL (high-level planning).

    Grid values:
    0 = unexplored
    1 = explored
    2 = anomaly
    """

    ACTIONS = {
        0: (0, 1),    # UP
        1: (0, -1),   # DOWN
        2: (-1, 0),   # LEFT
        3: (1, 0),    # RIGHT
        4: (0, 0),    # STAY / INVESTIGATE
    }

    def __init__(self, grid_size=6, n_agents=2, max_steps=100):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        # Grid initialization
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Place anomaly
        self.anomaly_pos = (
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        )
        self.grid[self.anomaly_pos] = 2

        # Initialize agent positions
        self.agent_pos = {}
        for i in range(self.n_agents):
            self.agent_pos[i] = (
                np.random.randint(self.grid_size),
                np.random.randint(self.grid_size)
            )

        self.step_count = 0
        self.done = False

        return self._get_obs()

    def _get_obs(self):
        obs = {}
        for i in range(self.n_agents):
            obs[i] = {
                "self_pos": self.agent_pos[i],
                "grid": self.grid.copy()
            }
        return obs

    def step(self, actions):
        rewards = {}
        self.step_count += 1

        for i, action in actions.items():
            dx, dy = self.ACTIONS[action]
            x, y = self.agent_pos[i]

            # Move agent
            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)

            self.agent_pos[i] = (new_x, new_y)

            # Reward logic
            if self.grid[new_x, new_y] == 0:
                rewards[i] = 5.0  # new exploration
                self.grid[new_x, new_y] = 1

            elif self.grid[new_x, new_y] == 2:
                rewards[i] = 20.0  # anomaly found

            else:
                rewards[i] = -1.0  # revisiting

            # Small time penalty
            rewards[i] -= 0.1

        # Done conditions
        all_explored = np.all(self.grid != 0)
        timeout = self.step_count >= self.max_steps

        self.done = all_explored or timeout

        return self._get_obs(), rewards, self.done, {}

    def render(self):
        grid_display = self.grid.copy()

        for i, (x, y) in self.agent_pos.items():
            grid_display[x, y] = 9  # mark agent

        print("\nGrid:")
        print(grid_display)
        print("Agent positions:", self.agent_pos)